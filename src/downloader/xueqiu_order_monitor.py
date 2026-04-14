"""雪球讨论 — 基本面信息一站式采集管线

单只股票完整流程:
    爬取当天评论 → LLM 过滤 → LLM 生成 wiki → 保存 → 下一只

过滤重点:
    基本面变化、产品订单、产品涨价/降价、产能/产量、
    营收利润、行业政策、管理层变动

用法:
    # 单只
    python -m src.downloader.xueqiu_order_monitor --codes 600519
    # 全量
    python -m src.downloader.xueqiu_order_monitor \
        --stock-csv /path/to/tushare_stock_basic.csv --resume
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, date
from pathlib import Path

import requests
from DrissionPage import ChromiumPage, ChromiumOptions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def _load_env_file():
    """从项目根目录 .env 文件加载环境变量 (不覆盖已有的)"""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file()

# ── 默认 LLM 配置 ─────────────────────────────────────────
DEFAULT_LLM_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
DEFAULT_LLM_MODEL = "qwen3.6-plus"

# ── 雪球 API ───────────────────────────────────────────────
XUEQIU_HOME = "https://xueqiu.com"
XUEQIU_STOCK_COMMENT_URL = "https://xueqiu.com/query/v1/symbol/search/status.json"


def _to_xueqiu_symbol(code: str) -> str:
    """将 A 股代码转为雪球格式, 如 600519 -> SH600519"""
    code = code.strip()
    if code.startswith(("SH", "SZ", "sh", "sz")):
        return code.upper()
    if code.startswith(("6", "9")):
        return f"SH{code}"
    elif code.startswith(("0", "3", "2")):
        return f"SZ{code}"
    return code.upper()


def _ts_to_str(ts) -> str:
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def _ts_to_date(ts) -> date | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts / 1000).date()
    except Exception:
        return None


def _clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text) if text else ""


# ═══════════════════════════════════════════════════════════
#  浏览器会话 (WAF 自适应)
# ═══════════════════════════════════════════════════════════

class XueqiuSession:
    """用 DrissionPage 管理雪球会话, 自动通过 WAF 并处理限流"""

    _WAF_KEYWORDS = ("Access Ver", "TIME: 20", "访问验证", "请完成验证", "Sorry, you")

    def __init__(self):
        self.page = None
        self._start_browser()

    def _start_browser(self):
        if self.page is not None:
            try:
                self.page.quit()
            except Exception:
                pass
        co = ChromiumOptions()
        co.set_browser_path("/usr/bin/chromium-browser")
        co.headless()
        co.set_argument("--no-sandbox")
        co.set_argument("--disable-dev-shm-usage")
        co.set_argument("--disable-gpu")
        log.info("启动浏览器并初始化雪球会话 ...")
        self.page = ChromiumPage(co)
        self._init_session()

    def _init_session(self):
        self.page.get(XUEQIU_HOME)
        time.sleep(5)
        log.info(f"雪球首页已加载: {self.page.title}")

    def refresh_session(self, cooldown: float = 30):
        log.warning(f"WAF/限流 — 冷却 {cooldown}s 后重启浏览器 ...")
        time.sleep(cooldown)
        self._start_browser()
        time.sleep(3)
        log.info("会话已刷新")

    def _get_body_prefix(self) -> str:
        try:
            return self.page.run_js(
                "return document.body.innerText.substring(0, 200)"
            ) or ""
        except Exception:
            return ""

    def _is_waf(self, body: str = None) -> bool:
        if body is None:
            body = self._get_body_prefix()
        return any(kw in body for kw in self._WAF_KEYWORDS)

    def fetch_json(self, url: str, retries: int = 3) -> dict:
        last_err = None
        for attempt in range(retries + 1):
            try:
                self.page.get(url)
                time.sleep(2)
                body = self._get_body_prefix()
                if self._is_waf(body):
                    raise RuntimeError(f"WAF/限流: {body[:60]}")
                return self.page.run_js(
                    "return JSON.parse(document.body.innerText)"
                )
            except Exception as e:
                last_err = e
                if attempt < retries:
                    backoff = [5, 30, 60][min(attempt, 2)]
                    log.warning(
                        f"fetch_json 失败 (第{attempt+1}次, 等{backoff}s): {e}"
                    )
                    if attempt == 0:
                        time.sleep(backoff)
                    else:
                        self.refresh_session(cooldown=backoff)
        raise last_err

    def close(self):
        try:
            self.page.quit()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
#  爬取
# ═══════════════════════════════════════════════════════════

def fetch_today_discussions(
    xq: XueqiuSession,
    symbol: str,
    page_size: int = 100,
    max_pages: int = 20,
    delay: float = 2.0,
    target_date: date | None = None,
) -> list[dict]:
    """分页拉取个股指定日期的所有讨论 (默认今天)"""
    target = target_date or date.today()
    all_posts = []
    page = 1

    while page <= max_pages:
        url = (
            f"{XUEQIU_STOCK_COMMENT_URL}?"
            f"symbol={symbol}&count={page_size}&comment=0"
            f"&page={page}&sort=time"
        )
        try:
            data = xq.fetch_json(url)
        except Exception as e:
            log.warning(f"[{symbol}] 第 {page} 页请求失败: {e}")
            break

        statuses = data.get("list", [])
        if not statuses:
            break

        reached_older = False
        for p in statuses:
            post_date = _ts_to_date(p.get("created_at"))
            if post_date is None:
                continue
            if post_date < target:
                reached_older = True
                break
            if post_date == target:
                all_posts.append(p)

        if reached_older:
            break
        page += 1
        if page <= max_pages:
            time.sleep(delay)

    log.info(f"[{symbol}] {target} 共拉取 {len(all_posts)} 条讨论 (遍历 {page} 页)")
    return all_posts


def _parse_posts(posts: list[dict]) -> list[dict]:
    parsed = []
    for p in posts:
        text = _clean_html(p.get("text", "") or "")
        title = _clean_html(p.get("title", "") or "")
        if not text and not title:
            continue
        parsed.append({
            "id": p.get("id"),
            "user": (p.get("user") or {}).get("screen_name", ""),
            "followers_count": (p.get("user") or {}).get("followers_count", 0),
            "created_at": _ts_to_str(p.get("created_at")),
            "title": title,
            "text": text[:2000],
            "retweet_count": p.get("retweet_count", 0),
            "reply_count": p.get("reply_count", 0),
            "like_count": p.get("like_count", 0),
        })
    return parsed


# ═══════════════════════════════════════════════════════════
#  LLM 调用
# ═══════════════════════════════════════════════════════════

def _llm_call(
    base_url: str, model: str, api_key: str,
    system_prompt: str, user_msg: str,
    temperature: float = 0.1, timeout: int = 120,
    retries: int = 2,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=timeout
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = 5 * (attempt + 1)
                log.warning(f"LLM 调用失败 (第{attempt+1}次): {e}, {wait}s 后重试")
                time.sleep(wait)
    raise last_err


def _parse_json_from_llm(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else []
    except json.JSONDecodeError:
        return []


# ═══════════════════════════════════════════════════════════
#  第一步: LLM 过滤
# ═══════════════════════════════════════════════════════════

FILTER_SYSTEM_PROMPT = """\
你是一个上市公司基本面信息筛选助手。你的任务是从雪球社区帖子中筛选出
包含**企业经营实质信息**的帖子。

【务必保留的内容 — 重点关注】
1. **基本面变化**: 营收、利润、毛利率、净利率、ROE、财报数据、业绩预告/快报
2. **产品订单变化**: 合同签订、中标、客户采购、出货量、预收款、合同负债、订单排期
3. **产品涨价/降价**: 提价通知、终端零售价变动、出厂价调整、渠道价格、批价变化
4. **产能与产量**: 新建产能、投产进度、产能利用率、库存变化、基酒/原料储备
5. **行业与政策**: 行业周期、竞争格局、监管政策、消费趋势、上下游变化
6. **管理层动态**: 人事变动、股权激励、回购增持、战略调整、官方声明
7. **渠道变化**: 经销商动态、直营/电商、渠道库存、动销数据

【必须过滤掉】
- 纯买卖操作 (买入/卖出/加仓/减仓/止损/建仓 等个人交易行为)
- 纯技术分析 (K线/均线/MACD/布林带/支撑位/压力位 等)
- 纯情绪发泄 (骂人/喊口号/无实质内容的吐槽)
- 与公司经营无关的闲聊
- 纯转发无评论

给你一组帖子 (编号 1~N)，请返回你认为相关的帖子编号 JSON 数组。
只返回 JSON 数组，如 [1, 3, 7]。如果没有相关帖子，返回 []。
不要输出任何解释文字。"""


class LLMTooSlowError(Exception):
    """LLM 单批次超时, 触发自动降级到 crawl-only 模式"""
    pass


def llm_filter_posts(
    posts: list[dict],
    base_url: str, model: str, api_key: str,
    batch_size: int = 30,
    batch_timeout: float = 90.0,
) -> list[dict]:
    """LLM 过滤帖子。

    如果单个 batch 耗时超过 batch_timeout 秒, 抛出 LLMTooSlowError
    以便调用方降级为 crawl-only 模式。
    """
    if not posts:
        return []

    kept = []
    for start in range(0, len(posts), batch_size):
        batch = posts[start:start + batch_size]
        lines = []
        for i, p in enumerate(batch):
            idx = start + i + 1
            snippet = p["text"][:400].replace("\n", " ")
            title_part = f"标题: {p['title']} | " if p["title"] else ""
            lines.append(f"[{idx}] {title_part}{snippet}")

        user_msg = "\n\n".join(lines)
        log.info(f"  LLM 过滤: 第 {start+1}~{start+len(batch)} 条 ...")

        t0 = time.time()
        try:
            reply = _llm_call(
                base_url, model, api_key, FILTER_SYSTEM_PROMPT, user_msg
            )
            elapsed = time.time() - t0
            log.info(f"    batch 耗时 {elapsed:.1f}s")
            if elapsed > batch_timeout:
                raise LLMTooSlowError(
                    f"单批次耗时 {elapsed:.1f}s 超过阈值 {batch_timeout}s"
                )
            selected_ids = _parse_json_from_llm(reply)
            for sid in selected_ids:
                if isinstance(sid, int) and 1 <= sid <= len(posts):
                    kept.append(posts[sid - 1])
        except LLMTooSlowError:
            raise  # 向上传播, 不吞掉
        except Exception as e:
            log.warning(f"  LLM 过滤失败 (batch {start}): {e}, 跳过此批次")

    log.info(f"  LLM 过滤完成: {len(posts)} -> {len(kept)} 条")
    return kept


# ═══════════════════════════════════════════════════════════
#  第二步: LLM 生成 wiki 页面 (Karpathy 3-layer)
#
#  三层架构:
#    raw/   — 不可变的每日原始数据 (JSON, wiki 生成后删除)
#    wiki/  — LLM 维护的 markdown 页面 (7 维度 + index + log)
#    SCHEMA.md — 结构约定
# ═══════════════════════════════════════════════════════════

WIKI_PAGES = {
    "overview": {
        "title": "公司总览",
        "prompt": "生成/更新「公司总览」页面。包含: 公司基本定位、当前状态一段话概述、近期关注重点 (3-5条)、关键指标快照 (有数据就写, 没有标N/A)。",
    },
    "orders_channels": {
        "title": "订单与渠道",
        "prompt": "生成/更新「订单与渠道」页面。包含: 订单/合同信息 (预收款、合同负债、中标等)、渠道动态 (直营、经销商、电商)、产品动销 (各产品线出货/价格/库存)。",
    },
    "price_changes": {
        "title": "产品价格",
        "prompt": "生成/更新「产品价格」页面。包含: 出厂价/批价/零售价变动、提价/降价动态、渠道价差、价格趋势。",
    },
    "financials": {
        "title": "财务与业绩",
        "prompt": "生成/更新「财务与业绩」页面。包含: 业绩预期与实际 (营收/净利/增速)、估值指标 (PE/PB/股息率)、分红与现金流、财报关键日期。量化数据保留原始数字。",
    },
    "capacity": {
        "title": "产能与产量",
        "prompt": "生成/更新「产能与产量」页面。包含: 当前产能与产量、产能规划与释放节奏、库存变化、产能壁垒。",
    },
    "industry_policy": {
        "title": "行业与政策",
        "prompt": "生成/更新「行业与政策」页面。包含: 行业周期定位、竞争格局 (同业对比)、政策法规、消费趋势、上下游变化。",
    },
    "management": {
        "title": "管理层与治理",
        "prompt": "生成/更新「管理层与治理」页面。包含: 管理层人事变动、回购/增持/资本配置、战略方向调整、官方声明。",
    },
}

WIKI_PAGE_SYSTEM = """\
你是一个上市公司知识库维护员。你的任务是根据当天的雪球社区帖子数据，
生成或更新一个结构化的 wiki 知识页面。

要求：
1. 提取帖子中的**事实性信息**，忽略纯情绪
2. 用**简洁的要点**组织信息，每条标注来源: > 📌 来源: @用户名
3. 如果是首次创建页面，建立完整框架
4. 如果是更新已有页面，融合新信息，新增内容用 🆕 标记
5. 输出纯 markdown，以 YAML frontmatter 开头:
   ---
   title: "页面标题"
   updated: "YYYY-MM-DD"
   sources_count: N
   ---
6. 保持客观中立
7. 某个方面无信息就写"暂无数据"
8. 新信息与已有内容矛盾时，保留两者并标注 ⚠️ 矛盾
9. 直接输出 markdown，不要包裹在代码块中"""

SCHEMA_TEMPLATE = """\
# {symbol} Wiki Schema

本文件定义了 {symbol} wiki 的结构约定 (Karpathy LLM-Wiki 模式)。

## 三层架构

1. **raw/** — 每日原始数据 (JSON)。LLM 只读，处理后删除。
2. **wiki/** — LLM 维护的 markdown 页面。每次 ingest 新数据时更新。
3. **SCHEMA.md** (本文件) — 约定结构、页面定义和工作流。

## 页面定义

| 页面 | 文件 | 内容 |
|------|------|------|
| 公司总览 | `wiki/overview.md` | 基本信息、当前状态、关注重点 |
| 订单与渠道 | `wiki/orders_channels.md` | 订单/合同/渠道动销/经销商 |
| 产品价格 | `wiki/price_changes.md` | 出厂价/批价/零售价变动 |
| 财务与业绩 | `wiki/financials.md` | 营收/利润/估值/分红 |
| 产能与产量 | `wiki/capacity.md` | 产能规划/产量/库存 |
| 行业与政策 | `wiki/industry_policy.md` | 行业周期/竞争/政策 |
| 管理层与治理 | `wiki/management.md` | 人事变动/回购/战略 |

## 特殊页面

- **wiki/index.md** — 所有页面的目录 (链接 + 摘要 + 更新日期)
- **wiki/log.md** — 按时间倒序的操作日志

## 约定

- YAML frontmatter: `title`, `updated`, `sources_count`
- 信息来源: `> 📌 来源: @用户名 (YYYY-MM-DD)`
- 矛盾标注: `⚠️ 矛盾`
- 新增内容: `🆕`
- 过时信息: ~~删除线~~
- 页面间引用: [[page_name]]
"""


def _prepare_posts_text(posts: list, max_posts: int = 50) -> str:
    lines = []
    for i, p in enumerate(posts[:max_posts]):
        text = p.get("text", "")[:400].replace("\n", " ")
        user = p.get("user", "匿名")
        followers = p.get("followers_count", 0)
        user_info = f"@{user}"
        if followers > 1000:
            user_info += f" (粉丝 {followers})"
        title = p.get("title", "")
        title_part = f"【{title}】" if title else ""
        lines.append(f"[{i+1}] {user_info} | {p.get('created_at','')}\n{title_part}{text}")
    return "\n\n".join(lines)


def _generate_wiki_page(
    page_name: str,
    page_def: dict,
    symbol: str,
    data_date: str,
    posts_text: str,
    existing_content: str | None,
    base_url: str, model: str, api_key: str,
) -> str:
    if existing_content:
        user_msg = (
            f"## 任务\n更新 {symbol} 的「{page_def['title']}」wiki 页面。\n\n"
            f"## 已有页面内容\n```markdown\n{existing_content}\n```\n\n"
            f"## 新数据 ({data_date})\n{posts_text}\n\n"
            f"## 页面要求\n{page_def['prompt']}\n\n"
            "基于已有内容融合新数据，新增信息用 🆕 标记。更新 frontmatter 中的 updated 和 sources_count。"
        )
    else:
        user_msg = (
            f"## 任务\n为 {symbol} 创建「{page_def['title']}」wiki 页面。\n\n"
            f"## 数据来源 ({data_date})\n{posts_text}\n\n"
            f"## 页面要求\n{page_def['prompt']}\n\n"
            "输出完整 markdown 页面，以 YAML frontmatter 开头。"
        )

    result = _llm_call(base_url, model, api_key, WIKI_PAGE_SYSTEM, user_msg,
                       temperature=0.3, timeout=180)
    result = result.strip()
    if result.startswith("```"):
        result = re.sub(r"^```\w*\n?", "", result)
        result = re.sub(r"\n?```$", "", result)
    return result


def _generate_index(symbol: str, wiki_dir: Path, data_date: str) -> str:
    lines = [
        "---",
        f'title: "{symbol} Wiki Index"',
        f'updated: "{data_date}"',
        "---", "",
        f"# {symbol} Wiki Index", "",
        "| 页面 | 文件 | 最后更新 |",
        "|------|------|----------|",
    ]
    for page_name, page_def in WIKI_PAGES.items():
        fp = wiki_dir / f"{page_name}.md"
        updated = data_date
        if fp.exists():
            content = fp.read_text(encoding="utf-8")
            m = re.search(r'updated:\s*"?(\d{4}-\d{2}-\d{2})"?', content)
            if m:
                updated = m.group(1)
        lines.append(
            f"| {page_def['title']} | [{page_name}.md]({page_name}.md) | {updated} |"
        )
    lines.extend(["", f"*自动生成于 {data_date}*"])
    return "\n".join(lines)


def _append_log(log_path: Path, symbol: str, data_date: str,
                total_posts: int, filtered_posts: int, pages_updated: list[str]):
    entry = (
        f"\n## [{data_date}] ingest | 雪球社区\n\n"
        f"- 原始帖子: {total_posts}\n"
        f"- 筛选保留: {filtered_posts}\n"
        f"- 更新页面: {', '.join(pages_updated)}\n"
    )
    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8")
        if "---" in existing:
            parts = existing.split("---", 2)
            if len(parts) >= 3:
                new_content = f"---{parts[1]}---{entry}{parts[2]}"
                log_path.write_text(new_content, encoding="utf-8")
                return
        log_path.write_text(existing + entry, encoding="utf-8")
    else:
        header = (
            "---\n"
            f'title: "{symbol} Wiki Log"\n'
            f'updated: "{data_date}"\n'
            "---\n\n"
            f"# {symbol} Wiki Log\n\n"
            "按时间倒序记录每次 ingest 操作。\n"
        )
        log_path.write_text(header + entry, encoding="utf-8")


# ═══════════════════════════════════════════════════════════
#  主流程: 爬取 → 过滤 → Karpathy Wiki → 删 JSON
# ═══════════════════════════════════════════════════════════

def monitor_stock(
    code: str,
    xq: XueqiuSession,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_api_key: str = "",
    output_dir: str = "results/xueqiu_orders",
    page_delay: float = 2.0,
    crawl_only: bool = False,
    target_date: date | None = None,
    llm_timeout: float = 90.0,
) -> dict:
    """单只股票管线。

    crawl_only=True:  爬取 → 保存全量 JSON 到 raw/ (供 Copilot agent 后续处理)
    crawl_only=False: 爬取 → LLM 过滤 → summary MD → wiki(7页) → 删 JSON
    target_date:      指定抓取日期 (默认今天)
    llm_timeout:      LLM 单批次超时秒数, 超过则自动降级为 crawl-only
    """
    symbol = _to_xueqiu_symbol(code)
    log.info(f"=== 处理 {symbol} ===")

    target = target_date or date.today()
    today_str = target.strftime("%Y%m%d")
    data_date = str(target)
    stock_dir = Path(output_dir) / symbol
    raw_dir = stock_dir / "raw"
    wiki_dir = stock_dir / "wiki"
    schema_path = stock_dir / "SCHEMA.md"

    # ① 爬取指定日期所有讨论
    raw_posts = fetch_today_discussions(xq, symbol, delay=page_delay, target_date=target)
    if not raw_posts:
        log.info(f"[{symbol}] 今日无讨论数据")
        return {"symbol": symbol, "status": "no_data"}

    posts = _parse_posts(raw_posts)
    log.info(f"[{symbol}] 解析后 {len(posts)} 条有效帖子")

    # 保存全量帖子 JSON 到 raw/
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / f"{today_str}.json"
    result_data = {
        "symbol": symbol,
        "date": data_date,
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_posts_today": len(posts),
        "filtered_count": 0,
        "filtered_posts": posts,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    log.info(f"  saved raw/{today_str}.json ({len(posts)} 条)")

    # crawl-only 模式: 到此为止
    if crawl_only:
        log.info(f"[{symbol}] crawl-only 完成")
        return {"symbol": symbol, "status": "crawled", "total": len(posts)}

    # ── 以下为 LLM 全自动模式 ──

    # ② LLM 过滤: 只保留基本面/订单/价格相关
    try:
        filtered = llm_filter_posts(
            posts, llm_base_url, llm_model, llm_api_key,
            batch_timeout=llm_timeout,
        )
    except LLMTooSlowError as e:
        log.warning(
            f"[{symbol}] ⚡ LLM 太慢 ({e}), 自动降级为 crawl-only 模式。"
            f" JSON 已保存, 请用 Copilot agent 处理。"
        )
        return {"symbol": symbol, "status": "crawled", "total": len(posts)}
    if not filtered:
        log.info(f"[{symbol}] 过滤后无基本面相关讨论")
        # JSON 保留, 以便后续 Copilot 处理
        return {"symbol": symbol, "status": "no_relevant"}

    # ③ 更新 JSON 为筛选后版本
    result_data["filtered_count"] = len(filtered)
    result_data["filtered_posts"] = filtered
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    # ④ 生成 SCHEMA.md (仅首次)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    if not schema_path.exists():
        schema_path.write_text(
            SCHEMA_TEMPLATE.format(symbol=symbol), encoding="utf-8"
        )
        log.info("  created SCHEMA.md")

    # ⑤ LLM 生成/更新每个 wiki 页面 (Karpathy ingest)
    posts_text = _prepare_posts_text(filtered)
    pages_updated = []

    for page_name, page_def in WIKI_PAGES.items():
        page_path = wiki_dir / f"{page_name}.md"
        existing = None
        if page_path.exists():
            existing = page_path.read_text(encoding="utf-8")

        log.info(f"  LLM wiki/{page_name}.md ...")
        try:
            content = _generate_wiki_page(
                page_name, page_def, symbol, data_date,
                posts_text, existing,
                llm_base_url, llm_model, llm_api_key,
            )
            page_path.write_text(content, encoding="utf-8")
            pages_updated.append(page_name)
            log.info(f"    ✓ {page_name}.md ({len(content)} chars)")
        except Exception as e:
            log.error(f"    ✗ {page_name}.md 失败: {e}")

    # ⑥ 更新 index.md
    index_content = _generate_index(symbol, wiki_dir, data_date)
    (wiki_dir / "index.md").write_text(index_content, encoding="utf-8")
    log.info("  ✓ index.md")

    # ⑦ 追加 log.md
    _append_log(
        wiki_dir / "log.md", symbol, data_date,
        len(posts), len(filtered), pages_updated,
    )
    log.info("  ✓ log.md")

    # ⑧ 生成 summary MD (持久保存在 raw/) 然后删除 JSON
    if pages_updated:
        _save_summary_md(raw_dir, today_str, symbol, data_date, posts, filtered)
        if json_path.exists():
            json_path.unlink()
            log.info(f"  🗑 raw/{today_str}.json 已删除")

    log.info(
        f"[{symbol}] 完成: {len(posts)} 总讨论 → "
        f"{len(filtered)} 条基本面 → "
        f"{len(pages_updated)}/{len(WIKI_PAGES)} wiki 页面已更新"
    )
    return {
        "symbol": symbol,
        "status": "ok",
        "total": len(posts),
        "filtered": len(filtered),
        "pages_updated": len(pages_updated),
    }


def _save_summary_md(raw_dir: Path, today_str: str, symbol: str,
                     data_date: str, all_posts: list, filtered: list):
    """生成 raw/{date}_summary.md — 筛选后帖子的持久存档 (不删除)"""
    lines = [
        f"# {symbol} 基本面信息摘要 ({data_date})",
        "",
        f"> 数据来源: 雪球社区 | 今日总讨论 {len(all_posts)} 条 | 筛选后 {len(filtered)} 条",
        "",
    ]
    for i, p in enumerate(filtered, 1):
        user = p.get('user', '匿名')
        time_str = p.get('created_at', '')
        title = p.get('title', '')
        text = p.get('text', '')[:800]
        title_line = f"**{title}**\n\n" if title else ""
        lines.append(f"### [{i}] @{user} ({time_str})")
        lines.append("")
        lines.append(f"{title_line}{text}")
        lines.append("")

    summary_path = raw_dir / f"{today_str}_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  ✓ raw/{today_str}_summary.md ({len(filtered)} 条)")


# ═══════════════════════════════════════════════════════════
#  批量入口
# ═══════════════════════════════════════════════════════════

def _load_codes_from_csv(csv_path: str) -> list[str]:
    codes = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("symbol") or "").strip()
            if code:
                codes.append(code)
    return codes


def main():
    parser = argparse.ArgumentParser(
        description="雪球基本面采集: 爬取 -> LLM过滤 -> wiki生成"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--codes", help="股票代码, 逗号分隔")
    group.add_argument("--stock-csv", help="股票列表 CSV (含 symbol 列)")
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument(
        "--output-dir",
        default="/nvme5/xtang/gp-workspace/gp-data/xueqiu_orders",
    )
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="股票间延迟秒数 (防限流, >= 3)",
    )
    parser.add_argument("--page-delay", type=float, default=2.0)
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="跳过今天已有 wiki 或 JSON 的股票",
    )
    parser.add_argument(
        "--crawl-only", action="store_true", default=False,
        help="纯爬取模式: 只保存全量 JSON, 不调用 LLM (供 Copilot agent 后续处理)",
    )
    parser.add_argument(
        "--date", default=None,
        help="指定抓取日期 YYYYMMDD (默认今天), 例: --date 20260413",
    )
    parser.add_argument(
        "--llm-timeout", type=float, default=90.0,
        help="LLM 单批次超时秒数 (默认90), 超过自动降级为 crawl-only",
    )
    args = parser.parse_args()

    api_key = (
        args.llm_api_key
        or os.environ.get("DASHSCOPE_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key and not args.crawl_only:
        log.error(
            "需要 LLM API Key: --llm-api-key 或设置 DASHSCOPE_API_KEY (或使用 --crawl-only)"
        )
        sys.exit(1)

    if args.stock_csv:
        codes = _load_codes_from_csv(args.stock_csv)
        log.info(f"从 CSV 加载 {len(codes)} 只股票")
    else:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]

    if not codes:
        log.error("无股票代码")
        sys.exit(1)

    # 解析目标日期
    if args.date:
        target_date = datetime.strptime(args.date, "%Y%m%d").date()
    else:
        target_date = date.today()
    log.info(f"目标日期: {target_date}")

    xq = XueqiuSession()

    today_str = target_date.strftime("%Y%m%d")
    stats = {"ok": 0, "no_data": 0, "no_relevant": 0, "fail": 0, "resumed": 0, "crawled": 0}
    consecutive_empty = 0
    WAF_THRESHOLD = 5
    waf_cooldown = 120

    try:
        for i, code in enumerate(codes):
            if i > 0:
                time.sleep(args.delay)

            # 断点续传
            if args.resume:
                sym = _to_xueqiu_symbol(code)
                stock_path = Path(args.output_dir) / sym
                # crawl-only: 跳过已有 JSON 的
                if args.crawl_only:
                    raw_json = stock_path / "raw" / f"{today_str}.json"
                    if raw_json.exists():
                        stats["resumed"] += 1
                        if stats["resumed"] % 100 == 0:
                            log.info(f"已跳过 {stats['resumed']} 只 (已爬取)")
                        continue
                else:
                    # 全流程: 跳过已有 wiki log 或 summary MD 的
                    log_md = stock_path / "wiki" / "log.md"
                    summary_md = stock_path / "raw" / f"{today_str}_summary.md"
                    if summary_md.exists():
                        stats["resumed"] += 1
                        if stats["resumed"] % 100 == 0:
                            log.info(f"已跳过 {stats['resumed']} 只 (今天已处理)")
                        continue
                    if log_md.exists():
                        log_content = log_md.read_text(encoding="utf-8")
                        if f"[{date.today()}] ingest" in log_content:
                            stats["resumed"] += 1
                            if stats["resumed"] % 100 == 0:
                                log.info(f"已跳过 {stats['resumed']} 只 (今天已处理)")
                            continue

            try:
                result = monitor_stock(
                    code, xq,
                    llm_base_url=args.llm_base_url,
                    llm_model=args.llm_model,
                    llm_api_key=api_key,
                    output_dir=args.output_dir,
                    page_delay=args.page_delay,
                    crawl_only=args.crawl_only,
                    target_date=target_date,
                    llm_timeout=args.llm_timeout,
                )
                status = result.get("status", "fail")
                stats[status] = stats.get(status, 0) + 1

                if status == "ok":
                    consecutive_empty = 0
                    waf_cooldown = 120
                else:
                    consecutive_empty += 1
            except Exception as e:
                log.error(f"[{code}] 异常: {e}")
                stats["fail"] += 1
                consecutive_empty += 1

            # 连续无数据 -> 可能被限流
            if consecutive_empty >= WAF_THRESHOLD:
                log.warning(
                    f"连续 {consecutive_empty} 只无数据, "
                    f"冷却 {waf_cooldown}s ..."
                )
                xq.refresh_session(cooldown=waf_cooldown)
                waf_cooldown = min(waf_cooldown * 2, 600)
                consecutive_empty = 0

            # 进度
            done = i + 1
            if done % 50 == 0 or done == len(codes):
                log.info(
                    f"进度: {done}/{len(codes)} | "
                    f"有信息: {stats['ok']} | "
                    f"无数据: {stats['no_data']} | "
                    f"无相关: {stats['no_relevant']} | "
                    f"失败: {stats['fail']} | "
                    f"跳过: {stats['resumed']}"
                )
    finally:
        xq.close()

    log.info(
        f"全部完成: {len(codes)} 只 | "
        f"有信息: {stats['ok']} | 无数据: {stats['no_data']} | "
        f"无相关: {stats['no_relevant']} | 失败: {stats['fail']} | "
        f"跳过: {stats['resumed']}"
    )


if __name__ == "__main__":
    main()
