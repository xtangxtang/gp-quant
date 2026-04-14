"""Stock Wiki Ingest — Karpathy LLM-Wiki pattern for per-stock knowledge bases

每个股票拥有自己的 wiki 目录结构：
    {data_dir}/{symbol}/
        raw/              ← 不可变的每日 JSON
        wiki/             ← LLM 维护的 wiki 页面
            index.md
            log.md
            overview.md
            orders_channels.md
            financials.md
            capacity.md
            industry_policy.md
            management.md
            market_sentiment.md
        SCHEMA.md

Operations:
    ingest  — 处理新的每日 JSON，更新所有 wiki 页面
    lint    — 检查 wiki 健康状况

用法:
    python -m src.wiki.stock_wiki_ingest --symbol SH600519 \\
        --data-dir /nvme5/xtang/gp-workspace/gp-data/xueqiu_orders
    python -m src.wiki.stock_wiki_ingest --symbol SH600519 --date 20260413
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, date
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── env loading ────────────────────────────────────────────

def _load_env_file():
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

# ── LLM ────────────────────────────────────────────────────

DEFAULT_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
DEFAULT_MODEL = "qwen3.6-plus"


def _llm_call(
    base_url: str, model: str, api_key: str,
    system_prompt: str, user_msg: str,
    temperature: float = 0.3, timeout: int = 180,
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
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = 5 * (attempt + 1)
                log.warning(f"LLM call failed (attempt {attempt+1}): {e}, retry in {wait}s")
                time.sleep(wait)
    raise last_err


# ── Wiki page definitions ──────────────────────────────────

WIKI_PAGES = [
    "overview",
    "orders_channels",
    "financials",
    "capacity",
    "industry_policy",
    "management",
    "market_sentiment",
]

SCHEMA_TEMPLATE = """\
# {symbol} Wiki Schema

本文件定义了 {symbol} wiki 的结构约定，指导 LLM 如何维护这套知识库。

## 三层架构

1. **raw/** — 不可变的每日原始数据 (JSON)。LLM 只读，永不修改。
2. **wiki/** — LLM 维护的 markdown 页面。每次 ingest 新的 raw source 时，LLM 读取
   原始数据和已有 wiki 页面，然后更新所有相关页面。
3. **SCHEMA.md** (本文件) — 约定 wiki 的结构、页面定义和工作流。

## 页面定义

| 页面 | 文件 | 内容 |
|------|------|------|
| 总览 | `wiki/overview.md` | 公司基本信息、当前状态的一段话总结、关键指标快照 |
| 订单/渠道 | `wiki/orders_channels.md` | 订单、合同、渠道动销、经销商、预收款等 |
| 财务/业绩 | `wiki/financials.md` | 营收、利润、估值、分红、现金流等 |
| 产能/产量 | `wiki/capacity.md` | 产能规划、基酒产量、产能释放节奏等 |
| 行业/政策 | `wiki/industry_policy.md` | 行业周期、竞争格局、监管政策、消费趋势等 |
| 管理层 | `wiki/management.md` | 管理层变动、公司治理、资本配置决策等 |
| 市场情绪 | `wiki/market_sentiment.md` | 多空分歧、机构观点、市场博弈等 |

## 特殊页面

- **wiki/index.md** — 所有页面的目录，每页一行链接 + 摘要 + 最后更新日期
- **wiki/log.md** — 按时间倒序的操作日志，记录每次 ingest 的日期、来源数量、更新了哪些页面

## 约定

- 每个页面顶部有 YAML frontmatter: `title`, `updated`, `sources_count`
- 页面内用 `> 📌 来源: xxx (YYYY-MM-DD)` 标注信息来源
- 新信息与已有内容矛盾时，保留两者并标注 `⚠️ 矛盾`
- 过时的信息用 ~~删除线~~ 标记但不删除
- 页面间用 `[[page_name]]` 风格互相引用
"""

# ── Ingest prompts ─────────────────────────────────────────

INGEST_SYSTEM_PROMPT = """\
你是一个上市公司知识库维护员。你的任务是根据当天的雪球社区帖子数据，生成或更新
一个结构化的 wiki 知识页面。

要求：
1. 提取帖子中的**事实性信息**，忽略纯情绪表达
2. 用**简洁的要点**组织信息，每条注明来源 (发帖人)
3. 如果是首次创建页面，建立完整的框架
4. 如果是更新已有页面，融合新信息，标注新增内容的日期
5. 输出纯 markdown，以 YAML frontmatter 开头
6. 保持客观中立的语气
"""

PAGE_PROMPTS = {
    "overview": """\
生成/更新「公司总览」页面。包含：
- 公司全称、代码、行业
- 当前状态的一段话概述（基于最新信息）
- 关键指标快照表格（股价区间、PE、股息率、市值等，有数据就写，没有标N/A）
- 近期关注重点（3-5条）

YAML frontmatter 格式：
```yaml
---
title: "公司总览"
updated: "YYYY-MM-DD"
sources_count: N
---
```""",

    "orders_channels": """\
生成/更新「订单与渠道」页面。包含：
- 订单/合同信息（预收款、合同负债、中标等）
- 渠道动态（直营、经销商、i茅台/电商平台等）
- 产品动销（各产品线出货、价格、库存等）
- 渠道变革趋势

每条信息标注来源。如果帖子中没有相关信息，该分类下写"暂无数据"。""",

    "financials": """\
生成/更新「财务与业绩」页面。包含：
- 业绩预期与实际（营收、净利润、增速）
- 估值指标（PE、PB、股息率）
- 分红与现金流
- 财报关键日期

每条信息标注来源。量化数据尽量保留原始数字。""",

    "capacity": """\
生成/更新「产能与产量」页面。包含：
- 当前产能与产量数据
- 产能规划与释放节奏
- 基酒存量与可售量对应关系
- 产能壁垒与护城河

每条信息标注来源。""",

    "industry_policy": """\
生成/更新「行业与政策」页面。包含：
- 行业周期定位（当前处于什么阶段）
- 竞争格局（同业对比）
- 政策法规（监管、行业政策）
- 消费趋势

每条信息标注来源。""",

    "management": """\
生成/更新「管理层动态」页面。包含：
- 管理层人事变动
- 公司治理（回购、增持、资本配置）
- 战略方向调整
- 官方澄清/声明

每条信息标注来源。""",

    "market_sentiment": """\
生成/更新「市场情绪」页面。包含：
- 多方主要论点（看涨理由）
- 空方主要论点（看跌理由）
- 机构/大V观点（有影响力的发言人）
- 情绪指标（互动量分布、关注度等）

每条信息标注来源。""",
}


def _prepare_posts_text(posts: list, max_posts: int = 50, max_chars: int = 400) -> str:
    """将帖子列表压缩为 LLM 可消费的文本"""
    lines = []
    for i, p in enumerate(posts[:max_posts]):
        text = p.get("text", "")[:max_chars]
        user = p.get("user", "匿名")
        likes = p.get("like_count", 0)
        replies = p.get("reply_count", 0)
        lines.append(f"[{i+1}] @{user} (👍{likes} 💬{replies}): {text}")
    return "\n\n".join(lines)


def _generate_page(
    page_name: str,
    symbol: str,
    data_date: str,
    posts_text: str,
    existing_content: str | None,
    base_url: str,
    model: str,
    api_key: str,
) -> str:
    """调用 LLM 生成或更新一个 wiki 页面"""
    page_instruction = PAGE_PROMPTS[page_name]

    if existing_content:
        user_msg = f"""## 任务
更新 {symbol} 的「{page_name}」wiki 页面。

## 已有页面内容
```markdown
{existing_content}
```

## 新数据 ({data_date})
以下是今天从雪球社区筛选的 {symbol} 相关帖子：

{posts_text}

## 页面要求
{page_instruction}

请基于已有内容，融合新数据后输出完整的更新版页面。新增信息用 `🆕` 标记。
更新 frontmatter 中的 `updated` 日期和 `sources_count`。"""
    else:
        user_msg = f"""## 任务
为 {symbol} 创建「{page_name}」wiki 页面。

## 数据来源 ({data_date})
以下是从雪球社区筛选的 {symbol} 相关帖子（已过滤掉纯情绪/技术面内容）：

{posts_text}

## 页面要求
{page_instruction}

请输出完整的 markdown 页面，以 YAML frontmatter 开头。"""

    result = _llm_call(base_url, model, api_key, INGEST_SYSTEM_PROMPT, user_msg)
    # Strip markdown code fence if LLM wraps output
    result = result.strip()
    if result.startswith("```"):
        result = re.sub(r"^```\w*\n?", "", result)
        result = re.sub(r"\n?```$", "", result)
    return result


def _generate_index(symbol: str, wiki_dir: Path, data_date: str) -> str:
    """生成 index.md — 所有页面的目录"""
    lines = [
        "---",
        f'title: "{symbol} Wiki Index"',
        f'updated: "{data_date}"',
        "---",
        "",
        f"# {symbol} Wiki Index",
        "",
        "| 页面 | 文件 | 最后更新 |",
        "|------|------|----------|",
    ]
    for page in WIKI_PAGES:
        fp = wiki_dir / f"{page}.md"
        updated = data_date
        if fp.exists():
            # Try to extract updated from frontmatter
            content = fp.read_text(encoding="utf-8")
            m = re.search(r'updated:\s*"?(\d{4}-\d{2}-\d{2})"?', content)
            if m:
                updated = m.group(1)
        title_map = {
            "overview": "公司总览",
            "orders_channels": "订单与渠道",
            "financials": "财务与业绩",
            "capacity": "产能与产量",
            "industry_policy": "行业与政策",
            "management": "管理层动态",
            "market_sentiment": "市场情绪",
        }
        lines.append(f"| {title_map.get(page, page)} | [{page}.md]({page}.md) | {updated} |")

    lines.extend([
        "",
        f"*自动生成于 {data_date}*",
    ])
    return "\n".join(lines)


def _append_log(log_path: Path, symbol: str, data_date: str,
                total_posts: int, filtered_posts: int, pages_updated: list[str]):
    """追加 ingest 记录到 log.md"""
    entry = (
        f"\n## [{data_date}] ingest | 雪球社区\n\n"
        f"- 原始帖子: {total_posts}\n"
        f"- 筛选保留: {filtered_posts}\n"
        f"- 更新页面: {', '.join(pages_updated)}\n"
    )
    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8")
        # Insert after frontmatter
        if "---" in existing:
            parts = existing.split("---", 2)
            if len(parts) >= 3:
                # frontmatter is parts[1], content is parts[2]
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


# ── Main ingest flow ───────────────────────────────────────

def ingest(
    symbol: str,
    data_dir: str,
    target_date: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
):
    """Ingest a daily JSON into the stock's wiki"""
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise ValueError("Missing DASHSCOPE_API_KEY")

    stock_dir = Path(data_dir) / symbol
    raw_dir = stock_dir / "raw"
    wiki_dir = stock_dir / "wiki"
    schema_path = stock_dir / "SCHEMA.md"

    # Find JSON to ingest
    if target_date:
        json_candidates = [stock_dir / f"{target_date}.json", raw_dir / f"{target_date}.json"]
    else:
        # Find latest JSON
        json_candidates = sorted(stock_dir.glob("*.json"), reverse=True) + \
                         sorted(raw_dir.glob("*.json"), reverse=True)

    json_path = None
    for c in json_candidates:
        if c.is_file():
            json_path = c
            break

    if not json_path:
        log.error(f"No JSON found for {symbol} in {stock_dir}")
        return

    log.info(f"Ingesting {json_path.name} for {symbol}")

    # Load data
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    filtered_posts = data.get("filtered_posts", [])
    total_posts = data.get("total_posts_today", len(filtered_posts))
    data_date = data.get("date", json_path.stem)
    # Normalize date format
    if len(data_date) == 8:
        data_date = f"{data_date[:4]}-{data_date[4:6]}-{data_date[6:]}"

    if not filtered_posts:
        log.warning(f"No filtered posts in {json_path.name}")
        return

    log.info(f"  {len(filtered_posts)} filtered posts, {total_posts} total")

    # Create directory structure
    raw_dir.mkdir(parents=True, exist_ok=True)
    wiki_dir.mkdir(parents=True, exist_ok=True)

    # Move JSON to raw/ if not already there
    if json_path.parent != raw_dir:
        dest = raw_dir / json_path.name
        if not dest.exists():
            import shutil
            shutil.copy2(json_path, dest)
            log.info(f"  Copied {json_path.name} → raw/")

    # Write SCHEMA.md if not exists
    if not schema_path.exists():
        schema_path.write_text(
            SCHEMA_TEMPLATE.format(symbol=symbol),
            encoding="utf-8",
        )
        log.info("  Created SCHEMA.md")

    # Prepare posts text
    posts_text = _prepare_posts_text(filtered_posts)

    # Generate/update each wiki page
    pages_updated = []
    for page_name in WIKI_PAGES:
        page_path = wiki_dir / f"{page_name}.md"
        existing = None
        if page_path.exists():
            existing = page_path.read_text(encoding="utf-8")

        log.info(f"  Generating wiki/{page_name}.md ...")
        try:
            content = _generate_page(
                page_name=page_name,
                symbol=symbol,
                data_date=data_date,
                posts_text=posts_text,
                existing_content=existing,
                base_url=base_url,
                model=model,
                api_key=api_key,
            )
            page_path.write_text(content, encoding="utf-8")
            pages_updated.append(page_name)
            log.info(f"    ✓ {page_name}.md ({len(content)} chars)")
        except Exception as e:
            log.error(f"    ✗ {page_name}.md failed: {e}")

    # Generate index
    index_content = _generate_index(symbol, wiki_dir, data_date)
    (wiki_dir / "index.md").write_text(index_content, encoding="utf-8")
    log.info("  ✓ index.md")

    # Append log
    _append_log(
        wiki_dir / "log.md", symbol, data_date,
        total_posts, len(filtered_posts), pages_updated,
    )
    log.info("  ✓ log.md")

    # Extract summary to raw/{date}_summary.md
    summary_text = data.get("summary", "")
    if summary_text:
        date_str = data_date.replace("-", "")
        summary_path = raw_dir / f"{date_str}_summary.md"
        summary_path.write_text(summary_text, encoding="utf-8")
        log.info(f"  ✓ raw/{date_str}_summary.md ({len(summary_text)} chars)")

    log.info(f"Ingest complete: {len(pages_updated)}/{len(WIKI_PAGES)} pages updated")


# ── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stock Wiki Ingest")
    parser.add_argument("--symbol", required=True, help="股票代码 (e.g. SH600519)")
    parser.add_argument(
        "--data-dir",
        default="/nvme5/xtang/gp-workspace/gp-data/xueqiu_orders",
        help="数据根目录",
    )
    parser.add_argument("--date", default=None, help="指定日期 (e.g. 20260413)")
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-api-key", default=None)
    args = parser.parse_args()

    ingest(
        symbol=args.symbol,
        data_dir=args.data_dir,
        target_date=args.date,
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key=args.llm_api_key,
    )


if __name__ == "__main__":
    main()
