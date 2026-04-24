"""雪球行业分析深度文章采集 — 纯 requests 模式（无需浏览器）

特点：
- 只需要雪球 cookie（XQ_A_TOKEN），不需要 DrissionPage/浏览器
- 星计划创作者分流：星计划 → 直接深度分析；普通用户 → 规则粗筛 + LLM 判断
- 结构化输出到个股目录

用法:
    # 单股测试
    python scripts/xueqiu_deep_analysis.py --codes 600519 --cookie "xq_a_token=xxx"

    # 批量 + LLM 分析
    python scripts/xueqiu_deep_analysis.py \
        --codes 600519,000858,600036 \
        --cookie "xq_a_token=xxx" \
        --days 3

    # 仅采集不分析（crawl-only）
    python scripts/xueqiu_deep_analysis.py \
        --codes 600519 --cookie "xq_a_token=xxx" --crawl-only
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── 配置 ───────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = "/home/xtang/gp-workspace/gp-data/xueqiu_analysis"
DEFAULT_LLM_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
DEFAULT_LLM_MODEL = "qwen3.6-plus"

XUEQIU_API = "https://xueqiu.com/query/v1/symbol/search/status.json"

# 星计划关键词（用户简介/认证信息中检测）
STAR_CREATOR_KEYWORDS = [
    "星计划", "雪球星计划", "专栏作家", "签约作者",
    "创作者", "V+", "金V", "认证用户",
]

# 基本面/行业关键词（规则粗筛）
FUNDAMENTAL_KEYWORDS = [
    "研报", "财报", "行业", "基本面", "估值", "深度", "分析", "逻辑",
    "拆解", "赛道", "护城河", "ROE", "PE", "PB", "DCF",
    "营收", "利润", "毛利率", "净利率",
    "订单", "合同", "产能", "产量", "库存",
    "涨价", "降价", "提价", "批价",
    "政策", "监管", "竞争", "格局", "壁垒",
    "分红", "回购", "增持", "减持",
    "业绩", "预告", "快报", "季报", "年报", "中报",
    "财务", "现金流", "负债", "资产",
    "机构", "基金", "券商", "投行",
    "研报", "评级", "目标价", "买入", "卖出",
]


# ═══════════════════════════════════════════════════════════
#  代码转换
# ═══════════════════════════════════════════════════════════

def to_xueqiu_symbol(code: str) -> str:
    """将股票代码转为雪球格式"""
    code = code.strip().upper()
    if code.startswith(("SH", "SZ")):
        return code
    if code.startswith(("6", "9")):
        return f"SH{code}"
    if code.startswith(("0", "3", "2")):
        return f"SZ{code}"
    # 美股/港股保持原样
    return code


# ═══════════════════════════════════════════════════════════
#  雪球采集
# ═══════════════════════════════════════════════════════════

class XueqiuFetcher:
    """纯 requests 雪球采集器"""

    def __init__(self, cookie_str: str):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://xueqiu.com/",
            "Origin": "https://xueqiu.com",
        })
        # 设置 cookie
        if "xq_a_token" in cookie_str:
            token = cookie_str.split("xq_a_token=")[1].split(";")[0]
            self.session.cookies.set("xq_a_token", token, domain=".xueqiu.com")
        else:
            self.session.cookies.set("xq_a_token", cookie_str, domain=".xueqiu.com")

    def fetch_posts(
        self, symbol: str,
        max_pages: int = 10,
        page_size: int = 20,
        delay: float = 2.0,
    ) -> list[dict]:
        """分页拉取帖子"""
        all_posts = []
        for page in range(1, max_pages + 1):
            url = (
                f"{XUEQIU_API}?"
                f"symbol={symbol}&count={page_size}&comment=0"
                f"&page={page}&sort=time"
            )
            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                log.warning(f"[{symbol}] 第 {page} 页请求失败: {e}")
                break

            statuses = data.get("list", [])
            if not statuses:
                break

            for p in statuses:
                text = re.sub(r"<[^>]+>", "", p.get("text", "") or "")
                title = re.sub(r"<[^>]+>", "", p.get("title", "") or "")
                user = p.get("user") or {}

                if not text and not title:
                    continue

                post = {
                    "id": p.get("id"),
                    "title": title,
                    "text": text[:3000],
                    "user_id": user.get("id"),
                    "user_name": user.get("screen_name", ""),
                    "user_description": user.get("description", ""),
                    "followers_count": user.get("followers_count", 0),
                    "verified": user.get("verified", False),
                    "verified_reason": user.get("verified_reason", ""),
                    "created_at": self._ts_to_str(p.get("created_at")),
                    "retweet_count": p.get("retweet_count", 0),
                    "reply_count": p.get("reply_count", 0),
                    "like_count": p.get("like_count", 0),
                    "target": p.get("target", ""),
                    "source": p.get("source", ""),
                }
                all_posts.append(post)

            # 翻页延迟 + 随机抖动
            jitter = delay + random.uniform(0.5, 1.5)
            time.sleep(jitter)

            if page % 5 == 0:
                log.info(f"  [{symbol}] 已爬 {page} 页, 收集 {len(all_posts)} 条")

        log.info(f"[{symbol}] 爬取完成: {len(all_posts)} 条")
        return all_posts

    @staticmethod
    def _ts_to_str(ts) -> str:
        if ts is None:
            return ""
        try:
            return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(ts)


# ═══════════════════════════════════════════════════════════
#  星计划创作者检测
# ═══════════════════════════════════════════════════════════

def is_star_creator(post: dict) -> bool:
    """判断帖子作者是否为星计划创作者"""
    # 1. 粉丝数 > 5000 + 认证用户 → 高概率
    if post.get("followers_count", 0) > 5000 and post.get("verified"):
        reason = post.get("verified_reason", "") or post.get("user_description", "")
        if any(kw in reason for kw in STAR_CREATOR_KEYWORDS):
            return True

    # 2. 简介含星计划关键词
    desc = post.get("user_description", "") or ""
    if any(kw in desc for kw in STAR_CREATOR_KEYWORDS):
        return True

    # 3. 认证原因含关键词
    verified = post.get("verified_reason", "") or ""
    if any(kw in verified for kw in STAR_CREATOR_KEYWORDS):
        return True

    return False


# ═══════════════════════════════════════════════════════════
#  规则粗筛
# ═══════════════════════════════════════════════════════════

def rule_filter(post: dict, min_length: int = 300, min_likes: int = 3) -> bool:
    """规则粗筛：关键词 + 长度 + 互动"""
    text = (post.get("title", "") + " " + post.get("text", "")).lower()

    # 长度检查
    if len(post.get("text", "")) < min_length:
        return False

    # 关键词检查
    has_keyword = any(kw.lower() in text for kw in FUNDAMENTAL_KEYWORDS)
    if not has_keyword:
        return False

    # 互动检查（可选）
    if post.get("like_count", 0) >= min_likes:
        return True
    if post.get("reply_count", 0) >= 3:
        return True

    # 互动不够但有强关键词 → 也保留
    strong_keywords = ["研报", "财报", "深度分析", "基本面", "估值模型"]
    if any(kw in text for kw in strong_keywords):
        return True

    return False


# ═══════════════════════════════════════════════════════════
#  LLM 调用
# ═══════════════════════════════════════════════════════════

def llm_call(
    base_url: str, model: str, api_key: str,
    system_prompt: str, user_msg: str,
    temperature: float = 0.1, timeout: int = 60,
) -> str:
    """通用 LLM 调用"""
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

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def llm_judge_post(
    post_text: str, base_url: str, model: str, api_key: str,
) -> dict:
    """LLM 判断帖子是否为高质量基本面分析"""
    system_prompt = """\
请判断以下帖子是否为基本面/行业分析类内容。

判断标准：
- 有数据支撑（具体数字、财报数据、行业数据）
- 有逻辑链条（因果推理、比较分析）
- 有明确观点（非情绪化、非纯技术面）

只返回 JSON，不要其他内容：
{"keep": true/false, "reason": "简要理由"}
"""
    reply = llm_call(base_url, model, api_key, system_prompt, post_text[:2000])

    # 解析 JSON
    reply = reply.strip()
    if reply.startswith("```"):
        reply = re.sub(r"^```\w*\n?", "", reply)
        reply = re.sub(r"\n?```$", "", reply)
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        return {"keep": False, "reason": f"JSON 解析失败: {reply[:100]}"}


def llm_analyze_post(
    post: dict, base_url: str, model: str, api_key: str,
) -> str:
    """LLM 深度分析帖子"""
    system_prompt = """\
你是资深行业分析师。请对以下雪球帖子进行深度分析，按以下结构输出：

1. **核心观点**（1-2句话总结）
2. **行业逻辑**（竞争格局、产业链位置、趋势判断）
3. **基本面数据**（营收/利润/估值等具体数字，如有）
4. **关键事件/催化剂**（可能影响股价的事件）
5. **风险提示**（潜在风险因素）
6. **分析价值评分**（1-5分，1=信息量低，5=深度研报级）

保持客观中立，直接输出分析结果，不要包裹在代码块中。
"""
    title = post.get("title", "")
    text = post.get("text", "")
    user = post.get("user_name", "")
    date = post.get("created_at", "")
    content = f"标题: {title}\n作者: @{user}\n时间: {date}\n\n正文:\n{text}"

    return llm_call(
        base_url, model, api_key, system_prompt, content,
        temperature=0.3, timeout=90,
    )


# ═══════════════════════════════════════════════════════════
#  输出保存
# ═══════════════════════════════════════════════════════════

def save_analysis(
    post: dict, analysis_text: str, symbol: str,
    author_type: str, output_dir: str,
):
    """保存分析结果到个股目录"""
    base = Path(output_dir) / symbol
    if author_type == "star_creator":
        target_dir = base / "star_creators"
    else:
        target_dir = base / "regular_analyses"

    target_dir.mkdir(parents=True, exist_ok=True)

    date_str = post.get("created_at", "unknown")[:10].replace("-", "")
    author = post.get("user_name", "anonymous").replace(" ", "_")
    post_id = post.get("id", "unknown")

    filename = f"{date_str}_{author}_{post_id}.md"
    filepath = target_dir / filename

    # 如果已存在，跳过
    if filepath.exists():
        log.info(f"  跳过已存在: {filepath}")
        return False

    content = f"""---
title: "{post.get('title', '无标题')}"
author: "@{post.get('user_name', '匿名')}"
author_type: {author_type}
date: "{post.get('created_at', '')}"
symbol: "{symbol}"
source_url: "https://xueqiu.com/{post.get('target', '')}"
---

## 帖子原文

**@{post.get('user_name', '匿名')}** ({post.get('created_at', '')})

{post.get('title', '')}

{post.get('text', '')}

---

## LLM 深度分析

{analysis_text}

---

## 原始数据

- 点赞: {post.get('like_count', 0)}
- 评论: {post.get('reply_count', 0)}
- 转发: {post.get('retweet_count', 0)}
- 粉丝: {post.get('followers_count', 0)}
"""
    filepath.write_text(content, encoding="utf-8")
    log.info(f"  保存: {filepath}")
    return True


def update_index(symbol: str, output_dir: str):
    """更新个股索引"""
    base = Path(output_dir) / symbol
    if not base.exists():
        return

    entries = []
    for subdir in ["star_creators", "regular_analyses"]:
        dir_path = base / subdir
        if not dir_path.exists():
            continue
        for md_file in sorted(dir_path.glob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            m = re.search(r'title:\s*"([^"]*)"', content)
            title = m.group(1) if m else md_file.stem
            m2 = re.search(r'date:\s*"([^"]*)"', content)
            date = m2.group(1) if m2 else ""
            m3 = re.search(r'author:\s*"([^"]*)"', content)
            author = m3.group(1) if m3 else ""
            entries.append({
                "type": "star_creator" if subdir == "star_creators" else "regular",
                "title": title,
                "author": author,
                "date": date,
                "file": str(md_file.relative_to(base)),
            })

    # 写索引
    index_path = base / "_index.md"
    lines = [
        f"# {symbol} 雪球分析索引",
        "",
        f"共 {len(entries)} 篇分析",
        "",
        "| 类型 | 标题 | 作者 | 日期 | 文件 |",
        "|------|------|------|------|------|",
    ]
    for e in entries:
        type_label = "⭐星计划" if e["type"] == "star_creator" else "普通"
        lines.append(
            f"| {type_label} | {e['title']} | {e['author']} | {e['date']} | [{e['file']}]({e['file']}) |"
        )
    lines.append("")
    index_path.write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def process_stock(
    code: str, fetcher: XueqiuFetcher,
    llm_base_url: str, llm_model: str, llm_api_key: str,
    output_dir: str, max_pages: int = 5,
    crawl_only: bool = False,
) -> dict:
    """单股处理管线"""
    symbol = to_xueqiu_symbol(code)
    log.info(f"\n{'='*50}")
    log.info(f"处理: {symbol}")
    log.info(f"{'='*50}")

    # ① 爬取
    posts = fetcher.fetch_posts(symbol, max_pages=max_pages, delay=2.0)
    if not posts:
        log.info(f"[{symbol}] 无帖子数据")
        return {"symbol": symbol, "total": 0}

    stats = {"total": len(posts), "star": 0, "regular_kept": 0, "regular_skipped": 0, "saved": 0}

    # ② 分流处理
    for i, post in enumerate(posts):
        creator = is_star_creator(post)
        if creator:
            stats["star"] += 1

        log.info(
            f"  [{i+1}/{len(posts)}] @{post['user_name']} "
            f"| 粉丝:{post['followers_count']} "
            f"| {'⭐星计划' if creator else '普通用户'} "
            f"| 点赞:{post['like_count']} 评论:{post['reply_count']}"
        )

        if crawl_only:
            continue

        if creator:
            # 星计划 → 直接深度分析
            try:
                analysis = llm_analyze_post(post, llm_base_url, llm_model, llm_api_key)
                save_analysis(post, analysis, symbol, "star_creator", output_dir)
                stats["saved"] += 1
            except Exception as e:
                log.warning(f"    LLM 分析失败: {e}")
        else:
            # 普通用户 → 规则粗筛 + LLM 判断
            if not rule_filter(post):
                stats["regular_skipped"] += 1
                continue

            stats["regular_kept"] += 1

            try:
                result = llm_judge_post(post["text"], llm_base_url, llm_model, llm_api_key)
                if result.get("keep"):
                    log.info(f"    LLM 判定: 保留 ({result.get('reason', '')})")
                    analysis = llm_analyze_post(post, llm_base_url, llm_model, llm_api_key)
                    save_analysis(post, analysis, symbol, "regular", output_dir)
                    stats["saved"] += 1
                else:
                    log.info(f"    LLM 判定: 跳过 ({result.get('reason', '')})")
                    stats["regular_skipped"] += 1
            except Exception as e:
                log.warning(f"    LLM 判断失败: {e}")

        # 帖子间延迟
        time.sleep(random.uniform(0.5, 1.5))

    # ③ 更新索引
    update_index(symbol, output_dir)

    log.info(f"\n[{symbol}] 统计: {stats}")
    return {"symbol": symbol, **stats}


def main():
    parser = argparse.ArgumentParser(description="雪球行业分析深度文章采集")
    parser.add_argument("--codes", required=True, help="股票代码，逗号分隔")
    parser.add_argument("--cookie", default="", help="雪球 cookie 或 xq_a_token")
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pages", type=int, default=5, help="每只股票最大爬取页数")
    parser.add_argument("--days", type=int, default=1, help="采集天数（影响页数估算）")
    parser.add_argument("--crawl-only", action="store_true", help="仅采集不分析")

    args = parser.parse_args()

    # cookie
    cookie = args.cookie or os.environ.get("XQ_COOKIE", "")
    if not cookie:
        log.error("需要雪球 cookie: --cookie 或 XQ_COOKIE 环境变量")
        log.info("获取方式: 浏览器登录 xueqiu.com → F12 → Application → Cookies → xq_a_token")
        sys.exit(1)

    # LLM API
    api_key = (
        args.llm_api_key
        or os.environ.get("DASHSCOPE_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key and not args.crawl_only:
        log.error("需要 LLM API Key（或使用 --crawl-only 仅采集）")
        sys.exit(1)

    # 解析代码
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    if not codes:
        log.error("无股票代码")
        sys.exit(1)

    # 估算页数（每天约 1 页）
    max_pages = max(args.max_pages, args.days)

    log.info(f"开始: {len(codes)} 只股票, max_pages={max_pages}")
    if args.crawl_only:
        log.info("模式: 仅采集（crawl-only）")

    fetcher = XueqiuFetcher(cookie)

    total_stats = {"total": 0, "star": 0, "regular_kept": 0, "regular_skipped": 0, "saved": 0}

    for code in codes:
        result = process_stock(
            code, fetcher,
            args.llm_base_url, args.llm_model, api_key,
            args.output_dir, max_pages=max_pages,
            crawl_only=args.crawl_only,
        )
        for k in total_stats:
            total_stats[k] += result.get(k, 0)

        # 股票间冷却
        if len(codes) > 1:
            time.sleep(random.uniform(3, 6))

    log.info(f"\n{'='*50}")
    log.info(f"总计: {total_stats}")
    log.info(f"{'='*50}")


if __name__ == "__main__":
    main()
