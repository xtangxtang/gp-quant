"""雪球讨论批量采集 + Wiki 生成 — 两阶段模式

两个阶段:
  Phase 1 (crawl):   按股票逐只爬取 → 按日期拆分保存 raw/YYYYMMDD.json
  Phase 2 (process): 按日期从旧到新 → 对每只已爬股票做 LLM 过滤 → summary → wiki

Phase 1 支持两种模式:
  - 快速模式 (--recent-days N): 只爬最近 N 天, 每只 1~2 页, ~1 天完成全部
  - 全量模式 (默认):            爬全部历史, 每只 ~50 页, ~30 天完成

断点续传:
  Phase 1 快速: raw/_recent_done 标记 → 跳过
  Phase 1 全量: raw/_crawl_done 标记 → 跳过
  Phase 2:      YYYYMMDD_filtered.json → 跳过 LLM

用法:
    # 快速: 先爬最近 2 天 (约 1 天完成 5500 只)
    python -m src.downloader.xueqiu_batch_history crawl \\
        --stock-csv /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv \\
        --recent-days 2 --resume

    # 全量: 慢慢爬全部历史
    python -m src.downloader.xueqiu_batch_history crawl \\
        --stock-csv /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv \\
        --resume

    # Phase 2: 按日期处理 wiki
    python -m src.downloader.xueqiu_batch_history process --resume

    # 单只股票爬取
    python -m src.downloader.xueqiu_batch_history crawl --codes 600519
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path

from src.downloader.xueqiu_order_monitor import (
    XueqiuSession,
    _to_xueqiu_symbol,
    _ts_to_date,
    _parse_posts,
    XUEQIU_STOCK_COMMENT_URL,
    WIKI_PAGES,
    SCHEMA_TEMPLATE,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    LLMTooSlowError,
    llm_filter_posts,
    _prepare_posts_text,
    _generate_wiki_page,
    _generate_index,
    _append_log,
    _save_summary_md,
    _load_codes_from_csv,
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "/nvme5/xtang/gp-workspace/gp-data/xueqiu_orders"


# ═══════════════════════════════════════════════════════════
#  Phase 1: Crawl — 按股票爬取, 按日期拆分保存
# ═══════════════════════════════════════════════════════════

def fetch_history_posts(
    xq: XueqiuSession,
    symbol: str,
    start_date: date,
    end_date: date,
    page_size: int = 100,
    max_pages: int = 2000,
    delay: float = 2.0,
) -> list[dict]:
    """一次性爬取 [start_date, end_date] 区间所有讨论 (从新到旧分页)"""
    all_posts = []
    page_num = 1
    first_page_failed = False

    while page_num <= max_pages:
        url = (
            f"{XUEQIU_STOCK_COMMENT_URL}?"
            f"symbol={symbol}&count={page_size}&comment=0"
            f"&page={page_num}&sort=time"
        )
        try:
            # retries=1: WAF 快速失败 (省 ~135s/次), 不做无效重试
            data = xq.fetch_json(url, retries=1)
        except Exception as e:
            log.warning(f"[{symbol}] 第 {page_num} 页请求失败: {e}")
            if page_num == 1:
                first_page_failed = True
            break

        statuses = data.get("list", [])
        if not statuses:
            break

        past_range = False
        for p in statuses:
            post_date = _ts_to_date(p.get("created_at"))
            if post_date is None:
                continue
            if post_date < start_date:
                past_range = True
                break
            if post_date <= end_date:
                all_posts.append(p)

        if past_range:
            break

        page_num += 1
        if page_num <= max_pages:
            jitter = delay + random.uniform(1.0, delay)
            time.sleep(jitter)

        if page_num % 20 == 0:
            log.info(
                f"  [{symbol}] 已爬 {page_num} 页, "
                f"收集 {len(all_posts)} 条"
            )

        # 每 30 页中途暂停, 拆散连续请求避免触发 WAF 阈值
        if page_num % 30 == 0 and page_num > 1:
            pause = random.uniform(60, 90)
            log.info(
                f"  [{symbol}] 中途冷却 {pause:.0f}s "
                f"(已爬 {page_num} 页)"
            )
            time.sleep(pause)

    log.info(
        f"[{symbol}] 历史爬取: {start_date}~{end_date}, "
        f"{len(all_posts)} 条, {page_num} 页"
    )
    # 第 1 页就失败 = WAF 封锁, 返回 None 区分于 "真的无帖子"(空 list)
    if first_page_failed and not all_posts:
        return None
    return all_posts


def _split_posts_by_date(posts: list[dict]) -> dict[str, list[dict]]:
    """将 parsed posts 按 created_at 日期分组, 返回 {YYYYMMDD: [post, ...]}"""
    by_date: dict[str, list[dict]] = defaultdict(list)
    for p in posts:
        dt_str = p.get("created_at", "")
        if not dt_str:
            continue
        # created_at 格式: "2025-04-14 10:30:00" (由 _parse_posts 产生)
        date_key = dt_str[:10].replace("-", "")
        if len(date_key) == 8 and date_key.isdigit():
            by_date[date_key].append(p)
    return dict(by_date)


def crawl_stock(
    code: str,
    xq: XueqiuSession,
    start_date: date,
    end_date: date,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    page_delay: float = 2.0,
) -> dict:
    """Phase 1: 爬取单只股票全部历史, 按日期拆分保存 raw/YYYYMMDD.json"""
    symbol = _to_xueqiu_symbol(code)
    stock_dir = Path(output_dir) / symbol
    raw_dir = stock_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ① 爬取
    raw_posts = fetch_history_posts(
        xq, symbol, start_date, end_date, delay=page_delay,
    )
    if raw_posts is None:
        # WAF 封锁, 不标记完成, 下次 --resume 会重试
        log.warning(f"[{symbol}] WAF 封锁, 跳过 (不标记完成)")
        return {"symbol": symbol, "status": "waf_blocked", "total": 0}
    if not raw_posts:
        log.info(f"[{symbol}] 无历史讨论")
        (raw_dir / "_crawl_done").write_text(
            f"completed={datetime.now().isoformat()}\ntotal=0\n",
            encoding="utf-8",
        )
        return {"symbol": symbol, "status": "no_data", "total": 0}

    # ② 解析
    posts = _parse_posts(raw_posts)
    log.info(f"[{symbol}] 解析 {len(posts)} 条有效帖子")

    # ③ 按日期拆分保存
    by_date = _split_posts_by_date(posts)
    saved_dates = 0
    for date_key, day_posts in sorted(by_date.items()):
        json_path = raw_dir / f"{date_key}.json"
        if not json_path.exists():
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(day_posts, f, ensure_ascii=False)
            saved_dates += 1

    # ④ 标记完成
    (raw_dir / "_crawl_done").write_text(
        f"completed={datetime.now().isoformat()}\n"
        f"total={len(posts)}\n"
        f"dates={saved_dates}\n"
        f"date_range={min(by_date.keys())}~{max(by_date.keys())}\n",
        encoding="utf-8",
    )

    log.info(
        f"[{symbol}] 爬取完成: {len(posts)} 条 → "
        f"{saved_dates} 天, {min(by_date.keys())}~{max(by_date.keys())}"
    )
    return {
        "symbol": symbol, "status": "crawled",
        "total": len(posts), "dates": saved_dates,
    }


def crawl_stock_recent(
    code: str,
    xq: XueqiuSession,
    recent_days: int,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    page_delay: float = 2.0,
) -> dict:
    """快速模式: 只爬最近 N 天的帖子, 通常 1~3 页即完成"""
    symbol = _to_xueqiu_symbol(code)
    stock_dir = Path(output_dir) / symbol
    raw_dir = stock_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    from datetime import timedelta
    end_date = date.today()
    start_date = end_date - timedelta(days=recent_days)

    raw_posts = fetch_history_posts(
        xq, symbol, start_date, end_date, delay=page_delay,
    )
    if raw_posts is None:
        log.warning(f"[{symbol}] WAF 封锁, 跳过")
        return {"symbol": symbol, "status": "waf_blocked", "total": 0}
    if not raw_posts:
        log.info(f"[{symbol}] 最近 {recent_days} 天无讨论")
        (raw_dir / "_recent_done").write_text(
            f"completed={datetime.now().isoformat()}\ndays={recent_days}\ntotal=0\n",
            encoding="utf-8",
        )
        return {"symbol": symbol, "status": "no_data", "total": 0}

    posts = _parse_posts(raw_posts)
    by_date = _split_posts_by_date(posts)

    # 保存 (合并已有同日数据)
    saved_dates = 0
    for date_key, day_posts in sorted(by_date.items()):
        json_path = raw_dir / f"{date_key}.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                existing = json.load(f)
            # 合并去重 (按 id)
            seen_ids = {p.get("id") for p in existing}
            merged = existing + [p for p in day_posts if p.get("id") not in seen_ids]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False)
        else:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(day_posts, f, ensure_ascii=False)
        saved_dates += 1

    (raw_dir / "_recent_done").write_text(
        f"completed={datetime.now().isoformat()}\n"
        f"days={recent_days}\n"
        f"total={len(posts)}\n"
        f"dates={saved_dates}\n"
        f"date_range={min(by_date.keys())}~{max(by_date.keys())}\n",
        encoding="utf-8",
    )

    log.info(
        f"[{symbol}] 近{recent_days}天: {len(posts)} 条 → "
        f"{saved_dates} 天, {min(by_date.keys())}~{max(by_date.keys())}"
    )
    return {
        "symbol": symbol, "status": "crawled",
        "total": len(posts), "dates": saved_dates,
    }


def run_phase1_recent(args):
    """Phase 1 快速模式: 只爬最近 N 天, 每只 1~2 页, 低 WAF 风险"""
    if args.stock_csv:
        codes = _load_codes_from_csv(args.stock_csv)
        log.info(f"从 CSV 加载 {len(codes)} 只股票")
    else:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]

    if args.limit > 0:
        codes = codes[:args.limit]

    log.info(
        f"Phase 1 快速模式: {len(codes)} 只股票, "
        f"最近 {args.recent_days} 天, page_delay={args.page_delay}s"
    )

    xq = XueqiuSession()
    stats = defaultdict(int)
    waf_retry_queue = []
    consecutive_waf = 0

    try:
        for i, code in enumerate(codes):
            sym = _to_xueqiu_symbol(code)

            if args.resume:
                marker = Path(args.output_dir) / sym / "raw" / "_recent_done"
                if marker.exists():
                    stats["resumed"] += 1
                    if stats["resumed"] % 500 == 0:
                        log.info(f"已跳过 {stats['resumed']} 只")
                    continue

            # 快速模式: 短延迟即可 (每只仅 1~2 页, WAF 风险低)
            if i > 0 and consecutive_waf == 0:
                time.sleep(random.uniform(3.0, 6.0))

            try:
                result = crawl_stock_recent(
                    code, xq,
                    recent_days=args.recent_days,
                    output_dir=args.output_dir,
                    page_delay=args.page_delay,
                )
                status = result.get("status", "fail")
                stats[status] += 1
                total = result.get("total", 0)

                if status == "crawled":
                    consecutive_waf = 0
                    # 按帖子量分级: 热门股换代理 (预防WAF), 普通股短延迟
                    if total > 100:
                        log.info(f"  热门股 ({total}条), 轮换代理")
                        xq.refresh_session(cooldown=5, rotate=True)
                    elif total > 50:
                        log.info(f"  帖子较多 ({total}条), 轮换代理")
                        xq.refresh_session(cooldown=3, rotate=True)
                    elif total > 20:
                        time.sleep(random.uniform(5, 10))
                    # else: 下轮 3-6s 延迟够了

                elif status == "waf_blocked":
                    consecutive_waf += 1
                    waf_retry_queue.append(code)
                    # WAF → 直接换代理, 不用长等 (新 IP 没被封)
                    log.warning(
                        f"  WAF (连续第{consecutive_waf}次), 轮换代理"
                    )
                    xq.refresh_session(cooldown=5, rotate=True)

                elif status == "no_data":
                    consecutive_waf = 0

            except Exception as e:
                log.error(f"[{code}] 异常: {e}")
                stats["fail"] += 1

            done = i + 1
            if done % 50 == 0 or done == len(codes):
                log.info(
                    f"[快速] 进度: {done}/{len(codes)} | "
                    + " | ".join(
                        f"{k}: {v}" for k, v in sorted(stats.items())
                    )
                    + (f" | 待重试: {len(waf_retry_queue)}" if waf_retry_queue else "")
                )

        # WAF 重试
        if waf_retry_queue:
            log.info(f"\nWAF 重试: {len(waf_retry_queue)} 只")
            for j, code in enumerate(waf_retry_queue):
                sym = _to_xueqiu_symbol(code)
                if (Path(args.output_dir) / sym / "raw" / "_recent_done").exists():
                    continue
                if j > 0:
                    time.sleep(random.uniform(5, 10))
                try:
                    result = crawl_stock_recent(
                        code, xq,
                        recent_days=args.recent_days,
                        output_dir=args.output_dir,
                        page_delay=args.page_delay,
                    )
                    if result.get("status") == "waf_blocked":
                        xq.refresh_session(cooldown=5, rotate=True)
                except Exception:
                    pass
    finally:
        xq.close()

    log.info(
        f"\n快速模式完成: {len(codes)} 只\n"
        + "\n".join(f"  {k}: {v}" for k, v in sorted(stats.items()))
    )


def run_phase1(args):
    """Phase 1 全量模式: 逐只股票爬取全部历史"""
    if args.stock_csv:
        codes = _load_codes_from_csv(args.stock_csv)
        log.info(f"从 CSV 加载 {len(codes)} 只股票")
    else:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]

    if args.limit > 0:
        codes = codes[:args.limit]

    start = (
        datetime.strptime(args.start_date, "%Y%m%d").date()
        if args.start_date
        else date(2020, 1, 1)
    )
    end = (
        datetime.strptime(args.end_date, "%Y%m%d").date()
        if args.end_date
        else date.today()
    )

    log.info(
        f"Phase 1 (Crawl): {len(codes)} 只股票, "
        f"{start} ~ {end}, page_delay={args.page_delay}s"
    )

    xq = XueqiuSession()
    stats = defaultdict(int)
    waf_retry_queue = []  # WAF 封锁的股票, 最后统一重试
    need_delay = False    # 标记下一只是否需要基础延迟
    consecutive_waf = 0   # 连续 WAF 封锁计数, 用于递增冷却

    try:
        for i, code in enumerate(codes):
            sym = _to_xueqiu_symbol(code)

            # Resume: 已爬完就跳过
            if args.resume:
                done_marker = Path(args.output_dir) / sym / "raw" / "_crawl_done"
                if done_marker.exists():
                    stats["resumed"] += 1
                    if stats["resumed"] % 100 == 0:
                        log.info(f"已跳过 {stats['resumed']} 只")
                    continue

            # 基础延迟 (仅在上一只是 no_data 等轻量情况时)
            if need_delay:
                time.sleep(random.uniform(5.0, 10.0))
                need_delay = False

            try:
                result = crawl_stock(
                    code, xq,
                    start_date=start,
                    end_date=end,
                    output_dir=args.output_dir,
                    page_delay=args.page_delay,
                )
                status = result.get("status", "fail")
                stats[status] += 1
                total = result.get("total", 0)

                if status == "crawled":
                    consecutive_waf = 0
                    # ── 按爬取量分级冷却 (中途已暂停, 此处可较短) ──
                    if total > 500:
                        cool = random.uniform(60, 120)
                        log.info(f"  大量帖子 ({total}), 冷却 {cool:.0f}s + 刷新会话")
                        xq.refresh_session(cooldown=cool)
                    elif total > 200:
                        cool = random.uniform(30, 60)
                        log.info(f"  帖子多 ({total}), 冷却 {cool:.0f}s + 刷新会话")
                        xq.refresh_session(cooldown=cool)
                    elif total > 50:
                        time.sleep(random.uniform(15, 30))
                    else:
                        time.sleep(random.uniform(8, 15))

                elif status == "waf_blocked":
                    # ── WAF → 立即长冷却 + 刷新, 连续 WAF 递增 ──
                    consecutive_waf += 1
                    waf_retry_queue.append(code)
                    base_cool = 300 + consecutive_waf * 120  # 300/420/540/...
                    cool = random.uniform(base_cool, base_cool + 120)
                    cool = min(cool, 900)  # 上限 15 分钟
                    log.warning(
                        f"  WAF 封锁 (连续第{consecutive_waf}次), "
                        f"冷却 {cool:.0f}s + 刷新会话"
                    )
                    xq.refresh_session(cooldown=cool)

                elif status == "no_data":
                    need_delay = True
                else:
                    need_delay = True

            except Exception as e:
                log.error(f"[{code}] 异常: {e}")
                stats["fail"] += 1
                need_delay = True

            done = i + 1
            if done % 10 == 0 or done == len(codes):
                log.info(
                    f"[Phase1] 进度: {done}/{len(codes)} | "
                    + " | ".join(
                        f"{k}: {v}" for k, v in sorted(stats.items())
                    )
                    + (f" | 待重试: {len(waf_retry_queue)}" if waf_retry_queue else "")
                )

        # ── WAF 重试队列: 全部遍历后统一重试 ──────────
        if waf_retry_queue:
            log.info(
                f"\n{'='*50}\n"
                f"WAF 重试队列: {len(waf_retry_queue)} 只股票\n"
                f"{'='*50}"
            )
            retry_ok = 0
            retry_fail = 0
            for j, code in enumerate(waf_retry_queue):
                sym = _to_xueqiu_symbol(code)
                done_marker = Path(args.output_dir) / sym / "raw" / "_crawl_done"
                if done_marker.exists():
                    retry_ok += 1
                    continue

                # 重试间较长延迟
                if j > 0:
                    time.sleep(random.uniform(30, 60))

                try:
                    result = crawl_stock(
                        code, xq,
                        start_date=start,
                        end_date=end,
                        output_dir=args.output_dir,
                        page_delay=args.page_delay,
                    )
                    st = result.get("status", "fail")
                    if st == "crawled":
                        retry_ok += 1
                        stats["retry_ok"] += 1
                        total = result.get("total", 0)
                        if total > 200:
                            xq.refresh_session(cooldown=random.uniform(60, 120))
                        else:
                            time.sleep(random.uniform(15, 30))
                    elif st == "waf_blocked":
                        retry_fail += 1
                        stats["retry_waf"] += 1
                        xq.refresh_session(cooldown=random.uniform(180, 300))
                    else:
                        retry_ok += 1
                except Exception as e:
                    log.error(f"[{code}] 重试异常: {e}")
                    retry_fail += 1

                if (j + 1) % 5 == 0:
                    log.info(f"[重试] {j+1}/{len(waf_retry_queue)} | ok: {retry_ok} | fail: {retry_fail}")

            log.info(f"WAF 重试完成: ok={retry_ok}, fail={retry_fail}")

    finally:
        xq.close()

    log.info(
        f"\nPhase 1 完成: {len(codes)} 只\n"
        + "\n".join(f"  {k}: {v}" for k, v in sorted(stats.items()))
    )


# ═══════════════════════════════════════════════════════════
#  Phase 2: Process — 按日期从旧到新, 逐股票 LLM 过滤 + Wiki
# ═══════════════════════════════════════════════════════════

def _collect_pending_tasks(
    output_dir: str,
    resume: bool = False,
) -> dict[str, list[str]]:
    """扫描所有已爬股票目录, 收集待处理的 {YYYYMMDD: [symbol, ...]}.

    只返回有 raw/YYYYMMDD.json 的条目。
    如果 resume=True, 跳过已有 YYYYMMDD_filtered.json 的。
    """
    base = Path(output_dir)
    date_stocks: dict[str, list[str]] = defaultdict(list)

    if not base.exists():
        return {}

    for stock_dir in sorted(base.iterdir()):
        if not stock_dir.is_dir():
            continue
        raw_dir = stock_dir / "raw"
        if not raw_dir.exists():
            continue

        symbol = stock_dir.name
        for json_file in sorted(raw_dir.glob("????????.json")):
            date_key = json_file.stem
            if not date_key.isdigit() or len(date_key) != 8:
                continue
            # resume: 已有过滤缓存则跳过
            if resume:
                filtered_path = raw_dir / f"{date_key}_filtered.json"
                if filtered_path.exists():
                    continue
            date_stocks[date_key].append(symbol)

    return dict(date_stocks)


def _process_stock_date(
    symbol: str,
    date_key: str,
    output_dir: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    llm_timeout: float,
) -> dict:
    """处理单只股票的单日数据: LLM 过滤 → summary → wiki 增量更新"""
    stock_dir = Path(output_dir) / symbol
    raw_dir = stock_dir / "raw"
    wiki_dir = stock_dir / "wiki"
    schema_path = stock_dir / "SCHEMA.md"
    data_date = f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:8]}"

    json_path = raw_dir / f"{date_key}.json"
    filtered_cache = raw_dir / f"{date_key}_filtered.json"

    # 加载帖子
    with open(json_path, encoding="utf-8") as f:
        posts = json.load(f)

    if not posts:
        # 保存空缓存标记
        with open(filtered_cache, "w", encoding="utf-8") as f:
            json.dump([], f)
        return {"symbol": symbol, "date": date_key, "status": "empty"}

    # ── LLM 过滤 (有缓存则跳过) ──────────────────────
    if filtered_cache.exists():
        with open(filtered_cache, encoding="utf-8") as f:
            filtered = json.load(f)
        log.info(f"  [{symbol}/{date_key}] 过滤缓存: {len(filtered)} 条")
    else:
        try:
            filtered = llm_filter_posts(
                posts, llm_base_url, llm_model, llm_api_key,
                batch_timeout=llm_timeout,
            )
        except LLMTooSlowError as e:
            log.warning(f"  [{symbol}/{date_key}] LLM 太慢 ({e}), 跳过")
            return {"symbol": symbol, "date": date_key, "status": "llm_slow"}
        except Exception as e:
            log.error(f"  [{symbol}/{date_key}] LLM 过滤失败: {e}")
            return {"symbol": symbol, "date": date_key, "status": "fail"}

        # 保存过滤结果 (即使为空也保存, 避免重复调 LLM)
        with open(filtered_cache, "w", encoding="utf-8") as f:
            json.dump(filtered if filtered else [], f, ensure_ascii=False)

        if filtered:
            log.info(f"  [{symbol}/{date_key}] 过滤: {len(posts)} → {len(filtered)} 条")
            _save_summary_md(raw_dir, date_key, symbol, data_date, posts, filtered)
        else:
            log.info(f"  [{symbol}/{date_key}] 过滤后无基本面帖子")

    # 过滤后无内容 → 不更新 wiki
    if not filtered:
        return {"symbol": symbol, "date": date_key, "status": "no_relevant",
                "total": len(posts), "filtered": 0}

    # ── Wiki 增量更新 ─────────────────────────────────
    wiki_dir.mkdir(parents=True, exist_ok=True)
    if not schema_path.exists():
        schema_path.write_text(
            SCHEMA_TEMPLATE.format(symbol=symbol), encoding="utf-8",
        )

    posts_text = _prepare_posts_text(filtered)
    pages_updated = []

    for page_name, page_def in WIKI_PAGES.items():
        page_path = wiki_dir / f"{page_name}.md"
        existing = (
            page_path.read_text(encoding="utf-8")
            if page_path.exists()
            else None
        )
        try:
            content = _generate_wiki_page(
                page_name, page_def, symbol, data_date,
                posts_text, existing,
                llm_base_url, llm_model, llm_api_key,
            )
            page_path.write_text(content, encoding="utf-8")
            pages_updated.append(page_name)
        except Exception as e:
            log.error(f"    [wiki] {page_name}.md 生成失败: {e}")

    if pages_updated:
        idx = _generate_index(symbol, wiki_dir, data_date)
        (wiki_dir / "index.md").write_text(idx, encoding="utf-8")
        _append_log(
            wiki_dir / "log.md", symbol, data_date,
            len(posts), len(filtered), pages_updated,
        )

    log.info(
        f"  [{symbol}/{date_key}] Wiki: {len(filtered)} 条, "
        f"{len(pages_updated)}/{len(WIKI_PAGES)} 页"
    )

    # 删除原始 JSON (已有 filtered + summary)
    if pages_updated and json_path.exists():
        json_path.unlink()

    return {
        "symbol": symbol, "date": date_key, "status": "ok",
        "total": len(posts), "filtered": len(filtered),
        "pages": len(pages_updated),
    }


def run_phase2(args):
    """Phase 2 主流程: 按日期从旧到新, 逐股票处理"""
    api_key = (
        args.llm_api_key
        or os.environ.get("DASHSCOPE_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        log.error("需要 LLM API Key (--llm-api-key 或 DASHSCOPE_API_KEY 环境变量)")
        sys.exit(1)

    # 扫描待处理任务
    log.info(f"扫描 {args.output_dir} ...")
    date_stocks = _collect_pending_tasks(args.output_dir, resume=args.resume)

    if not date_stocks:
        log.info("无待处理数据")
        return

    all_dates = sorted(date_stocks.keys())  # 从旧到新

    # 可选: 限定日期范围
    if args.start_date:
        start_key = args.start_date.replace("-", "")
        all_dates = [d for d in all_dates if d >= start_key]
    if args.end_date:
        end_key = args.end_date.replace("-", "")
        all_dates = [d for d in all_dates if d <= end_key]

    if args.limit > 0:
        all_dates = all_dates[:args.limit]

    total_tasks = sum(len(date_stocks.get(d, [])) for d in all_dates)
    log.info(
        f"Phase 2 (Process): {len(all_dates)} 天, "
        f"{total_tasks} 个待处理 (stock×date)"
    )

    stats = defaultdict(int)
    processed = 0

    for date_key in all_dates:
        symbols = sorted(date_stocks.get(date_key, []))
        if not symbols:
            continue

        log.info(
            f"\n{'='*50}\n"
            f"日期: {date_key} ({len(symbols)} 只股票)\n"
            f"{'='*50}"
        )

        for symbol in symbols:
            try:
                result = _process_stock_date(
                    symbol, date_key, args.output_dir,
                    args.llm_base_url, args.llm_model,
                    api_key, args.llm_timeout,
                )
                status = result.get("status", "fail")
                stats[status] += 1
            except Exception as e:
                log.error(f"  [{symbol}/{date_key}] 异常: {e}")
                stats["fail"] += 1

            processed += 1
            if processed % 50 == 0:
                log.info(
                    f"[Phase2] 进度: {processed}/{total_tasks} | "
                    + " | ".join(
                        f"{k}: {v}" for k, v in sorted(stats.items())
                    )
                )

    log.info(
        f"\nPhase 2 完成: {processed} 个任务\n"
        + "\n".join(f"  {k}: {v}" for k, v in sorted(stats.items()))
    )


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="雪球讨论批量采集 + Wiki (两阶段: crawl → process)",
    )
    subparsers = parser.add_subparsers(dest="phase", help="运行阶段")

    # ── Phase 1: crawl ──
    p1 = subparsers.add_parser("crawl", help="Phase 1: 按股票爬取 → 按日期拆分保存")
    group1 = p1.add_mutually_exclusive_group(required=True)
    group1.add_argument("--codes", help="股票代码, 逗号分隔")
    group1.add_argument("--stock-csv", help="股票列表 CSV (含 symbol 列)")
    p1.add_argument("--start-date", default=None, help="起始日期 YYYYMMDD")
    p1.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD")
    p1.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p1.add_argument("--page-delay", type=float, default=5.0, help="翻页延迟秒数")
    p1.add_argument("--resume", action="store_true", help="跳过已爬完的股票")
    p1.add_argument("--limit", type=int, default=0, help="最多处理 N 只股票")
    p1.add_argument("--recent-days", type=int, default=0,
                     help="快速模式: 只爬最近 N 天 (0=全量)")

    # ── Phase 2: process ──
    p2 = subparsers.add_parser("process", help="Phase 2: 按日期从旧到新处理 wiki")
    p2.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p2.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    p2.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p2.add_argument("--llm-api-key", default="")
    p2.add_argument("--llm-timeout", type=float, default=120.0,
                     help="LLM 单批次超时秒数")
    p2.add_argument("--start-date", default=None, help="只处理此日期之后 YYYYMMDD")
    p2.add_argument("--end-date", default=None, help="只处理此日期之前 YYYYMMDD")
    p2.add_argument("--resume", action="store_true", help="跳过已有 filtered 缓存的")
    p2.add_argument("--limit", type=int, default=0, help="最多处理 N 天")

    args = parser.parse_args()

    if args.phase == "crawl":
        if getattr(args, 'recent_days', 0) > 0:
            run_phase1_recent(args)
        else:
            run_phase1(args)
    elif args.phase == "process":
        run_phase2(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
