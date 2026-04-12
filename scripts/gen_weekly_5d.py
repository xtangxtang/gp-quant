#!/usr/bin/env python3
"""
从日线行情生成5交易日窗口的周线行情，字段与日线完全一致。

聚合规则：
- ts_code: 取自日线
- trade_date: 取窗口最后一天
- open: 窗口第一天的 open
- high: 窗口内 max(high)
- low: 窗口内 min(low)
- close: 窗口最后一天的 close
- pre_close: 窗口第一天的 pre_close
- change: close - pre_close
- pct_chg: change / pre_close * 100
- vol/amount: sum
- circ_mv/total_mv/float_share/free_share/total_share: 取最后一天
- dv_ratio/dv_ttm/pb/pe/pe_ttm/ps/ps_ttm: 取最后一天
- turnover_rate/turnover_rate_f: sum
- volume_ratio: 取最后一天
- 资金流字段 (buy_*/sell_*/net_mf_*): sum
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

DATA_DIR = "/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full"
OUT_DIR = "/nvme5/xtang/gp-workspace/gp-data/tushare-weekly-5d"
WINDOW = 5

# 字段分类
FIRST_COLS = ["open", "pre_close"]
LAST_COLS = [
    "close", "circ_mv", "dv_ratio", "dv_ttm", "float_share", "free_share",
    "pb", "pe", "pe_ttm", "ps", "ps_ttm", "total_mv", "total_share",
    "volume_ratio",
]
MAX_COLS = ["high"]
MIN_COLS = ["low"]
SUM_COLS = [
    "vol", "amount", "turnover_rate", "turnover_rate_f",
    "buy_elg_amount", "buy_elg_vol", "buy_lg_amount", "buy_lg_vol",
    "buy_md_amount", "buy_md_vol", "buy_sm_amount", "buy_sm_vol",
    "net_mf_amount", "net_mf_vol",
    "sell_elg_amount", "sell_elg_vol", "sell_lg_amount", "sell_lg_vol",
    "sell_md_amount", "sell_md_vol", "sell_sm_amount", "sell_sm_vol",
]
# change 和 pct_chg 需要重新计算


def process_file(filepath):
    """处理单个股票的日线文件，生成周线。"""
    try:
        full_cols = [
            "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
            "change", "pct_chg", "vol", "amount", "circ_mv", "dv_ratio", "dv_ttm",
            "float_share", "free_share", "pb", "pe", "pe_ttm", "ps", "ps_ttm",
            "total_mv", "total_share", "turnover_rate", "turnover_rate_f", "volume_ratio",
            "buy_elg_amount", "buy_elg_vol", "buy_lg_amount", "buy_lg_vol",
            "buy_md_amount", "buy_md_vol", "buy_sm_amount", "buy_sm_vol",
            "net_mf_amount", "net_mf_vol", "sell_elg_amount", "sell_elg_vol",
            "sell_lg_amount", "sell_lg_vol", "sell_md_amount", "sell_md_vol",
            "sell_sm_amount", "sell_sm_vol",
        ]
        # 跳过 header，用完整 44 列名强制解析；列不够的行自动填 NaN
        df = pd.read_csv(
            filepath, header=None, skiprows=1,
            names=full_cols, dtype={"trade_date": str},
        )
        if len(df) < WINDOW:
            return filepath, 0, None

        # 按日期升序排列
        df.sort_values("trade_date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 计算每行属于哪个5日窗口（从头开始，每5天一组）
        n = len(df)
        full_groups = n // WINDOW
        if full_groups == 0:
            return filepath, 0, None

        # 只处理完整的5日窗口
        df_cut = df.iloc[: full_groups * WINDOW].copy()
        df_cut["group"] = np.arange(len(df_cut)) // WINDOW

        rows = []
        for g, grp in df_cut.groupby("group"):
            row = {}
            row["ts_code"] = grp.iloc[0]["ts_code"]
            row["trade_date"] = grp.iloc[-1]["trade_date"]  # 窗口最后一天

            for c in FIRST_COLS:
                row[c] = grp.iloc[0][c]
            for c in LAST_COLS:
                row[c] = grp.iloc[-1][c]
            for c in MAX_COLS:
                row[c] = grp[c].max()
            for c in MIN_COLS:
                row[c] = grp[c].min()
            for c in SUM_COLS:
                row[c] = grp[c].sum(min_count=1)  # 全 NaN 则为 NaN

            # 重新计算 change 和 pct_chg
            pre = row["pre_close"]
            close = row["close"]
            if pd.notna(pre) and pd.notna(close) and pre != 0:
                row["change"] = round(close - pre, 4)
                row["pct_chg"] = round((close - pre) / pre * 100, 4)
            else:
                row["change"] = np.nan
                row["pct_chg"] = np.nan

            rows.append(row)

        # 保持与日线相同的列顺序
        col_order = list(df.columns)
        result = pd.DataFrame(rows, columns=col_order)

        # 写出
        basename = os.path.basename(filepath)
        out_path = os.path.join(OUT_DIR, basename)
        result.to_csv(out_path, index=False)
        return filepath, len(result), None

    except Exception as e:
        return filepath, 0, str(e)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    total = len(files)
    print(f"共 {total} 个日线文件，开始生成 {WINDOW} 日周线...")

    workers = min(16, cpu_count())
    done = 0
    errors = []

    with Pool(workers) as pool:
        for filepath, nrows, err in pool.imap_unordered(process_file, files, chunksize=32):
            done += 1
            if err:
                errors.append((filepath, err))
            if done % 500 == 0 or done == total:
                print(f"  进度: {done}/{total}  ({done*100//total}%)")

    print(f"\n完成! 共生成 {done - len(errors)} 个周线文件")
    if errors:
        print(f"失败 {len(errors)} 个:")
        for f, e in errors[:10]:
            print(f"  {f}: {e}")


if __name__ == "__main__":
    main()
