"""
大盘趋势判断策略 - 数据加载模块

负责批量加载个股日线、指数、涨跌停、两融、SHIBOR、行业分类等数据。
"""

import glob
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MarketTrendConfig

# daily-full 使用的列
_DAILY_COLS = [
    "trade_date", "open", "high", "low", "close", "pct_chg",
    "vol", "amount", "turnover_rate",
    "buy_elg_amount", "buy_lg_amount", "sell_elg_amount", "sell_lg_amount",
    "net_mf_amount",
]


def load_all_stocks(cfg: MarketTrendConfig) -> Dict[str, pd.DataFrame]:
    """加载所有个股日线数据。

    返回 {symbol: DataFrame}，DataFrame 按 trade_date 升序排列。
    只保留 A 股 (sh6*, sz0*, sz3*) 且数据量 >= min_bars。
    """
    files = glob.glob(os.path.join(cfg.data_dir, "*.csv"))
    result: Dict[str, pd.DataFrame] = {}
    start_int = int(cfg.start_date) if cfg.start_date else 0
    end_int = int(cfg.end_date) if cfg.end_date else 99999999

    for fpath in files:
        fname = os.path.basename(fpath).replace(".csv", "")
        # 只保留 A 股
        if not (fname.startswith("sh6") or fname.startswith("sz0") or fname.startswith("sz3")):
            continue
        try:
            df = pd.read_csv(fpath, usecols=_DAILY_COLS)
            for col in _DAILY_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["trade_date", "close"]).copy()
            df["trade_date"] = df["trade_date"].astype(int)
            df = df.sort_values("trade_date").reset_index(drop=True)
            # 日期范围过滤：保留 start_date 前 max(ma_long, entropy_window) 天的数据
            buffer = max(cfg.ma_long, cfg.entropy_window, cfg.momentum_window) + 10
            if end_int < 99999999:
                df = df[df["trade_date"] <= end_int]
            if start_int > 0 and len(df) > buffer:
                # 找到 start_date 对应的位置，往前保留 buffer 行
                idx_start = df["trade_date"].searchsorted(start_int, side="left")
                keep_from = max(0, idx_start - buffer)
                df = df.iloc[keep_from:].reset_index(drop=True)
            if len(df) < cfg.min_bars:
                continue
            result[fname] = df
        except Exception:
            continue
    return result


def load_index(cfg: MarketTrendConfig) -> pd.DataFrame:
    """加载大盘指数日线。"""
    path = os.path.join(cfg.index_dir, f"{cfg.index_code}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["trade_date", "close", "open", "high", "low", "pct_chg", "vol", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["trade_date", "close"]).copy()
    df["trade_date"] = df["trade_date"].astype(int)
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def load_stk_limit(cfg: MarketTrendConfig) -> Dict[str, pd.DataFrame]:
    """加载涨跌停价格数据。

    返回 {symbol: DataFrame} 含 trade_date, up_limit, down_limit。
    """
    if not cfg.stk_limit_dir or not os.path.exists(cfg.stk_limit_dir):
        return {}
    files = glob.glob(os.path.join(cfg.stk_limit_dir, "*.csv"))
    result: Dict[str, pd.DataFrame] = {}
    for fpath in files:
        fname = os.path.basename(fpath).replace(".csv", "")
        if not (fname.startswith("sh6") or fname.startswith("sz0") or fname.startswith("sz3")):
            continue
        # stk_limit 文件名可能是 bj920000 或其他格式，需要映射
        # 格式: trade_date, ts_code, up_limit, down_limit
        try:
            df = pd.read_csv(fpath, usecols=["trade_date", "up_limit", "down_limit"])
            for col in ["trade_date", "up_limit", "down_limit"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna().copy()
            df["trade_date"] = df["trade_date"].astype(int)
            result[fname] = df
        except Exception:
            continue
    return result


def load_margin(cfg: MarketTrendConfig) -> pd.DataFrame:
    """加载两融余额数据 (全市场按交易所汇总)。

    返回按日聚合后的 DataFrame:
      trade_date, rzye(融资余额), rzmre(融资买入), rzche(融资偿还),
      rqye(融券余额), rzrqye(融资融券余额)
    """
    if not cfg.margin_path or not os.path.exists(cfg.margin_path):
        return pd.DataFrame()
    df = pd.read_csv(cfg.margin_path)
    for col in ["trade_date", "rzye", "rzmre", "rzche", "rqye", "rzrqye"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["trade_date"]).copy()
    df["trade_date"] = df["trade_date"].astype(int)
    # 按日汇总 (原始数据可能按交易所拆分)
    agg = df.groupby("trade_date").agg({
        "rzye": "sum",
        "rzmre": "sum",
        "rzche": "sum",
        "rqye": "sum",
        "rzrqye": "sum",
    }).reset_index()
    agg = agg.sort_values("trade_date").reset_index(drop=True)
    return agg


def load_shibor(cfg: MarketTrendConfig) -> pd.DataFrame:
    """加载 SHIBOR 利率数据。

    返回: date(int), on, 1w, 1m, ...
    """
    if not cfg.shibor_path or not os.path.exists(cfg.shibor_path):
        return pd.DataFrame()
    df = pd.read_csv(cfg.shibor_path)
    df.rename(columns={"date": "trade_date"}, inplace=True)
    df["trade_date"] = pd.to_numeric(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"]).copy()
    df["trade_date"] = df["trade_date"].astype(int)
    for col in ["on", "1w", "2w", "1m", "3m", "6m", "9m", "1y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def load_industry_map(cfg: MarketTrendConfig) -> Dict[str, str]:
    """加载行业分类映射: ts_code -> L1 行业名称。

    使用 index_member_all.csv 的 l1_name 字段。
    """
    if not cfg.index_member_path or not os.path.exists(cfg.index_member_path):
        return {}
    try:
        df = pd.read_csv(cfg.index_member_path, usecols=["ts_code", "l1_name"])
        df = df.dropna(subset=["ts_code", "l1_name"])
        # ts_code 格式: 600000.SH -> sh600000
        mapping: Dict[str, str] = {}
        for _, row in df.iterrows():
            ts = str(row["ts_code"])
            industry = str(row["l1_name"])
            # 转换 600000.SH -> sh600000
            parts = ts.split(".")
            if len(parts) == 2:
                code, exchange = parts
                prefix = exchange.lower()
                symbol = f"{prefix}{code}"
                mapping[symbol] = industry
        return mapping
    except Exception:
        return {}


def load_basic_names(cfg: MarketTrendConfig) -> Dict[str, str]:
    """加载股票名称映射 (用于 ST 过滤)。

    返回 {symbol: name}。
    """
    if not cfg.basic_path or not os.path.exists(cfg.basic_path):
        return {}
    try:
        df = pd.read_csv(cfg.basic_path, usecols=["ts_code", "name"])
        df = df.dropna(subset=["ts_code", "name"])
        result: Dict[str, str] = {}
        for _, row in df.iterrows():
            ts = str(row["ts_code"])
            parts = ts.split(".")
            if len(parts) == 2:
                code, exchange = parts
                symbol = f"{exchange.lower()}{code}"
                result[symbol] = str(row["name"])
        return result
    except Exception:
        return {}


def get_trading_dates(stocks: Dict[str, pd.DataFrame],
                      start_date: str, end_date: str) -> np.ndarray:
    """从已加载的股票数据中提取交易日序列。"""
    all_dates = set()
    start_int = int(start_date) if start_date else 0
    end_int = int(end_date) if end_date else 99999999
    for df in stocks.values():
        td = df["trade_date"].values
        mask = (td >= start_int) & (td <= end_int)
        all_dates.update(td[mask].tolist())
    return np.sort(np.array(list(all_dates), dtype=np.int64))
