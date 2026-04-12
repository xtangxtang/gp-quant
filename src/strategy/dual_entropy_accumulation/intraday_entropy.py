"""
日内熵分析器 (Intraday Entropy Analyzer)

从单日 1 分钟交易数据中提取熵特征，回答：
- 今天的市场微观结构处于什么状态？
- 日内是否出现了有序→混沌的切换？
- 与日线级别的熵特征相比，日内视角提供了什么增量信息？

数据来源：gp-data/trade/{stock_code}/{YYYY-MM-DD}.csv
列：时间,开盘,收盘,最高,最低,成交量(手),成交额(元),均价,换手率(%)

算法复用 StockState 中经过论文验证的计算方法：
- 排列熵 (Bandt & Pompe 2002)
- 路径不可逆性 (Seifert 2025)
- 主导特征值 (Ma et al. 2026)
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ========== A 股分钟数据时间段 ==========
MORNING_START = "09:30"
MORNING_END = "11:30"
AFTERNOON_START = "13:00"
AFTERNOON_END = "15:00"


@dataclass
class IntradayEntropyResult:
    """单日日内熵分析结果"""

    stock_code: str
    trade_date: str

    # ---- 全天汇总指标 ----
    perm_entropy: float              # 全天排列熵（归一化）
    path_irreversibility: float      # 全天路径不可逆性
    dominant_eigenvalue: float       # 全天主导特征值
    turnover_entropy: float          # 成交量分布熵

    # ---- 半天对比 ----
    perm_entropy_am: float           # 上午排列熵
    perm_entropy_pm: float           # 下午排列熵
    entropy_shift: float             # 下午 - 上午（正=下午更无序）

    # ---- 日内演化 ----
    entropy_curve: List[float]       # 滚动窗口熵演化曲线
    entropy_trend: float             # 熵趋势斜率（正=日内趋向无序）
    entropy_volatility: float        # 熵波动（日内熵稳定性）

    # ---- 微观结构 ----
    active_bar_ratio: float          # 有成交的分钟占比
    volume_concentration: float      # 成交量集中度（Herfindahl）
    price_range_ratio: float         # 日内振幅 / 收盘价

    # ---- 状态诊断 ----
    intraday_state: str              # compressed / ordered / chaotic / transitioning

    n_bars: int = 0                  # 有效分钟数

    def to_dict(self) -> Dict:
        return {
            'stock_code': self.stock_code,
            'trade_date': self.trade_date,
            'perm_entropy': self.perm_entropy,
            'path_irreversibility': self.path_irreversibility,
            'dominant_eigenvalue': self.dominant_eigenvalue,
            'turnover_entropy': self.turnover_entropy,
            'perm_entropy_am': self.perm_entropy_am,
            'perm_entropy_pm': self.perm_entropy_pm,
            'entropy_shift': self.entropy_shift,
            'entropy_trend': self.entropy_trend,
            'entropy_volatility': self.entropy_volatility,
            'active_bar_ratio': self.active_bar_ratio,
            'volume_concentration': self.volume_concentration,
            'price_range_ratio': self.price_range_ratio,
            'intraday_state': self.intraday_state,
            'n_bars': self.n_bars,
        }


class IntradayEntropyAnalyzer:
    """
    从 1 分钟 K 线数据计算日内熵特征。

    典型用法::

        analyzer = IntradayEntropyAnalyzer()
        result = analyzer.analyze_day(
            "sh600000",
            "/path/to/gp-data/trade/sh600000/2026-04-07.csv",
        )
        print(result.perm_entropy, result.intraday_state)
    """

    def __init__(
        self,
        rolling_window: int = 60,
        perm_order: int = 3,
        irrev_threshold_sigma: float = 0.5,
        ar_order: int = 2,
    ):
        """
        参数
        ----
        rolling_window : int
            日内滚动熵的窗口大小（分钟），默认 60
        perm_order : int
            排列熵的嵌入维度
        irrev_threshold_sigma : float
            路径不可逆性的离散化阈值（sigma 倍数）
        ar_order : int
            主导特征值的 AR 阶数
        """
        self.rolling_window = rolling_window
        self.perm_order = perm_order
        self.irrev_threshold_sigma = irrev_threshold_sigma
        self.ar_order = ar_order

    # ================================================================
    #  数据加载
    # ================================================================

    def load_minute_csv(self, csv_path: str) -> pd.DataFrame:
        """加载单日分钟 CSV 文件。"""
        df = pd.read_csv(csv_path)

        col_map = {
            '时间': 'time',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量(手)': 'volume',
            '成交额(元)': 'amount',
            '均价': 'vwap',
            '换手率(%)': 'turnover_rate',
        }
        df = df.rename(columns=col_map)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)

        # 数值列强制转换
        for col in ['open', 'close', 'high', 'low', 'volume', 'amount', 'turnover_rate']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    # ================================================================
    #  核心特征计算（复用 StockState 的算法逻辑）
    # ================================================================

    def _permutation_entropy(self, values: np.ndarray, order: int = 3) -> float:
        """
        排列熵（归一化）。

        Bandt & Pompe (2002) 标准方法。
        """
        if len(values) < order + 2:
            return np.nan

        counts = {}
        for i in range(len(values) - order + 1):
            pattern = tuple(np.argsort(values[i:i + order], kind='mergesort'))
            counts[pattern] = counts.get(pattern, 0) + 1

        if not counts:
            return np.nan

        freq = np.array(list(counts.values()), dtype=np.float64)
        prob = freq / freq.sum()
        entropy = -np.sum(prob * np.log(prob))

        normalizer = np.log(math.factorial(order))
        if normalizer <= 0:
            return np.nan

        return float(entropy / normalizer)

    def _path_irreversibility(self, returns: np.ndarray) -> float:
        """
        路径不可逆性（KL 散度）。

        基于 Seifert (2025) 粗粒化轨迹方法。
        """
        if len(returns) < 15:
            return np.nan

        sigma = np.std(returns)
        if sigma < 1e-10:
            return 0.0

        threshold = self.irrev_threshold_sigma * sigma
        states = np.zeros(len(returns), dtype=np.int64)
        states[returns < -threshold] = -1
        states[returns > threshold] = 1

        n_states = 3
        counts = np.zeros((n_states, n_states), dtype=np.float64)
        for t in range(len(states) - 1):
            i, j = int(states[t] + 1), int(states[t + 1] + 1)
            if 0 <= i < n_states and 0 <= j < n_states:
                counts[i, j] += 1.0

        total = counts.sum()
        if total < 10:
            return np.nan

        forward = counts / total
        backward = counts.T / total

        mask = (forward > 1e-10) & (backward > 1e-10)
        if not np.any(mask):
            return 0.0

        kl_div = np.sum(forward[mask] * np.log(forward[mask] / backward[mask]))
        return max(0.0, float(kl_div))

    def _dominant_eigenvalue(self, returns: np.ndarray, order: int = 2) -> float:
        """
        主导特征值（Yule-Walker）。

        基于 Ma et al. (2026) 自相关提取特征值方法。
        """
        values = returns[np.isfinite(returns)]
        if len(values) < max(12, order + 6):
            return np.nan

        centered = values - np.mean(values)
        if np.std(centered) <= 1e-12:
            return np.nan

        acov = []
        for lag in range(order + 1):
            left = centered[:len(centered) - lag]
            right = centered[lag:]
            if len(right) == 0:
                return np.nan
            acov.append(np.dot(left, right) / len(right))

        system = np.array([
            [acov[abs(i - j)] for j in range(order)]
            for i in range(order)
        ])
        rhs = np.array(acov[1:order + 1])

        try:
            phi = np.linalg.solve(system + np.eye(order) * 1e-8, rhs)
        except np.linalg.LinAlgError:
            return np.nan

        companion = np.zeros((order, order))
        companion[0, :] = phi
        for i in range(1, order):
            companion[i, i - 1] = 1.0

        eigs = np.linalg.eigvals(companion)
        dominant = eigs[np.argmax(np.abs(eigs))]
        return float(np.real(dominant))

    def _turnover_entropy(self, volumes: np.ndarray) -> float:
        """
        成交量分布的 Shannon 熵。

        衡量成交量在时间上的分散/集中程度。
        均匀分布 → 高熵（散户随机交易）
        集中在少数分钟 → 低熵（主力集中操作）
        """
        volumes = volumes[volumes > 0]
        if len(volumes) < 5:
            return np.nan

        prob = volumes / volumes.sum()
        entropy = -np.sum(prob * np.log(prob))

        # 归一化到 [0, 1]
        max_entropy = np.log(len(prob))
        if max_entropy <= 0:
            return np.nan

        return float(entropy / max_entropy)

    def _volume_concentration(self, volumes: np.ndarray) -> float:
        """
        成交量 Herfindahl 集中度指数。

        接近 0 = 均匀分散
        接近 1 = 高度集中
        """
        total = volumes.sum()
        if total <= 0:
            return np.nan
        shares = volumes / total
        return float(np.sum(shares ** 2))

    # ================================================================
    #  日内滚动熵演化
    # ================================================================

    def _rolling_entropy_curve(self, returns: np.ndarray, window: int = 60) -> np.ndarray:
        """
        计算日内滚动排列熵曲线。

        返回长度为 len(returns) - window + 1 的数组。
        """
        if len(returns) < window:
            return np.array([])

        curve = []
        for i in range(len(returns) - window + 1):
            e = self._permutation_entropy(returns[i:i + window], self.perm_order)
            curve.append(e)

        return np.array(curve)

    # ================================================================
    #  分段计算（上午 / 下午）
    # ================================================================

    def _split_am_pm(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """将分钟数据拆分为上午和下午两段。"""
        times = df['time']
        hour = times.dt.hour

        am = df[(hour >= 9) & (hour < 12)].copy()  # 9:30 - 11:30
        pm = df[(hour >= 13) & (hour < 16)].copy()  # 13:00 - 15:00

        return am, pm

    # ================================================================
    #  状态诊断
    # ================================================================

    def _diagnose_state(
        self,
        perm_entropy: float,
        path_irrev: float,
        dominant_eig: float,
        entropy_trend: float,
    ) -> str:
        """
        根据日内熵特征判断微观结构状态。

        compressed : 低熵 + 低不可逆性 = 压缩/蓄力
        ordered    : 低熵 + 高方向性 = 有序趋势
        chaotic    : 高熵 + 高不可逆性 = 无序混沌
        transitioning : 中间态 / 正在切换
        """
        if np.isnan(perm_entropy) or np.isnan(path_irrev):
            return 'unknown'

        if perm_entropy < 0.65 and path_irrev < 0.05:
            return 'compressed'

        if perm_entropy < 0.75 and abs(dominant_eig) > 0.6:
            return 'ordered'

        if perm_entropy > 0.85 and path_irrev > 0.08:
            return 'chaotic'

        if abs(entropy_trend) > 0.001:
            return 'transitioning'

        return 'ordered' if perm_entropy < 0.80 else 'chaotic'

    # ================================================================
    #  主分析入口
    # ================================================================

    def analyze_day(
        self,
        stock_code: str,
        csv_path: str,
    ) -> Optional[IntradayEntropyResult]:
        """
        分析单日 1 分钟数据的熵特征。

        参数
        ----
        stock_code : str
            股票代码
        csv_path : str
            分钟级 CSV 文件路径

        返回
        ----
        IntradayEntropyResult 或 None（数据不足时）
        """
        if not os.path.exists(csv_path):
            return None

        df = self.load_minute_csv(csv_path)
        if len(df) < 30:
            return None

        trade_date = df['time'].iloc[0].strftime('%Y-%m-%d')

        # ---- 收益率序列 ----
        close = df['close'].values.astype(np.float64)
        close_valid = close[np.isfinite(close) & (close > 0)]
        if len(close_valid) < 30:
            return None

        returns = np.diff(np.log(close_valid))
        volumes = df['volume'].values.astype(np.float64)

        # ---- 全天指标 ----
        perm_ent = self._permutation_entropy(returns, self.perm_order)
        path_irrev = self._path_irreversibility(returns)
        dom_eig = self._dominant_eigenvalue(returns, self.ar_order)
        turn_ent = self._turnover_entropy(volumes)

        # ---- 上午 / 下午 ----
        am_df, pm_df = self._split_am_pm(df)

        def _segment_entropy(seg_df):
            c = seg_df['close'].values.astype(np.float64)
            c = c[np.isfinite(c) & (c > 0)]
            if len(c) < 15:
                return np.nan
            r = np.diff(np.log(c))
            return self._permutation_entropy(r, self.perm_order)

        perm_am = _segment_entropy(am_df)
        perm_pm = _segment_entropy(pm_df)
        entropy_shift = (perm_pm - perm_am) if np.isfinite(perm_pm) and np.isfinite(perm_am) else np.nan

        # ---- 日内熵演化曲线 ----
        curve = self._rolling_entropy_curve(returns, self.rolling_window)
        if len(curve) >= 5:
            finite_curve = curve[np.isfinite(curve)]
            if len(finite_curve) >= 5:
                x = np.arange(len(finite_curve))
                slope = np.polyfit(x, finite_curve, 1)[0]
                entropy_trend = float(slope)
                entropy_volatility = float(np.std(finite_curve))
            else:
                entropy_trend = 0.0
                entropy_volatility = 0.0
        else:
            entropy_trend = 0.0
            entropy_volatility = 0.0

        # ---- 微观结构指标 ----
        active_bars = np.sum(volumes > 0)
        active_ratio = float(active_bars / len(volumes)) if len(volumes) > 0 else 0.0

        vol_conc = self._volume_concentration(volumes[volumes > 0]) if np.any(volumes > 0) else np.nan

        high_all = df['high'].max()
        low_all = df['low'].min()
        close_last = df['close'].iloc[-1]
        price_range = (high_all - low_all) / close_last if close_last > 0 else 0.0

        # ---- 状态诊断 ----
        state = self._diagnose_state(perm_ent, path_irrev, dom_eig, entropy_trend)

        return IntradayEntropyResult(
            stock_code=stock_code,
            trade_date=trade_date,
            perm_entropy=perm_ent,
            path_irreversibility=path_irrev,
            dominant_eigenvalue=dom_eig,
            turnover_entropy=turn_ent,
            perm_entropy_am=perm_am,
            perm_entropy_pm=perm_pm,
            entropy_shift=entropy_shift,
            entropy_curve=curve.tolist() if len(curve) > 0 else [],
            entropy_trend=entropy_trend,
            entropy_volatility=entropy_volatility,
            active_bar_ratio=active_ratio,
            volume_concentration=vol_conc,
            price_range_ratio=float(price_range),
            intraday_state=state,
            n_bars=len(returns),
        )

    def analyze_stock_range(
        self,
        stock_code: str,
        trade_dir: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        批量分析一只股票的多日日内熵，返回 DataFrame。

        参数
        ----
        stock_code : str
            股票代码
        trade_dir : str
            该股票的分钟数据目录，如 gp-data/trade/sh600000/
        start_date, end_date : str, optional
            日期范围过滤（YYYY-MM-DD 格式）
        """
        if not os.path.isdir(trade_dir):
            return pd.DataFrame()

        csv_files = sorted(f for f in os.listdir(trade_dir) if f.endswith('.csv'))

        if start_date:
            csv_files = [f for f in csv_files if f.replace('.csv', '') >= start_date]
        if end_date:
            csv_files = [f for f in csv_files if f.replace('.csv', '') <= end_date]

        rows = []
        for csv_file in csv_files:
            csv_path = os.path.join(trade_dir, csv_file)
            result = self.analyze_day(stock_code, csv_path)
            if result is not None:
                rows.append(result.to_dict())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        return df
