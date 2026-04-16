"""
从交易明细/逐笔数据计算熵的核心模块

理论框架基于：
1. Seifert (2025): Universal bounds on entropy production from fluctuating coarse-grained trajectories
2. Bandt & Pompe (2002): Permutation Entropy 标准方法
3. GP-QUANT 12 篇论文的统一结论

核心思想：
- 市场是非平衡开放系统，熵产生衡量不可逆性
- 从粗粒化观测（K 线、逐笔）只能获得熵产生的下界
- 多时间尺度耦合携带不同的熵信息
"""

import math
from typing import Literal

import numpy as np
import pandas as pd


# =============================================================================
# 工具函数
# =============================================================================

def _safe_div(num: np.ndarray | pd.Series, den: np.ndarray | pd.Series) -> np.ndarray:
    """安全的除法，避免除零"""
    n = np.asarray(num, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    out = np.full(len(n), np.nan, dtype=np.float64)
    mask = np.isfinite(n) & np.isfinite(d) & (np.abs(d) > 1e-12)
    out[mask] = n[mask] / d[mask]
    return out


def _rolling_apply_1d(values: np.ndarray, window: int, func) -> np.ndarray:
    """在一维数组上滚动应用函数"""
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if window <= 0:
        return out
    for i in range(window - 1, len(arr)):
        out[i] = float(func(arr[max(0, i - window + 1): i + 1]))
    return out


def _discretize_trinary(values: np.ndarray, threshold_factor: float = 0.5) -> np.ndarray:
    """
    三态离散化：基于阈值将连续值映射到 {-1, 0, +1}

    用于粗粒化状态空间，符合 Seifert 2025 的粗粒化框架
    """
    arr = np.asarray(values, dtype=np.float64)
    sigma = np.nanstd(arr)
    if sigma <= 1e-12:
        return np.zeros(len(arr), dtype=np.int64)

    threshold = threshold_factor * sigma
    states = np.zeros(len(arr), dtype=np.int64)
    states[arr < -threshold] = -1
    states[arr > threshold] = 1
    return states


# =============================================================================
# 1. 路径不可逆性熵 (Path Irreversibility Entropy)
# 来源：Seifert (2025) - 粗粒化熵产生下界
# =============================================================================

def path_irreversibility_entropy(returns: np.ndarray, order_flow: np.ndarray = None) -> float:
    """
    计算路径不可逆性熵 - 熵产生的下界代理

    基于 Seifert 2025 论文中"从粗粒化轨迹估计熵产生下界"的方法。

    核心思想：
    - 将价格收益率和订单流离散化为三态
    - 计算状态转移的概率分布
    - 比较正向转移 P(i→j) 与反向转移 P(j→i) 的 KL 散度

    参数
    ----
    returns : np.ndarray
        价格收益率序列（或对数收益率）
    order_flow : np.ndarray, optional
        订单流序列（主动买 - 主动卖），如不提供则仅使用价格

    返回
    ----
    float
        路径不可逆性熵（KL 散度，非负）
        - 接近 0：系统接近可逆/平衡
        - 较大：系统强非平衡，存在单向驱动力
    """
    # 价格状态离散化
    ret_states = _discretize_trinary(returns, threshold_factor=0.5)

    # 如果有订单流，构造联合状态
    if order_flow is not None and len(order_flow) == len(returns):
        of_states = _discretize_trinary(order_flow, threshold_factor=0.5)
        # 联合状态：price_state * 3 + flow_state + 4 (映射到 0-8)
        joint_states = (ret_states + 1) * 3 + (of_states + 1)
    else:
        joint_states = ret_states + 1  # 映射到 0, 1, 2

    # 计算状态转移计数
    n_states = 9 if order_flow is not None else 3
    counts = np.zeros((n_states, n_states), dtype=np.float64)

    for t in range(len(joint_states) - 1):
        if np.isnan(joint_states[t]) or np.isnan(joint_states[t + 1]):
            continue
        i, j = int(joint_states[t]), int(joint_states[t + 1])
        if 0 <= i < n_states and 0 <= j < n_states:
            counts[i, j] += 1.0

    # 计算 KL 散度
    total = float(counts.sum())
    if total <= 1.0:
        return np.nan

    forward = counts / total
    backward = counts.T / total

    # 只在双向都有观测的地方计算
    mask = (forward > 1e-10) & (backward > 1e-10)
    if not np.any(mask):
        return 0.0

    kl_divergence = float(np.sum(forward[mask] * np.log(forward[mask] / backward[mask])))
    return max(0.0, kl_divergence)


def rolling_path_irreversibility(
    returns: pd.Series,
    order_flow: pd.Series = None,
    window: int = 60
) -> pd.Series:
    """滚动计算路径不可逆性熵"""
    ret_arr = returns.to_numpy(dtype=np.float64)
    of_arr = order_flow.to_numpy(dtype=np.float64) if order_flow is not None else None

    result = _rolling_apply_1d(
        np.arange(len(ret_arr)),
        window,
        lambda idx: path_irreversibility_entropy(
            ret_arr[int(idx):int(idx) + 1] if isinstance(idx, float) else ret_arr[max(0, int(idx[0] - window + 1)):int(idx[-1]) + 1],
            of_arr[max(0, int(idx[0] - window + 1)):int(idx[-1]) + 1] if of_arr is not None else None
        )
    )
    # 简化版本：直接使用 _rolling_apply_1d 的原始形式
    result = np.full(len(ret_arr), np.nan, dtype=np.float64)
    for i in range(window - 1, len(ret_arr)):
        window_ret = ret_arr[i - window + 1: i + 1]
        window_of = of_arr[i - window + 1: i + 1] if of_arr is not None else None
        result[i] = path_irreversibility_entropy(window_ret, window_of)

    return pd.Series(result, index=returns.index)


# =============================================================================
# 2. 等待时间分布熵 (Waiting Time Entropy)
# 来源：Seifert 2025 - 等待时间分布方法
# =============================================================================

def waiting_time_entropy(trade_times: pd.Series, window: int = 100) -> float:
    """
    从交易等待时间分布计算熵

    基于 Seifert 2025 论文中"等待时间分布方法"估计熵产生。

    核心思想：
    - 等待时间分布反映系统的动力学复杂度
    - 在平衡态，等待时间应接近指数分布（高熵）
    - 在有序态，等待时间分布集中（低熵）

    参数
    ----
    trade_times : pd.Series
        交易时间戳序列（datetime 类型）
    window : int
        滚动窗口大小

    返回
    ----
    float
        等待时间熵
    """
    times = pd.to_datetime(trade_times).dropna()
    if len(times) < 10:
        return np.nan

    # 计算等待时间（秒）
    wait_times = times.diff().dt.total_seconds().dropna()

    if len(wait_times) < 8:
        return np.nan

    # 对数分箱
    min_wait = max(wait_times.min(), 0.1)
    max_wait = wait_times.max()
    if max_wait <= min_wait:
        return np.nan

    n_bins = min(20, max(5, int(np.sqrt(len(wait_times)))))
    log_bins = np.logspace(np.log10(min_wait), np.log10(max_wait), n_bins + 1)

    hist, _ = np.histogram(wait_times, bins=log_bins)

    # 计算 Shannon 熵
    total = float(hist.sum())
    if total <= 0:
        return np.nan

    prob = hist / total
    prob = prob[prob > 0]

    entropy = float(-(prob * np.log(prob)).sum())

    # 归一化到 [0, 1]
    max_entropy = np.log(n_bins)
    if max_entropy > 0:
        entropy = entropy / max_entropy

    return entropy


def rolling_waiting_time_entropy(trade_times: pd.Series, window: int = 100) -> pd.Series:
    """滚动计算等待时间熵"""
    result = np.full(len(trade_times), np.nan, dtype=np.float64)

    for i in range(window - 1, len(trade_times)):
        window_times = trade_times.iloc[max(0, i - window + 1): i + 1]
        result[i] = waiting_time_entropy(window_times, window)

    return pd.Series(result, index=trade_times.index)


# =============================================================================
# 3. 多尺度换手率熵 (Multi-scale Turnover Entropy)
# 来源：12 篇论文统一结论 - 时变耦合网络熵
# =============================================================================

def turnover_rate_entropy(turnover_series: np.ndarray, n_bins: int = 10) -> float:
    """
    计算换手率序列的 Shannon 熵

    换手率分布反映市场参与度的离散程度：
    - 高熵：参与者分散，市场接近随机
    - 低熵：参与者集中，存在主导力量

    参数
    ----
    turnover_series : np.ndarray
        换手率时间序列
    n_bins : int
        分箱数量

    返回
    ----
    float
        Shannon 熵（归一化到 [0, 1]）
    """
    arr = np.asarray(turnover_series, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 5:
        return np.nan

    hist, _ = np.histogram(arr, bins=n_bins)

    total = float(hist.sum())
    if total <= 0:
        return np.nan

    prob = hist / total
    prob = prob[prob > 0]

    entropy = float(-(prob * np.log(prob)).sum())

    # 归一化
    max_entropy = np.log(n_bins)
    if max_entropy > 0:
        entropy = entropy / max_entropy

    return entropy


def multi_scale_turnover_entropy(
    df_tick: pd.DataFrame,
    scales: list[str] = None
) -> pd.DataFrame:
    """
    计算多时间尺度的换手率熵

    参数
    ----
    df_tick : pd.DataFrame
        交易明细数据，需包含：
        - trade_time: 交易时间戳
        - volume: 成交量
        - amount: 成交额
        - turnover_rate: 换手率（如有）
    scales : list[str]
        时间尺度列表，如 ['1min', '5min', '15min', 'daily']

    返回
    ----
    pd.DataFrame
        各时间尺度的换手率熵
    """
    if scales is None:
        scales = ['1min', '5min', '15min', '30min', '60min']

    if df_tick is None or df_tick.empty:
        return pd.DataFrame()

    df = df_tick.copy()
    df['trade_time'] = pd.to_datetime(df['trade_time'])
    df = df.sort_values('trade_time').reset_index(drop=True)

    # 计算各时间尺度的累积换手
    df.set_index('trade_time', inplace=True)

    results = {'trade_time': df.index}

    for scale in scales:
        # 解析时间尺度
        if scale.endswith('min'):
            freq = scale.replace('min', 'T')
        elif scale.endswith('hour'):
            freq = scale.replace('hour', 'H')
        elif scale == 'daily':
            freq = 'D'
        else:
            freq = scale

        # 重采样计算累积换手
        if 'turnover_rate' in df.columns:
            resampled = df['turnover_rate'].resample(freq).sum()
        elif 'volume' in df.columns and 'total_shares' in df.columns:
            resampled = df['volume'].resample(freq).sum() / df['total_shares'].iloc[0]
        else:
            continue

        # 滚动窗口计算熵
        if len(resampled) >= 20:
            rolling_entropy = resampled.rolling(window=20, min_periods=10).apply(
                lambda x: turnover_rate_entropy(x, n_bins=10), raw=True
            )
            results[f'turnover_entropy_{scale}'] = rolling_entropy.values

    return pd.DataFrame(results)


def rolling_turnover_entropy(
    turnover_series: pd.Series,
    window: int = 60,
    n_bins: int = 10
) -> pd.Series:
    """
    滚动计算换手率熵

    参数
    ----
    turnover_series : pd.Series
        换手率时间序列
    window : int
        滚动窗口大小
    n_bins : int
        分箱数量

    返回
    ----
    pd.Series
        滚动换手率熵
    """
    arr = turnover_series.to_numpy(dtype=np.float64)
    result = np.full(len(arr), np.nan, dtype=np.float64)

    for i in range(window - 1, len(arr)):
        window_vals = arr[i - window + 1: i + 1]
        result[i] = turnover_rate_entropy(window_vals, n_bins=n_bins)

    return pd.Series(result, index=turnover_series.index)


# =============================================================================
# 4. 排列熵 (Permutation Entropy) - 用于高频数据
# 来源：标准方法，12 篇论文中作为诊断工具引用
# =============================================================================

def permutation_entropy(window_values: np.ndarray, order: int = 3) -> float:
    """
    计算排列熵（Bandt & Pompe, 2002）

    优点：
    - 对噪声鲁棒
    - 不需要分箱
    - 适合高频数据

    参数
    ----
    window_values : np.ndarray
        窗口内数据
    order : int
        排列阶数（默认 3）

    返回
    ----
    float
        归一化排列熵 [0, 1]
    """
    values = np.asarray(window_values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) < order + 2:
        return np.nan

    # 计算序列表
    counts: dict[tuple[int, ...], int] = {}
    for idx in range(len(values) - order + 1):
        pattern = tuple(np.argsort(values[idx: idx + order], kind='mergesort'))
        counts[pattern] = counts.get(pattern, 0) + 1

    if not counts:
        return np.nan

    # 计算 Shannon 熵
    freq = np.asarray(list(counts.values()), dtype=np.float64)
    prob = freq / float(freq.sum())
    entropy = float(-(prob * np.log(prob)).sum())

    # 归一化
    normalizer = float(np.log(math.factorial(order)))
    if normalizer <= 0:
        return np.nan

    return entropy / normalizer


def rolling_permutation_entropy(
    series: pd.Series,
    window: int = 60,
    order: int = 3
) -> pd.Series:
    """
    滚动计算排列熵 — 滑动窗口直方图加速版

    对 order=3 使用比较编码 + 滑动直方图, 避免逐窗口 np.argsort.
    复杂度从 O(n * w * w) 降到 O(n).
    """
    arr = series.to_numpy(dtype=np.float64)
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window or window < order + 2:
        return pd.Series(result, index=series.index)

    if order != 3:
        # 非 order=3 退回原始逐窗口方式
        for i in range(window - 1, n):
            result[i] = permutation_entropy(arr[i - window + 1: i + 1], order=order)
        return pd.Series(result, index=series.index)

    # ── order=3 快速路径: 比较编码 + 滑动直方图 ──
    # 对每个连续三元组 (arr[j], arr[j+1], arr[j+2]) 编码为 0-7
    a = arr[:-2]   # arr[j]
    b = arr[1:-1]  # arr[j+1]
    c = arr[2:]    # arr[j+2]
    code = ((a > b).astype(np.int32) << 2) | \
           ((a > c).astype(np.int32) << 1) | \
           ((b > c).astype(np.int32))
    # 6 种有效排列映射到 {0,1,3,4,6,7}, 共 8 个桶

    n_codes = len(code)          # = n - 2
    n_tri = window - order + 1   # 每个窗口内的三元组数 = window - 2
    normalizer = math.log(math.factorial(order))  # log(6)

    if n_tri <= 0 or n_codes < n_tri:
        return pd.Series(result, index=series.index)

    # 初始窗口: codes[0 : n_tri]
    counts = np.zeros(8, dtype=np.int64)
    for j in range(n_tri):
        counts[code[j]] += 1

    def _entropy_from_counts(counts, total, normalizer):
        mask = counts > 0
        if not mask.any():
            return np.nan
        prob = counts[mask].astype(np.float64) / total
        return float(-(prob * np.log(prob)).sum()) / normalizer

    # 第一个有效位置
    result[window - 1] = _entropy_from_counts(counts, n_tri, normalizer)

    # 滑动窗口: 每步移除一个旧编码, 加入一个新编码
    for i in range(window, n):
        rem_idx = i - window      # 移除的编码索引
        add_idx = i - 2           # 新增的编码索引
        if rem_idx < n_codes:
            counts[code[rem_idx]] -= 1
        if add_idx < n_codes:
            counts[code[add_idx]] += 1
        result[i] = _entropy_from_counts(counts, n_tri, normalizer)

    return pd.Series(result, index=series.index)


# =============================================================================
# 5. 主导特征值 (Dominant Eigenvalue) - 临界减速预警
# 来源：Predicting the onset of period-doubling bifurcations (2026)
# =============================================================================

def dominant_eigenvalue_from_autocorr(
    window_values: np.ndarray,
    order: int = 2
) -> float:
    """
    从自相关函数提取主导特征值

    用于检测临界减速（critical slowing down），是早期预警信号的核心。

    参数
    ----
    window_values : np.ndarray
        窗口内数据
    order : int
        AR 模型阶数

    返回
    ----
    float
        主导特征值（实部）
        - 接近 1：临界减速，系统接近失稳
        - 接近 0：快速恢复
        - 负值：振荡倾向
    """
    values = np.asarray(window_values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) < max(12, order + 6):
        return np.nan

    # 中心化
    centered = values - float(np.mean(values))
    if float(np.std(centered)) <= 1e-12:
        return np.nan

    # 计算自协方差
    acov: list[float] = []
    for lag in range(order + 1):
        left = centered[:len(centered) - lag]
        right = centered[lag:]
        if len(right) == 0:
            return np.nan
        acov.append(float(np.dot(left, right)) / float(len(right)))

    # Yule-Walker 方程
    system = np.asarray(
        [[acov[abs(i - j)] for j in range(order)] for i in range(order)],
        dtype=np.float64
    )
    rhs = np.asarray(acov[1: order + 1], dtype=np.float64)

    try:
        phi = np.linalg.solve(system + np.eye(order, dtype=np.float64) * 1e-8, rhs)
    except np.linalg.LinAlgError:
        return np.nan

    # 构造伴随矩阵
    companion = np.zeros((order, order), dtype=np.float64)
    companion[0, :] = phi
    if order > 1:
        companion[1:, :-1] = np.eye(order - 1, dtype=np.float64)

    # 计算特征值
    eigvals = np.linalg.eigvals(companion)
    dominant = eigvals[np.argmax(np.abs(eigvals))]

    return float(np.real(dominant))


def rolling_dominant_eigenvalue(
    series: pd.Series,
    window: int = 60,
    order: int = 2
) -> pd.Series:
    """滚动计算主导特征值"""
    arr = series.to_numpy(dtype=np.float64)
    result = np.full(len(arr), np.nan, dtype=np.float64)

    for i in range(window - 1, len(arr)):
        window_vals = arr[i - window + 1: i + 1]
        result[i] = dominant_eigenvalue_from_autocorr(window_vals, order=order)

    return pd.Series(result, index=series.index)


# =============================================================================
# 6. 综合状态判别
# =============================================================================

def market_state_classifier(
    path_irrev: float,
    perm_entropy: float,
    dominant_eig: float = None,
    turnover_entropy: float = None
) -> Literal['ordered', 'weak_chaos', 'strong_chaos', 'critical']:
    """
    基于三个核心熵指标的市场状态分类

    参数
    ----
    path_irrev : float
        路径不可逆性熵
    perm_entropy : float
        排列熵
    dominant_eig : float, optional
        主导特征值（已弃用，保留用于向后兼容）
    turnover_entropy : float, optional
        换手率熵

    返回
    ----
    str
        市场状态：'ordered' | 'weak_chaos' | 'strong_chaos' | 'critical'
    """
    # 处理 NaN
    path_irrev = path_irrev if np.isfinite(path_irrev) else 0.5
    perm_entropy = perm_entropy if np.isfinite(perm_entropy) else 0.5
    turnover_entropy = turnover_entropy if (turnover_entropy is not None and np.isfinite(turnover_entropy)) else None

    # 高不可逆性 + 低排列熵 = 强有序（趋势/主力控盘）
    if path_irrev > 0.3 and perm_entropy < 0.4:
        return 'ordered'

    # 低不可逆性 + 高排列熵 = 强混沌（无序波动/散户博弈）
    if path_irrev < 0.1 and perm_entropy > 0.7:
        return 'strong_chaos'

    # 如果有换手率熵，用它进一步确认
    if turnover_entropy is not None:
        # 低换手熵 + 中等路径不可逆性 = 主力控盘
        if turnover_entropy < 0.5 and path_irrev > 0.15:
            return 'ordered'
        # 高换手熵 + 低路径不可逆性 = 散户博弈
        if turnover_entropy > 0.7 and path_irrev < 0.15:
            return 'strong_chaos'

    # 中间状态 = 弱混沌
    return 'weak_chaos'


# =============================================================================
# 7. 主接口：从交易明细构建熵特征框架
# =============================================================================

def build_tick_entropy_features(
    df_tick: pd.DataFrame,
    windows: dict = None
) -> pd.DataFrame:
    """
    从交易明细数据构建完整的熵特征框架

    只保留三个核心熵指标：
    1. 路径不可逆性熵 - 衡量主力控盘程度（单向驱动力）
    2. 排列熵 - 衡量有序/无序状态
    3. 换手率熵 - 衡量主力主导 vs 散户博弈

    参数
    ----
    df_tick : pd.DataFrame
        交易明细数据，需包含：
        - trade_time: 交易时间戳
        - price: 成交价
        - volume: 成交量
        - turnover_rate: 换手率（可选，用于计算换手率熵）
        - bs_flag: 买卖方向（可选，用于计算订单流）
    windows : dict
        各指标的滚动窗口配置
        默认：{'path_irrev': 60, 'perm_entropy': 60, 'turnover': 60}

    返回
    ----
    pd.DataFrame
        熵特征框架，包含：
        - trade_time: 交易时间戳
        - path_irreversibility: 路径不可逆性熵
        - permutation_entropy: 排列熵
        - turnover_entropy: 换手率熵（如有换手率数据）
        - market_state: 市场状态标签
    """
    if df_tick is None or df_tick.empty:
        return pd.DataFrame()

    if windows is None:
        windows = {
            'path_irrev': 60,
            'perm_entropy': 60,
            'turnover': 60
        }

    df = df_tick.copy()
    df['trade_time'] = pd.to_datetime(df['trade_time'])
    df = df.sort_values('trade_time').reset_index(drop=True)

    # 计算收益率
    df['log_ret'] = np.log(df['price']).diff()

    # 计算订单流（如有 bs_flag）
    if 'bs_flag' in df.columns:
        df['order_flow'] = df['bs_flag'].astype(float)
    else:
        # 简单启发：价格上涨视为买方主导
        df['order_flow'] = np.sign(df['log_ret'])

    # 准备换手率数据（如有）
    has_turnover = 'turnover_rate' in df.columns
    turnover_series = df['turnover_rate'] if has_turnover else None

    results = pd.DataFrame({'trade_time': df['trade_time']})

    # 1. 路径不可逆性熵（核心）
    if 'path_irrev' in windows:
        w = windows['path_irrev']
        results['path_irreversibility'] = rolling_path_irreversibility(
            df['log_ret'],
            df.get('order_flow'),
            window=w
        )

    # 2. 排列熵（核心）
    if 'perm_entropy' in windows:
        w = windows['perm_entropy']
        results['permutation_entropy'] = rolling_permutation_entropy(
            df['log_ret'],
            window=w,
            order=3
        )

    # 3. 换手率熵（核心）
    if 'turnover' in windows and has_turnover:
        w = windows['turnover']
        results['turnover_entropy'] = rolling_turnover_entropy(
            turnover_series,
            window=w
        )

    # 4. 市场状态分类
    if all(col in results.columns for col in ['path_irreversibility', 'permutation_entropy']):
        turnover_col = 'turnover_entropy' if 'turnover_entropy' in results.columns else None
        results['market_state'] = results.apply(
            lambda row: market_state_classifier(
                row['path_irreversibility'],
                row['permutation_entropy'],
                dominant_eig=np.nan,
                turnover_entropy=row.get('turnover_entropy', np.nan) if turnover_col else np.nan
            ),
            axis=1
        )

    return results
