"""
量子相干性模块 — 密度矩阵与退相干速率

理论框架：
- 借用量子力学的密度矩阵形式编码市场的"多空未决"状态
- 非对角元衰减速率度量市场从"叠加"到"方向确认"的收敛速度
- 与 tick_entropy.py 互补：tick_entropy 告诉你"系统在哪"，本模块告诉你"变化多快"

参考文献：
- Baumgratz, Cramer & Plenio (2014): Quantifying Coherence, PRL 113(14)
- Baaquie (2004): Quantum Finance, Cambridge University Press
- Busemeyer & Bruza (2012): Quantum Models of Cognition and Decision
"""

import numpy as np
import pandas as pd


# =============================================================================
# 1. 特征→状态向量映射
# =============================================================================

def _features_to_state_vector(
    perm_entropy: float,
    path_irrev: float,
    dom_eig: float,
) -> np.ndarray:
    """
    将三个核心特征映射为 4 维状态向量（概率幅）

    基态定义：
      |0⟩ = compressed   (低熵 + 低不可逆 + 低特征值)
      |1⟩ = transitioning (低熵 + 中等不可逆 + 高特征值)
      |2⟩ = trending      (中等熵 + 高不可逆)
      |3⟩ = chaotic       (高熵 + 低不可逆)

    使用 softmax 将亲和度分数转为概率幅。

    参数
    ----
    perm_entropy : float
        排列熵 [0, 1]
    path_irrev : float
        路径不可逆性 [0, +∞)，典型值 0~0.5
    dom_eig : float
        主导特征值 [-1, 1]

    返回
    ----
    np.ndarray
        4 维归一化状态向量（概率幅，sqrt(概率)）
    """
    pe = perm_entropy if np.isfinite(perm_entropy) else 0.5
    pi = path_irrev if np.isfinite(path_irrev) else 0.0
    de = dom_eig if np.isfinite(dom_eig) else 0.0

    # 对 path_irrev 做 min-cap 避免极端值
    pi = min(pi, 1.0)
    de_abs = min(abs(de), 1.0)

    # 各基态的亲和度评分（越高越倾向该状态）
    affinities = np.array([
        # compressed: 低熵 + 低不可逆 + 低特征值
        (1.0 - pe) * (1.0 - pi) * (1.0 - de_abs),
        # transitioning: 低熵 + 中等特征值走高（临界减速）
        (1.0 - pe) * de_abs,
        # trending: 中等熵 + 高不可逆（强方向性）
        pi * (0.5 + 0.5 * pe),
        # chaotic: 高熵 + 低不可逆
        pe * (1.0 - pi),
    ], dtype=np.float64)

    # softmax → 概率
    affinities = np.clip(affinities, 0.0, None)
    total = affinities.sum()
    if total < 1e-12:
        probs = np.ones(4, dtype=np.float64) / 4.0
    else:
        probs = affinities / total

    # 概率幅 = sqrt(概率)
    amplitudes = np.sqrt(probs)
    return amplitudes


# =============================================================================
# 2. 密度矩阵构造
# =============================================================================

def density_matrix_from_state_vectors(state_vectors: np.ndarray) -> np.ndarray:
    """
    从多个状态向量构造混合态密度矩阵

    ρ = (1/N) Σ |ψ_t⟩⟨ψ_t|

    参数
    ----
    state_vectors : np.ndarray
        形状 (N, d) 的状态向量数组，每行是一个 d 维状态向量

    返回
    ----
    np.ndarray
        d × d 密度矩阵
    """
    sv = np.asarray(state_vectors, dtype=np.float64)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)

    n, d = sv.shape
    if n == 0:
        return np.eye(d, dtype=np.float64) / d

    rho = np.zeros((d, d), dtype=np.float64)
    for t in range(n):
        psi = sv[t]
        rho += np.outer(psi, psi)
    rho /= n

    # 确保归一化 Tr(ρ) = 1
    trace = np.trace(rho)
    if trace > 1e-12:
        rho /= trace

    return rho


def density_matrix_from_features(
    perm_entropy_series: np.ndarray,
    path_irrev_series: np.ndarray,
    dom_eig_series: np.ndarray,
    window: int = 20,
) -> list[np.ndarray | None]:
    """
    从特征时间序列滚动构造密度矩阵

    参数
    ----
    perm_entropy_series : np.ndarray
        排列熵序列
    path_irrev_series : np.ndarray
        路径不可逆性序列
    dom_eig_series : np.ndarray
        主导特征值序列
    window : int
        滚动窗口

    返回
    ----
    list[np.ndarray | None]
        每个时间点的 4×4 密度矩阵，前 window-1 个为 None
    """
    pe = np.asarray(perm_entropy_series, dtype=np.float64)
    pi = np.asarray(path_irrev_series, dtype=np.float64)
    de = np.asarray(dom_eig_series, dtype=np.float64)

    n = len(pe)
    result: list[np.ndarray | None] = [None] * n

    # 预计算所有状态向量
    state_vecs = np.zeros((n, 4), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for t in range(n):
        if np.isfinite(pe[t]) and np.isfinite(pi[t]) and np.isfinite(de[t]):
            state_vecs[t] = _features_to_state_vector(pe[t], pi[t], de[t])
            valid[t] = True

    for i in range(window - 1, n):
        window_mask = valid[i - window + 1: i + 1]
        if window_mask.sum() < max(5, window // 4):
            continue
        window_vecs = state_vecs[i - window + 1: i + 1][window_mask]
        result[i] = density_matrix_from_state_vectors(window_vecs)

    return result


# =============================================================================
# 3. 相干性度量
# =============================================================================

def coherence_l1(rho: np.ndarray) -> float:
    """
    l1-范数相干性度量 (Baumgratz et al. 2014)

    C_{l1}(ρ) = Σ_{i≠j} |ρ_{ij}|

    归一化到 [0, 1]：除以 (d-1)

    参数
    ----
    rho : np.ndarray
        d × d 密度矩阵

    返回
    ----
    float
        归一化相干性 [0, 1]
        - 0: 完全退相干（对角密度矩阵），状态已确定
        - 1: 最大相干（最大叠加），方向完全未决
    """
    if rho is None:
        return np.nan
    d = rho.shape[0]
    off_diag_sum = np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))
    max_coherence = d - 1  # 上界
    if max_coherence <= 0:
        return 0.0
    return float(off_diag_sum / max_coherence)


def purity(rho: np.ndarray) -> float:
    """
    密度矩阵纯度

    γ = Tr(ρ²) ∈ [1/d, 1]

    参数
    ----
    rho : np.ndarray
        d × d 密度矩阵

    返回
    ----
    float
        纯度值
        - 1: 纯态（市场高度一致，单一状态主导）
        - 1/d: 最大混合态（完全不确定）
    """
    if rho is None:
        return np.nan
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(rho: np.ndarray) -> float:
    """
    von Neumann 熵

    S(ρ) = -Tr(ρ ln ρ) = -Σ_i λ_i ln(λ_i)

    归一化到 [0, 1]：除以 ln(d)

    参数
    ----
    rho : np.ndarray
        d × d 密度矩阵

    返回
    ----
    float
        归一化 von Neumann 熵 [0, 1]
        - 0: 纯态
        - 1: 最大混合态
    """
    if rho is None:
        return np.nan
    d = rho.shape[0]
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    if len(eigvals) == 0:
        return 0.0
    entropy = float(-np.sum(eigvals * np.log(eigvals)))
    max_entropy = np.log(d)
    if max_entropy > 0:
        entropy /= max_entropy
    return entropy


# =============================================================================
# 4. 退相干速率（核心新指标）
# =============================================================================

def coherence_decay_rate(
    coherence_series: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """
    计算相干性衰减速率

    ΔC/Δt ≈ (C(t) - C(t - window)) / window

    参数
    ----
    coherence_series : np.ndarray
        相干性时间序列
    window : int
        差分窗口

    返回
    ----
    np.ndarray
        衰减速率序列
        - 正值: 相干性在增加（不确定性增加/共识瓦解）
        - 负值: 退相干进行中（方向正在确认/共识形成）
        - |值| 大: 变化越快
    """
    c = np.asarray(coherence_series, dtype=np.float64)
    n = len(c)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        if np.isfinite(c[i]) and np.isfinite(c[i - window]):
            result[i] = (c[i] - c[i - window]) / window
    return result


# =============================================================================
# 5. 滚动计算 Pipeline（供 feature_engine 调用）
# =============================================================================

def compute_quantum_coherence_features(
    perm_entropy_series: pd.Series,
    path_irrev_series: pd.Series,
    dom_eig_series: pd.Series,
    rho_window: int = 20,
    decay_window: int = 5,
) -> pd.DataFrame:
    """
    一站式计算量子相干性特征，返回 DataFrame

    输出列:
      - coherence_l1:          l1-范数相干性 [0, 1]
      - purity:                密度矩阵纯度 [1/4, 1]
      - von_neumann_entropy:   von Neumann 熵 [0, 1]
      - coherence_decay_rate:  相干衰减速率
      - purity_norm:           纯度归一化到 [0, 1] (线性映射 [0.25, 1] → [0, 1])

    参数
    ----
    perm_entropy_series : pd.Series
        排列熵序列
    path_irrev_series : pd.Series
        路径不可逆性序列
    dom_eig_series : pd.Series
        主导特征值序列
    rho_window : int
        密度矩阵滚动窗口（默认 20）
    decay_window : int
        衰减速率差分窗口（默认 5）

    返回
    ----
    pd.DataFrame
        与输入序列等长的 DataFrame，前 rho_window-1 行为 NaN
    """
    pe = perm_entropy_series.to_numpy(dtype=np.float64)
    pi = path_irrev_series.to_numpy(dtype=np.float64)
    de = dom_eig_series.to_numpy(dtype=np.float64)
    n = len(pe)

    # 计算滚动密度矩阵
    rho_list = density_matrix_from_features(pe, pi, de, window=rho_window)

    # 提取指标
    coh = np.full(n, np.nan, dtype=np.float64)
    pur = np.full(n, np.nan, dtype=np.float64)
    vne = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        rho = rho_list[i]
        if rho is not None:
            coh[i] = coherence_l1(rho)
            pur[i] = purity(rho)
            vne[i] = von_neumann_entropy(rho)

    # 退相干速率
    cdr = coherence_decay_rate(coh, window=decay_window)

    # 纯度归一化 [1/4, 1] → [0, 1]
    pur_norm = np.clip((pur - 0.25) / 0.75, 0.0, 1.0)

    idx = perm_entropy_series.index
    return pd.DataFrame({
        'coherence_l1': coh,
        'purity': pur,
        'von_neumann_entropy': vne,
        'coherence_decay_rate': cdr,
        'purity_norm': pur_norm,
    }, index=idx)
