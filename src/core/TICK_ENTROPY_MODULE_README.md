# tick_entropy.py 模块说明

## 模块概述

**文件名**: `src/core/tick_entropy.py`

**理论框架**: 基于 Seifert (2025) 的**粗粒化熵产生理论**（Coarse-grained Entropy Production）

**核心作用**: 从交易明细/逐笔成交数据中计算**熵特征**，用于识别市场状态（有序/混沌/临界）和预警临界转变（崩盘/突破前兆）。

---

## 为什么需要这个模块？

### 问题背景

传统的量化指标（如波动率、RSI、MACD）只能描述价格的**统计特性**，但无法回答：

1. 市场当前是**有序状态**还是**无序状态**？
2. 市场是否正在接近某个**临界转变点**（如崩盘前兆、突破前兆）？
3. 当前行情是**主力主导**还是**散户随机博弈**？

### 熵的答案

**熵（Entropy）** 是热力学和统计物理中描述系统"无序度"或"复杂度"的核心量：

| 熵水平 | 市场状态 | 交易含义 |
|-------|---------|---------|
| **低熵** | 有序状态 | 存在主导力量（主力控盘、强趋势） |
| **中熵** | 弱混沌 | 正常博弈状态（可交易） |
| **高熵** | 强混沌 | 无序波动（避免趋势策略） |
| **熵骤变** | 临界转变 | 预警信号（可能突破或崩盘） |

---

## 理论基础与论文来源

### Seifert (2025) 粗粒化熵产生理论

**论文**: *Universal bounds on entropy production from fluctuating coarse-grained trajectories*  
**作者**: Udo Seifert (Universität Stuttgart)  
**日期**: December 2025  
**链接**: arXiv:2512.07772

**核心结论**:
1. 对于**部分可观测系统**（如金融市场），无法精确计算真实熵产生
2. 但可以从粗粒化观测数据（K 线、逐笔成交）计算**熵产生下界**
3. 这个下界是**普适的**，不依赖系统微观细节

**数学表达**:
$$\sigma_{\text{true}} \geq \sigma_{\text{coarse}} \geq 0$$

其中 $\sigma_{\text{coarse}}$ 可以从观测数据估计，包括：
- 路径不可逆性（转移不对称性）
- 等待时间分布熵
- 粗粒化 Shannon 熵

**本模块采用的方法**:
| 方法 | 论文章节 | 代码函数 |
|-----|---------|---------|
| 路径不可逆性 | Sec. III.A: Waiting time distributions | `path_irreversibility_entropy()` |
| 等待时间熵 | Sec. III.B: Coarse-grained trajectories | `waiting_time_entropy()` |

---

### 12 篇复杂系统论文的统一结论

GP-QUANT 文档中 12 篇论文的核心共识：
1. 市场应被理解为**部分可观测、时变耦合、强噪声、存在不可逆性的复杂系统**
2. 熵、分叉、混沌等概念最有用的不是做比喻，而是帮助构造**状态变量**、**预警指标**和**门控机制**
3. 对 `gp-quant` 当前框架，最值得新增的是**粗粒化不可逆性**、**主导特征值代理**、周期去偏和 **strategic abandonment**

---

## 5 个核心熵指标的详细计算

### 概述

本模块计算**三个核心熵指标**：

| # | 指标 | 论文来源 | 代码函数 |
|---|------|---------|---------|
| 1 | **路径不可逆性熵** | Seifert (2025) | `path_irreversibility_entropy()` |
| 2 | **排列熵** | Bandt & Pompe (2002) | `permutation_entropy()` |
| 3 | **换手率熵** | GP-QUANT 12 篇论文统一结论 | `turnover_rate_entropy()` |

**补充说明**:
- **等待时间熵**也是熵，但从交易时间间隔计算，不在核心指标内
- **主导特征值**不是熵，是动力学特征值，已从核心输出中移除

---

### 1. 路径不可逆性熵 (Path Irreversibility Entropy)

**论论文来源**: 
- **Seifert (2025)**: *Universal bounds on entropy production from fluctuating coarse-grained trajectories*
- 论文章节: **Section III.A - Waiting time distributions**

**物理含义**: 衡量系统时间反演对称性的破缺程度，是熵产生的下界代理。

#### 详细计算步骤

**步骤 1: 计算收益率序列**
$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

**步骤 2: 三态离散化**
将收益率映射到三个状态：
$$
s_t = \begin{cases} 
-1 & \text{if } r_t < -\sigma/2 \quad (\text{跌破}) \\
0 & \text{if } |r_t| \leq \sigma/2 \quad (\text{持平}) \\
+1 & \text{if } r_t > \sigma/2 \quad (\text{突破})
\end{cases}
$$
其中 $\sigma$ 是收益率的标准差。

**步骤 3: 构造联合状态**（如有订单流）
如果同时考虑订单流 $f_t$（主动买 - 主动卖），则构造联合状态：
$$S_t = 3 \cdot (s_t + 1) + (f_t + 1) \in \{0, 1, \dots, 8\}$$

**步骤 4: 计算状态转移矩阵**
统计从状态 $i$ 转移到状态 $j$ 的计数 $C_{ij}$：
$$C_{ij} = \sum_{t=1}^{T-1} \mathbb{I}(S_t = i, S_{t+1} = j)$$

**步骤 5: 计算正向和反向转移概率**
$$P_{ij} = \frac{C_{ij}}{\sum_{i,j} C_{ij}}, \quad P^{\text{rev}}_{ij} = P_{ji} = \frac{C_{ji}}{\sum_{i,j} C_{ij}}$$

**步骤 6: 计算 KL 散度（路径不可逆性熵）**
$$\sigma_{\text{path}} = \sum_{i,j: P_{ij} > 0, P_{ji} > 0} P_{ij} \ln\left(\frac{P_{ij}}{P_{ji}}\right)$$

#### 代码实现（对应 tick_entropy.py 第 57-94 行）

```python
def path_irreversibility_entropy(returns: np.ndarray, order_flow: np.ndarray = None) -> float:
    # 步骤 2: 三态离散化
    ret_states = _discretize_trinary(returns, threshold_factor=0.5)
    
    # 步骤 3: 联合状态
    if order_flow is not None:
        of_states = _discretize_trinary(order_flow, threshold_factor=0.5)
        joint_states = (ret_states + 1) * 3 + (of_states + 1)  # 0-8
    else:
        joint_states = ret_states + 1  # 0-2
    
    # 步骤 4: 计算状态转移计数
    n_states = 9 if order_flow is not None else 3
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    for t in range(len(joint_states) - 1):
        i, j = int(joint_states[t]), int(joint_states[t + 1])
        counts[i, j] += 1.0
    
    # 步骤 5-6: KL 散度
    forward = counts / total
    backward = counts.T / total
    mask = (forward > 1e-10) & (backward > 1e-10)
    kl_divergence = np.sum(forward[mask] * np.log(forward[mask] / backward[mask]))
    
    return max(0.0, kl_divergence)
```

#### 交易解读

| 值范围 | 市场状态 | 解释 |
|-------|---------|------|
| 0 - 0.05 | 接近平衡 | 无明显方向，散户随机博弈 |
| 0.05 - 0.15 | 弱非平衡 | 轻微主导力量 |
| 0.15 - 0.30 | 中等非平衡 | 主力开始控盘 |
| > 0.30 | 强非平衡 | 主力高度控盘，趋势明显 |

#### 如何判断主力 vs 散户

**主力控盘特征**:
- 路径不可逆性 **持续高企** (> 0.2)
- 订单流与价格方向**高度一致**
- 大单集中在特定方向

**散户博弈特征**:
- 路径不可逆性 **接近 0** (< 0.05)
- 订单流**多空分散**
- 无明显方向性

---

### 2. 等待时间熵 (Waiting Time Entropy)

**论文来源**: 
- **Seifert (2025)**: *Universal bounds on entropy production from fluctuating coarse-grained trajectories*
- 论文章节: **Section III.B - Coarse-grained trajectories**

**物理含义**: 从交易时间间隔分布提取的复杂度指标，反映交易活动的聚集程度。

#### 详细计算步骤

**步骤 1: 计算等待时间序列**
$$\Delta t_i = t_{i+1} - t_i$$
其中 $t_i$ 是第 $i$ 笔交易的时间戳。

**步骤 2: 对数分箱**
由于等待时间跨度可能很大（秒到分钟），使用对数分箱：
$$\text{bins} = [\log_{10}(\Delta t_{\min}), \log_{10}(\Delta t_{\min} + \delta), \dots, \log_{10}(\Delta t_{\max})]$$

**步骤 3: 计算概率分布**
$$p_k = \frac{n_k}{N}$$
其中 $n_k$ 是落在第 $k$ 个 bin 中的等待时间数量。

**步骤 4: 计算 Shannon 熵**
$$S = -\sum_{k: p_k > 0} p_k \ln(p_k)$$

**步骤 5: 归一化**
$$S_{\text{norm}} = \frac{S}{\ln(n_{\text{bins}})} \in [0, 1]$$

#### 代码实现（对应 tick_entropy.py 第 100-140 行）

```python
def waiting_time_entropy(trade_times: pd.Series, window: int = 100) -> float:
    # 步骤 1: 计算等待时间（秒）
    times = pd.to_datetime(trade_times).dropna()
    wait_times = times.diff().dt.total_seconds().dropna()
    
    # 步骤 2: 对数分箱
    min_wait = max(wait_times.min(), 0.1)
    max_wait = wait_times.max()
    n_bins = min(20, max(5, int(np.sqrt(len(wait_times)))))
    log_bins = np.logspace(np.log10(min_wait), np.log10(max_wait), n_bins + 1)
    
    # 步骤 3-4: 计算 Shannon 熵
    hist, _ = np.histogram(wait_times, bins=log_bins)
    prob = hist / total
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log(prob))
    
    # 步骤 5: 归一化
    max_entropy = np.log(n_bins)
    return entropy / max_entropy
```

#### 交易解读

| 值范围 | 市场状态 | 解释 |
|-------|---------|------|
| > 0.85 | 高熵 | 交易时间分散，正常状态 |
| 0.70 - 0.85 | 中熵 | 轻微聚集 |
| 0.50 - 0.70 | 低熵 | 交易聚集，可能有主力行为 |
| < 0.50 | 极低熵 | 高度聚集，异常信号 |

---

### 3. 排列熵 (Permutation Entropy)

**论文来源**: 
- **Bandt & Pompe (2002)**: *Permutation entropy: A natural complexity measure for time series*
- 期刊: Physical Review Letters, 88(17), 174102
- **GP-QUANT 12 篇论文引用**: *Beyond the Largest Lyapunov Exponent* (2026) 使用粗粒化 Shannon 熵诊断混沌

**物理含义**: 基于序列的序列表复杂度，对噪声鲁棒。

#### 详细计算步骤

**步骤 1: 提取窗口数据**
取长度为 $W$ 的窗口：$\{x_1, x_2, \dots, x_W\}$

**步骤 2: 生成序列表**
对于阶数 $m$（通常取 3），将窗口分成 $W-m+1$ 个重叠子序列：
$$\vec{x}_t = (x_t, x_{t+1}, \dots, x_{t+m-1})$$

对每个子序列计算排序模式（pattern）：
$$\pi_t = \text{argsort}(\vec{x}_t) \in S_m$$
其中 $S_m$ 是 $m$ 个元素的排列集合（共 $m!$ 种）。

**步骤 3: 统计模式频率**
$$p_\pi = \frac{\text{count}(\pi)}{W - m + 1}$$

**步骤 4: 计算 Shannon 熵**
$$H = -\sum_{\pi: p_\pi > 0} p_\pi \ln(p_\pi)$$

**步骤 5: 归一化**
$$H_{\text{norm}} = \frac{H}{\ln(m!)} \in [0, 1]$$

#### 代码实现（对应 tick_entropy.py 第 215-253 行）

```python
def permutation_entropy(window_values: np.ndarray, order: int = 3) -> float:
    # 步骤 1: 提取有效数据
    values = np.asarray(window_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    
    # 步骤 2-3: 生成序列表并计数
    counts = {}
    for idx in range(len(values) - order + 1):
        pattern = tuple(np.argsort(values[idx: idx + order], kind='mergesort'))
        counts[pattern] = counts.get(pattern, 0) + 1
    
    # 步骤 4: 计算 Shannon 熵
    freq = np.asarray(list(counts.values()), dtype=np.float64)
    prob = freq / freq.sum()
    entropy = -np.sum(prob * np.log(prob))
    
    # 步骤 5: 归一化
    normalizer = np.log(math.factorial(order))
    return entropy / normalizer
```

#### 排列模式示例（m=3）

对于序列 `[1.2, 0.8, 1.5]`：
- 排序后索引：`(1, 0, 2)`（因为 0.8 < 1.2 < 1.5）
- 模式：`(1, 0, 2)`

所有可能的 3! = 6 种模式：
- `(0, 1, 2)` - 单调递增
- `(0, 2, 1)` - 先增后减
- `(1, 0, 2)` - 中间最小
- `(1, 2, 0)` - 单调递减
- `(2, 0, 1)` - 中间最大
- `(2, 1, 0)` - 先减后增

#### 交易解读

| 值范围 | 市场状态 | 解释 |
|-------|---------|------|
| > 0.90 | 高熵 | 高度无序，随机游走 |
| 0.70 - 0.90 | 中熵 | 弱混沌，正常交易 |
| 0.50 - 0.70 | 低熵 | 有序状态，趋势形成 |
| < 0.50 | 极低熵 | 高度有序，主力控盘 |

---

### 4. 主导特征值 (Dominant Eigenvalue) - 临界减速预警

**论文来源**: 
- **Ma, Z., et al. (2026)**: *Predicting the onset of period-doubling bifurcations via dominant eigenvalue extracted from autocorrelation*
- arXiv:2603.05523
- GP-QUANT docs/papers/period_doubling_dominant_eigenvalue.pdf

**物理含义**: 从自相关结构提取系统的主导特征值，用于检测**临界减速**（critical slowing down）。

#### 详细计算步骤

**步骤 1: 计算自协方差函数**
对于中心化序列 $\{x_1, x_2, \dots, x_N\}$（均值为 0）：
$$\gamma(k) = \frac{1}{N-k} \sum_{t=1}^{N-k} x_t x_{t+k}$$

**步骤 2: 构建 Yule-Walker 方程**
对于 AR(p) 模型，解方程：
$$\begin{bmatrix}
\gamma(0) & \gamma(1) & \cdots & \gamma(p-1) \\
\gamma(1) & \gamma(0) & \cdots & \gamma(p-2) \\
\vdots & \vdots & \ddots & \vdots \\
\gamma(p-1) & \gamma(p-2) & \cdots & \gamma(0)
\end{bmatrix}
\begin{bmatrix}
\phi_1 \\ \phi_2 \\ \vdots \\ \phi_p
\end{bmatrix}
=
\begin{bmatrix}
\gamma(1) \\ \gamma(2) \\ \vdots \\ \gamma(p)
\end{bmatrix}$$

**步骤 3: 构造伴随矩阵**
$$\mathbf{C} = \begin{bmatrix}
\phi_1 & \phi_2 & \cdots & \phi_{p-1} & \phi_p \\
1 & 0 & \cdots & 0 & 0 \\
0 & 1 & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & 1 & 0
\end{bmatrix}$$

**步骤 4: 计算特征值**
求解 $\det(\mathbf{C} - \lambda \mathbf{I}) = 0$，得到特征值 $\lambda_1, \lambda_2, \dots, \lambda_p$。

**步骤 5: 提取主导特征值**
$$\lambda_{\text{dom}} = \max_i |\lambda_i|$$

取实部作为输出：$\text{Re}(\lambda_{\text{dom}})$

#### 代码实现（对应 tick_entropy.py 第 282-338 行）

```python
def dominant_eigenvalue_from_autocorr(window_values: np.ndarray, order: int = 2) -> float:
    # 步骤 1: 中心化
    centered = values - np.mean(values)
    
    # 步骤 2: 计算自协方差
    acov = []
    for lag in range(order + 1):
        left = centered[:len(centered) - lag]
        right = centered[lag:]
        acov.append(np.dot(left, right) / len(right))
    
    # 步骤 3: 解 Yule-Walker 方程
    system = [[acov[abs(i-j)] for j in range(order)] for i in range(order)]
    rhs = acov[1:order + 1]
    phi = np.linalg.solve(system + np.eye(order) * 1e-8, rhs)
    
    # 步骤 4: 构造伴随矩阵
    companion = np.zeros((order, order))
    companion[0, :] = phi
    if order > 1:
        companion[1:, :-1] = np.eye(order - 1)
    
    # 步骤 5: 计算主导特征值
    eigvals = np.linalg.eigvals(companion)
    dominant = eigvals[np.argmax(np.abs(eigvals))]
    
    return float(np.real(dominant))
```

#### 临界减速预警原理

**动力系统背景**:
考虑一个简单的一阶线性系统：
$$\frac{dx}{dt} = -\alpha x + \xi(t)$$

其中 $\alpha$ 是恢复率，$\xi$ 是噪声。

当系统接近临界点时：
- $\alpha \to 0$（恢复率趋于 0）
- 自相关性 $\rho(\tau) \to 1$
- 主导特征值 $|\lambda| \to 1$

**预警信号**:
$$|\lambda_{\text{dom}}| > 0.9 \implies \text{临界减速预警}$$

#### 交易解读

| 值范围 | 市场状态 | 解释 |
|-------|---------|------|
| $|\lambda| > 0.9$ | **临界状态** | 系统接近失稳，突破/崩盘前兆 |
| $0.7 < |\lambda| < 0.9$ | 弱临界 | 轻微减速 |
| $|\lambda| < 0.7$ | 正常状态 | 快速恢复 |
| $\lambda < 0$ | 振荡倾向 | 均值回归倾向 |

**倍周期分岔特征**:
- 当 $\lambda \to -1$ 时，预警**倍周期分岔**（period-doubling bifurcation）
- 这对应市场的"震荡加剧 → 方向选择"过程

---

### 5. 换手率熵 (Turnover Rate Entropy)

**论文来源**: 
- **GP-QUANT 12 篇论文统一结论**: 时变耦合网络熵
- **Entropy Production Rate in Stochastically Time-evolving Asymmetric Networks** (2026)

**物理含义**: 换手率分布的离散程度，反映市场参与度的集中程度。

#### 详细计算步骤

**步骤 1: 准备换手率序列**
可以是：
- 日换手率
- 周换手率（周累积）
- 月换手率（月累积）

**步骤 2: 等距分箱**
$$\text{bins} = [\min(T), \min(T) + \Delta, \dots, \max(T)]$$
其中 $\Delta = \frac{\max(T) - \min(T)}{n_{\text{bins}}}$

**步骤 3: 计算概率分布**
$$p_k = \frac{n_k}{N}$$

**步骤 4: 计算 Shannon 熵**
$$S = -\sum_{k: p_k > 0} p_k \ln(p_k)$$

**步骤 5: 归一化**
$$S_{\text{norm}} = \frac{S}{\ln(n_{\text{bins}})}$$

#### 代码实现（对应 tick_entropy.py 第 346-380 行）

```python
def turnover_rate_entropy(turnover_series: np.ndarray, n_bins: int = 10) -> float:
    # 步骤 1: 提取有效数据
    arr = np.asarray(turnover_series, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    
    # 步骤 2: 等距分箱
    hist, _ = np.histogram(arr, bins=n_bins)
    
    # 步骤 3-4: 计算 Shannon 熵
    prob = hist / total
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log(prob))
    
    # 步骤 5: 归一化
    max_entropy = np.log(n_bins)
    return entropy / max_entropy
```

#### 如何判断主力 vs 散户博弈

**主力控盘特征** (低换手率熵 < 0.5):
- 换手率分布**集中**在某些特定水平
- 表明交易量被少数大资金主导
- 常见于：高度控盘股、庄股

**散户博弈特征** (高换手率熵 > 0.7):
- 换手率分布**分散**
- 表明交易量来自众多分散的参与者
- 常见于：散户主导的股票、指数 ETF

---

## 市场状态分类器的详细逻辑

### 分类算法

```python
def market_state_classifier(path_irrev, perm_entropy, dominant_eig, turnover_entropy):
    # 1. 临界状态检测（最高优先级）
    if abs(dominant_eig) > 0.9:
        return 'critical'  # 临界转变预警
    
    # 2. 有序状态检测
    if path_irrev > 0.3 and perm_entropy < 0.4:
        return 'ordered'  # 主力控盘，趋势明显
    
    # 3. 强混沌检测
    if path_irrev < 0.1 and perm_entropy > 0.7:
        return 'strong_chaos'  # 无序波动
    
    # 4. 默认：弱混沌
    return 'weak_chaos'  # 正常交易状态
```

### 状态特征表

| 状态 | 路径不可逆性 | 排列熵 | 主导特征值 | 换手率熵 | 交易策略 |
|-----|------------|-------|-----------|---------|---------|
| **ordered** | > 0.3 | < 0.4 | 任意 | < 0.5 | 趋势跟踪 |
| **weak_chaos** | 0.1-0.3 | 0.5-0.7 | < 0.9 | 0.5-0.7 | 正常交易 |
| **strong_chaos** | < 0.1 | > 0.7 | < 0.9 | > 0.7 | 避免趋势 |
| **critical** | 任意 | 任意 | > 0.9 | 任意 | **预警** |

---

## 临界转变预警的完整流程

### 理论背景

根据 **Ma et al. (2026)** 的临界减速理论：

当动力系统接近**分岔点**（bifurcation point）时：
1. 系统的恢复率下降 → 自相关性增强
2. 方差增加 → 波动率上升
3. 主导特征值 $|\lambda| \to 1$

### 预警步骤

**步骤 1**: 滚动计算主导特征值
$$\lambda_{\text{dom}}(t) = \text{dominant\_eigenvalue}(r_{t-W:t})$$

**步骤 2**: 检测阈值穿越
$$\text{warning}(t) = \mathbb{I}(|\lambda_{\text{dom}}(t)| > 0.9)$$

**步骤 3**: 多尺度确认（可选）
如果多个时间尺度同时预警，信号更可靠：
$$\text{strong\_warning}(t) = \bigwedge_{\text{scales}} \text{warning}_{\text{scale}}(t)$$

### 代码示例

```python
# 计算滚动主导特征值
df['dominant_eig_60'] = rolling_dominant_eigenvalue(df['log_ret'], window=60)

# 预警信号
df['critical_warning'] = np.abs(df['dominant_eig_60']) > 0.9

# 多尺度确认
df['strong_warning'] = (
    (np.abs(df['dominant_eig_20']) > 0.9) &
    (np.abs(df['dominant_eig_60']) > 0.9) &
    (np.abs(df['dominant_eig_120']) > 0.9)
)
```

---

## 参考论文清单

### 核心理论论文

| # | 论文 | arXiv/链接 | 本模块使用的方法 |
|---|------|-----------|----------------|
| 1 | Seifert, U. (2025). Universal bounds on entropy production from fluctuating coarse-grained trajectories | arXiv:2512.07772 | 路径不可逆性、等待时间熵 |
| 2 | Ma, Z., et al. (2026). Predicting the onset of period-doubling bifurcations via dominant eigenvalue extracted from autocorrelation | arXiv:2603.05523 | 主导特征值、临界减速预警 |
| 3 | Bandt, C., & Pompe, B. (2002). Permutation entropy: A natural complexity measure for time series | PRL 88(17) | 排列熵 |

### GP-QUANT 文档中的相关论文

| # | 论文 | 文件路径 | 相关性 |
|---|------|---------|-------|
| 1 | Universal bounds on entropy production | docs/papers/entropy_bounds_coarse_grained.pdf | 粗粒化熵下界 |
| 2 | Predicting period-doubling bifurcations | docs/papers/period_doubling_dominant_eigenvalue.pdf | 主导特征值 |
| 3 | Entropy Production Rate in Time-evolving Networks | docs/papers/entropy_time_evolving_networks.pdf | 网络熵 |

---

## 附录：完整计算流程图

```
交易明细数据
     │
     ▼
┌─────────────────────────────────────────────┐
│  数据预处理                                  │
│  - 计算收益率：r_t = log(P_t / P_{t-1})     │
│  - 计算订单流：f_t = bs_flag                │
│  - 计算等待时间：Δt = t_{i+1} - t_i         │
└─────────────────────────────────────────────┘
     │
     ├─────────────┬─────────────┬─────────────┐
     ▼             ▼             ▼             ▼
┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ 路径    │  │ 排列熵   │  │ 主导     │  │ 等待时间 │
│ 不可逆性│  │          │  │ 特征值   │  │ 熵       │
│         │  │          │  │          │  │          │
│ 三态    │  │ 序列表   │  │ Yule-    │  │ 对数分箱 │
│ 离散化  │  │ 统计    │  │ Walker   │  │ Shannon  │
│ 转移    │  │ Shannon  │  │ 特征值   │  │ 熵       │
│ 矩阵    │  │ 熵       │  │ 计算     │  │          │
│ KL 散度 │  │          │  │          │  │          │
└─────────┘  └──────────┘  └──────────┘  └──────────┘
     │             │             │             │
     └─────────────┴─────────────┴─────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   市场状态分类器      │
              │                       │
              │  - ordered (有序)     │
              │  - weak_chaos (弱混沌)│
              │  - strong_chaos (强混沌)│
              │  - critical (临界)    │
              └───────────────────────┘
```

---

**文档生成日期**: 2026-04-04

**模块版本**: v1.0
