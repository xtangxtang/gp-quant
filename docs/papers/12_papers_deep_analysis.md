# 12 篇 PDF 论文逐篇深度解析

**生成日期**: 2026-04-09
**基于**: 12 篇 PDF 原文全文阅读（非仅摘要）

---

## 第一主线：熵与非平衡热力学（3篇）

### 📄 1. Entropy Production Rate in Stochastically Time-evolving Asymmetric Networks

- **作者**: Tuan Pham (Amsterdam) & Deepak Gupta (TU Berlin)
- **日期**: 2026-03-29
- **arXiv**: 2603.27658v1

#### 核心方程

$N$ 个非线性单元通过时变非互易耦合 $J_{ij}(t)$ 相互作用，耦合遵循 Ornstein-Uhlenbeck 过程，关联时间 $\tau_0$ 可调节 annealed↔quenched：

$$\dot{x}_i(t) = -x_i(t) + F\left[\sum_{j \neq i} J_{ij}(t) x_j(t)\right] + \zeta_i(t)$$

其中 $J_{ij}(t) = \mu/N + (g/\sqrt{N}) Z_{ij}(t)$，$Z_{ij}$ 遵循 OU 过程。

#### 关键数学结果

- 用 DMFT 将高维系统约化为有效单体过程（方程3），解析推导出任意瞬态时刻的 EPR 表达式（方程5）：

$$\langle \dot{s}_{\text{res}}(t) \rangle^{(\text{MF})} = 1/T [C_x(t,t) + C_F(t,t) - 2C_{xF}(t,t)] - 1$$

- **稳态核心公式**：$\dot{s}_{\text{NESS}} = -\ddot{\bar{C}}_x(0^+)/T + 1$，即 EPR 完全由自相关函数在零时延附近的曲率决定
- 线性情形下 EPR 用 Bessel 函数精确表达（方程13）

#### 四个相

- 顺磁 (M=0, Q=0)
- 铁磁/持续活动 (M>0, Q>0)
- 异步混沌 (M=0, Q>0)
- 同步混沌

#### EPR 与方差的关系

- 临界流形以下：不同 $\tau_0$ 的 EPR-Variance 曲线坍缩为一条
- 临界流形以上：EPR-Variance 关系开始依赖 $\tau_0$，但 log-log 斜率相同
- annealed disorder 越强（$\tau_0$ 越小），EPR 越高

#### 对金融的关键启示

市场的不可逆性应从**时变行业/股票耦合结构**来理解，而非仅看单票序列。annealed disorder 越强（$\tau_0$ 越小），EPR 越高——类比市场相关性剧烈变化期间不可逆性增大。

---

### 📄 2. Universal Bounds on Entropy Production from Coarse-Grained Trajectories

- **作者**: Udo Seifert (Universität Stuttgart)
- **日期**: 2025-12-08
- **arXiv**: 2512.07772v1
- **篇幅**: 18页综述

#### 核心问题

当只能观测到粗粒化轨迹时，如何给真实熵产生建立下界？

#### 离散状态的 EPR

$$\sigma = \sum_{ij} p_i k_{ij} \ln(p_i k_{ij} / p_j k_{ji}) \geq 0$$

#### 连续变量的 EPR

$$\sigma = \langle \nu^2(x) \rangle / D = \int dx \, \nu^2(x) p(x) / D$$

其中 $\nu(x) = \langle \dot{x} | x \rangle$ 是平均局部速度。

#### 四类下界方法

1. **状态合并法**：$\sigma \geq \sigma_{\text{app}} = \sum_{IJ} \nu_{IJ} \ln(\nu_{IJ}/\nu_{JI})$，使用 log-sum 不等式
2. **TUR（热力学不确定性关系）**：$\sigma \geq 2\langle j\rangle^2 / (T \cdot \text{var}[j])$ — 任意流的均值²/方差给出 EPR 下界。已应用于分子马达效率上界、红血球膜、精子游泳等
3. **相关函数法**：自相关振荡的相干次数 $N$ 给出 $\Delta S_{\text{osc}} \geq 4\pi^2 N$（猜想，弱噪声极限已证明）
4. **等待时间分布法（ticks）**：Fano 因子 $F$ 越小，EPR 越大；tick 精度有阈值

#### 轨迹粗粒化的主关系

$$\sigma = \langle \Delta s_{\text{tot}}[\gamma] \rangle / T = (1/T) \sum_\gamma p[\gamma] \ln(p[\gamma]/p[\tilde{\gamma}])$$

粗粒化必须满足时间反演对易性条件 $\Gamma[\tilde{\gamma}] = \tilde{\Gamma}[\Gamma[\gamma]]$。

#### 对金融的关键启示

金融里能做的永远只是**不可逆性下界代理**。TUR 是最直接可落地的——只需某个"流"（如净资金流）的均值和方差即可估计。`coarse_entropy_lb_20`、`path_irreversibility_20` 这样的字段是合理的代理指标。

---

### 📄 3. Near-optimality of Conservative Driving in Discrete Systems

- **作者**: Jann van der Meer & Andreas Dechant (京都大学)
- **日期**: 2026-02-20
- **arXiv**: 2602.18321v1

#### 核心设定

离散马尔科夫跳跃过程，跃迁率参数化为 $k_{ij}(t) = \kappa_{ij}(t) e^{A_{ij}(t)/2}$，其中 $\kappa$ 对称（动力学/能垒），$A$ 反对称（驱动力）。

#### 熵产生率

$$\sigma = \sum_{i,j} \omega_{ij}(t) F_{ij}(t) \sinh(F_{ij}(t)/2)$$

其中热力学力 $F_{ij} = A_{ij} + \ln(p_i/p_j)$。

#### 核心定理

**保守力协议的熵产生最多是最优（非保守）协议的 2 倍：**

$$\sigma^* \leq \sigma^{\text{cons}} \leq 2\sigma^*$$

这通过引入辅助量 $\rho = \sum \omega_{ij} C(F_{ij})$ 证明，其中 $C(x) = x\sinh(x/2) - 2[\cosh(x/2)-1]$，满足 $\sigma/2 \leq \rho \leq \sigma$，且 $\rho$ 被保守力最小化。

#### 物理直觉

- 非保守力优化的条件（方程6）与保守力零环路亲和力条件（方程7）仅在 $O(F_{ij}^3)$ 阶不同
- 三态系统数值优化中，非保守改进仅约 1-2%（图1）
- 环形网络能垒穿越例子中，非保守力通过平衡跨能垒流和环路流来降低耗散

#### 对金融的关键启示

简单分段调仓（保守策略）最多比最优执行多耗 2 倍成本。支持"近似最优但工程友好"的执行设计。不必追求过度复杂的路径依赖最优控制。

---

## 第二主线：分叉、临界转变与早期预警（4篇）

### 📄 4. Topological Detection of Hopf Bifurcations via Persistent Homology

- **作者**: Jhonathan Barrios, Yásser Echávez, Carlos F. Álvarez
- **日期**: 2026-03-28
- **arXiv**: 2603.27395v1

#### 核心方法 (四步)

1. 从时间序列做 Takens 延迟嵌入 $\Phi(t) = [x(t), x(t-\tau), \ldots, x(t-(m-1)\tau)] \in \mathbb{R}^m$
2. 对点云 $X_\mu = \{\Phi_\mu(t_k)\}$ 构建 Vietoris-Rips 复形 $VR(X, \varepsilon)$
3. 计算持久同调，跟踪 $H_1$ 类（一维环/洞）的出生和死亡
4. 定义**拓扑泛函** $\mathcal{T}(\mu) = \max(\text{persistence of } H_1 \text{ classes})$

#### 分叉判据

$\mathcal{T}$ 突然增大 → Hopf 分叉发生（稳定平衡点 → 极限环）

- 分叉前：轨道收缩为点云，$H_1$ 中无持久特征
- 分叉后：轨道形成环形结构，$H_1$ 中出现显著持久类

#### 验证系统

- **Hopf 标准型**：已知临界值，精确检测
- **Lorenz 系统**：更复杂非线性动力学
- **Belousov-Zhabotinsky 反应简化模型**：额外通用性测试

#### 代码与数据

作者在 GitHub 上公开了代码、处理数据和生成图形的 notebooks。

#### 对金融的关键启示

不需要写出动力学方程。从价格/换手/资金流做延迟嵌入后，**1维环结构寿命**可检测从震荡→趋势的形态切换。更适合作为二级确认层，而非主扫描因子。

---

### 📄 5. Predicting Period-Doubling Bifurcations via Dominant Eigenvalue (DE-AC)

- **作者**: Zhiqin Ma, Chunhua Zeng, Ting Gao, Jinqiao Duan（昆明理工/华中科大/大湾大学）
- **日期**: 2026-02-22
- **arXiv**: 2603.05523v1

#### 核心推导

利用 Ornstein-Uhlenbeck 过程近似分叉前的随机动力学，解析推导 lag-$\tau$ 自相关函数：

$$\text{ACF}(\tau) = \lambda^\tau$$

其中 $\lambda$ 是系统 Jacobian 的主导特征值。

#### 倍周期分叉判据

$\lambda \to -1$（离散时间）时发生倍周期分叉。

- 远离分叉（$\lambda = -0.125$）：ACF 近乎无相关
- 接近分叉（$\lambda = -0.5, -0.75$）：ACF 出现**阻尼振荡**，恢复时间延长（临界减速）

#### 实验验证

- **模拟数据**：Fox 心脏交替模型、Ricker 映射、Hénon 映射
- **实验数据**：小鸡心脏聚合体的倍周期分叉（正常心跳→心律失常）
- DE-AC 在灵敏度和特异度上**均优于方差、lag-1 自相关和动力学特征值 (DEV)**

#### 与传统 EWS 的对比优势

| 指标 | 是否有明确阈值 | 是否能分类分叉类型 | 需要超参数 |
|------|:---:|:---:|:---:|
| 方差 | ❌ | ❌ | 少 |
| lag-1 ACF | ❌ | ❌ | 少 |
| DEV (S-map) | ✅ ($|\lambda|=1$) | ✅ | 多（嵌入维、时延、非线性参数） |
| **DE-AC** | **✅** ($\lambda \to -1$) | **✅** | **少（仅 lag 范围）** |

#### 对金融的关键启示

直接替代 `ar1_20` 的升级方案。从滚动自相关结构提取 dominant eigenvalue，提供明确阈值 $|\lambda|=1$ 判断"是否接近结构翻转"。主要服务于"系统是不是接近结构翻转"，而不是直接告诉你哪只股票明天涨。

---

### 📄 6. Ultra-Early Prediction of Tipping Points (RCDyM)

- **作者**: Xin Li, Qunxi Zhu, Chengli Zhao 等（复旦大学、国防科技大学）
- **日期**: 2026-03-16
- **arXiv**: 2603.14944v1

#### 三阶段框架

**阶段 1**: 用连续 Reservoir Computing 从滑窗数据学习局部动力学：

$$\dot{r}(t) = \gamma\{-r(t) + \tanh[Ar(t) + W_{\text{in}}s(t) + b_r]\}$$

输出通过 ridge regression 训练：$\hat{s}(t) = W_{\text{out}}r(t) + b_s$

训练后得到自治系统：$\dot{r}(t) = \gamma\{-r(t) + \tanh[\tilde{A}r(t) + \tilde{b}]\}$，其中 $\tilde{A} = A + W_{\text{in}}W_{\text{out}}$

**阶段 2**: 从训练好的 RC 自治系统中提取三类动力学度量：
- Jacobian 矩阵的**主特征值**
- 最大 **Floquet 乘子**
- 最大 **Lyapunov 指数**

**阶段 3**: 对这些动力学度量做**趋势外推**，实现超早期预测

#### 关键创新

- 不需要变化的系统参数作为输入，纯数据驱动
- 可处理平衡点、极限环、混沌三类吸引子上的分叉
- 比传统 EWS 预警提前 30-60 天

#### 对金融的关键启示

最适合做**市场级 regime engine**。回答"市场是否正在逼近某种失稳类型"，但不适合单独决定买卖哪一只资产。更合适的做法是把它当作高层门控器，服务于现有熵-分叉扫描器。

---

### 📄 7. Statistical Warning Indicators for Abrupt Transitions with Slow Periodic Forcing

- **作者**: Florian Suerhoff, Andreas Morr, Sebastian Bathiany, Niklas Boers, Christian Kuehn (TUM/PIK)
- **日期**: 2026-03-27
- **arXiv**: 2603.26537v1

#### 模型

周期强迫过阻尼 Duffing 振子：

$$dx = \left(x - \frac{1}{3}x^3 + D_a\cos(\omega t)\right)dt + \sigma dW_t$$

- $D_a > 0$：强迫振幅
- $\omega > 0$：强迫频率（慢强迫 $\omega \ll 1$）
- 弛豫振荡：系统在两个阱之间周期性跳跃

#### 核心发现

1. **Floquet 乘子在慢强迫下失效**：$|\mu| \leq \exp(-K/\omega)$，指数衰减到极小值，实际上无法检测到接近 1 的升高
2. **传统 CSD 指标被周期背景严重扭曲**
3. **两类替代指标**：
   - **(i) 跨周期指标**：在去趋势的周期内段上计算方差/自相关，跨周期跟踪趋势
   - **(ii) 相位依赖指标**：fast-jump 相对于强迫相位的时机（jump phase）

#### 关键结论

**相位依赖指标提供最强预警能力**，显著优于传统跨周期方差/自相关。

#### 快-慢分析

引入相位变量 $s = \omega t$ 得到快-慢形式：$\omega \frac{dx}{ds} = x - x^3/3 + D_a\cos s$。临界流形上 $D_a \geq 2/3$ 时存在折叠点，$D_a < 2/3$ 时折叠消失，只剩小振幅单阱响应。

#### 对金融的关键启示

如果不处理**季节性（财报周期、节假日、两会、春节）**，`ar1_20`、`var_lift_10_20` 等指标会被周期背景污染。必须做 **phase-aware 校正**。补入这篇论文后，对早预警的结论比原来更谨慎：早预警不是不能做，但必须做周期去偏和相位校正。

---

## 第三主线：混沌与稳定性诊断（2篇）

### 📄 8. Scrambling at the Genesis of Chaos

- **作者**: Thomas R. Michel, Mathias Steinhuber, Juan Diego Urbina, Peter Schlagheck (列日大学/雷根斯堡大学)
- **日期**: 2026-03-27
- **arXiv**: 2603.26480v1

#### 核心问题

如何区分**局部不稳定**（不稳定不动点附近的指数增长 $\sim e^{\lambda_s t}$）和**全局混沌**（Lyapunov 指数 $\sim e^{\lambda_L t}$）？

#### OTOC 与稳定性矩阵

经典 OTOC 定义：$C(t) = \text{Tr}[\rho |\hat{A}(t), \hat{B}(0)|^2]$

经典极限下退化为 Poisson 括号的平方：$C(t) \propto \hbar^2 \int dq\,dp\, |\{A(q,p,t), B(q,p)\}|^2 W(q,p)$

对于 $\hat{A} = \hat{q}_i, \hat{B} = \hat{p}_j$，OTOC $\propto (\partial q_i / \partial q_j)^2 \propto e^{2\lambda t}$

#### 关键结果

- 在从可积到混沌的过渡中，存在**有效指数** $\lambda_{\text{eff}} \in [\lambda_s, \lambda_L]$
- 分析方法：对周期调制摆做解析近似，得到分离线附近的轨道行为
- 遵循非线性共振驱动的可积性破缺范式

#### 验证系统

- **Kicked rotor**: 标准可积性破缺模型
- **Driven pendulum**: 普适性验证
- 数值证实 OTOC 和稳定性矩阵迹的有效指数等价（差因子2）

#### 对金融的关键启示

价格或波动突然出现指数式扩张，**并不自动意味着市场进入了全局混沌状态**。很多时候那只是局部不稳定（支撑位失守或一段短时放大）。"混沌判断"应是**谨慎的风险标签**而非方向信号。

---

### 📄 9. Beyond the Largest Lyapunov Exponent: Entropy-Based Diagnostics of Chaos

- **作者**: Alessandro A. Trani, Pierfrancesco Di Cintio, Michele Ginolfi
- **期刊**: Astronomy & Astrophysics
- **日期**: 2026-03-25
- **arXiv**: 2603.24675v1

#### 核心对比

Hénon-Heiles 势中最大 Lyapunov 指数 $\lambda_{\max}$ vs 粗粒化 Shannon 熵 $S = -\sum p_i \ln p_i$

#### 最大 Lyapunov 指数

$$\lambda_{\max} = \lim_{t\to\infty} \lim_{\|W_0\|\to 0} \frac{1}{t} \ln \frac{\|W(t)\|}{\|W_0\|}$$

其中 $W$ 是切空间中的偏差向量，满足 $\delta\dot{x} = \delta p$，$\delta\dot{p} = -D^2V(x)\delta x$。

#### 关键发现

- **Hénon-Heiles 系统**：Shannon 熵和 $\lambda_{\max}$ 的能量依赖性高度一致，熵跟随从弱混沌到广泛混沌的过渡
- **N 体系统中二者分裂**：
  - $\lambda_{\max}$ 几乎不随 $N$ 变化（与粒子数无关）
  - Shannon 熵随 $N$ 增大**单调递减**
- Shannon 熵更好地捕捉**全局相空间混合**（而非仅最不稳定方向）
- **Pesin 不等式**：$S_{KS} \leq \sum_{\lambda_i > 0} \lambda_i$

#### N 体问题中 Lyapunov 指数的困境

不同研究对 $\lambda_{\max}$ 与 $N$ 的关系得出矛盾结论（有的发现增加、有的发现 $\propto N^{-1/3}$），说明 Lyapunov 指数本身不足以表征高维引力系统的混沌。

#### 对金融的关键启示

应更多依赖**分布型、截面型复杂度**（市场宽度熵、行业同步度、网络熵），而非试图给单个日线序列估一个"市场 Lyapunov 指数"。单一 Lyapunov 风格指标对高维系统描述力不足。

---

## 第四主线：非线性系统与物理约束学习（3篇）

### 📄 10. Communication-Induced Bifurcation in Power Packet Networks

- **作者**: Takashi Hikihara (京都大学)
- **日期**: 2026-03-28
- **arXiv**: 2603.27446v1

#### 核心模型

路由器 Langevin 方程：

$$\frac{dx_t}{dt} = -\nabla H(x_t, \lambda_t) + \sqrt{2D}\xi(t)$$

指数信息处理成本：

$$\Phi(u, D) = \kappa \cdot D \cdot (\exp(\beta u) - 1)$$

系统评估函数：$J(u) = \alpha G(u) - \Phi(u, D) - T\Delta S$

#### Sagawa-Ueda 关系

$$\langle W \rangle \leq -\Delta F + kT\langle I \rangle$$

mutual information 提供的理论上限为通过信息提取的功的改进。

#### 关键结果

1. **一阶不连续相变**：噪声 $D > D_c$ 时，最优控制 $u^*$ 不连续跳变到零 → 系统**"战略性放弃"精细控制**
2. **信息壁垒**：存在根本性的信息障碍，超过阈值后获取信息的成本超过收益
3. **网络扩展**：扩散耦合和空间平滑将分叉点外推 $D_c \to D_c + \Delta D_c(\text{coupling})$，增强集体弹性

#### 对金融的关键启示

高噪声/高拥挤/高滑点环境中，**继续精细控制反而更差**。支持引入 **strategic abandonment**——当市场噪声、冲击成本、拥挤度或信息处理代价超过阈值时，系统应主动降频、降仓甚至暂停。

---

### 📄 11. PINN vs Neural ODE on Morris-Lecar Model

- **作者**: Nikolaos M. Matzakos & Chrisovalantis Sfyrakis (ASPETE Athens)
- **日期**: 2026-03-27
- **arXiv**: 2603.26921v1

#### Morris-Lecar 模型

$$C\frac{dV}{dt} = -g_{Ca}M_\infty(V)(V-V_{Ca}) - g_KN(V-V_K) - g_L(V-V_L) + I$$

$$\frac{dN}{dt} = \phi\frac{N_\infty(V) - N}{\tau_N(V)}$$

快（$V$）-慢（$N$）系统，$\phi \ll 1$。

#### 两种方法对比

| 特性 | PINN | Neural ODE |
|------|------|------------|
| 核心思想 | 将微分方程嵌入损失函数 | 直接从数据学习向量场 |
| 物理一致性 | 强制满足方程 | 无保证 |
| 数据效率 | 高（物理约束补偿数据不足） | 较低 |
| 灵活性 | 较低（受方程限制） | 高（黑箱） |
| 可解释性 | 高 | 低 |

#### 三种分叉 regime 测试

- **Hopf 分叉**：平衡点失稳，产生极限环
- **鞍节点极限环 (SNLC)**：极限环通过鞍节点消失
- **同宿轨道**：极限环与鞍点碰撞

#### 核心结论

- PINN 在**刚性和敏感分叉**下更准更稳（嵌入物理方程约束）
- NODE 更灵活但是黑盒，在**尖锐切换附近更脆弱**
- 高级 NODE 变体（ANODE、latent NODE）在刚性动力学下的表现仍是开放问题

#### 对金融的关键启示

gray-box 方向更稳。只要对市场有部分结构先验（状态连续性、边界约束、切换稀疏性、持久性约束），就应把这些约束嵌入状态模型，而不是完全交给黑盒。

---

### 📄 12. Comparative Investigation of Thermodynamic Structure-Informed Neural Networks

- **作者**: Guojie Li & Liu Hong (中山大学)
- **日期**: 2026-03-26
- **arXiv**: 2603.26803v1
- **篇幅**: 30页

#### 系统对比五种结构嵌入

**保守系统**：
- **Newtonian**：力平衡 $m\ddot{x} = F$
- **Lagrangian**：变分原理 $\delta\int L\,dt = 0$，$L = T - V$
- **Hamiltonian**：辛结构 $\dot{q} = \partial H/\partial p$，$\dot{p} = -\partial H/\partial q$

**耗散系统**：
- **Onsager 变分原理**：$\min\{\frac{1}{2}\langle\dot{x}, M^{-1}\dot{x}\rangle + \Psi(\dot{x}) + \Phi(x)\}$
- **扩展不可逆热力学 (EIT)**：含热流和耗散流的守恒律

#### 测试案例

- 保守：弹簧振子、单摆、双摆
- 耗散：阻尼摆、扩散方程、Fisher-Kolmogorov 方程

#### 核心发现

1. Newton 残差 PINN 能重建状态但**无法可靠恢复物理量**（如能量、动量）
2. 结构保持形式显著提高**参数识别精度、热力学一致性、鲁棒性**
3. 更强的结构约束不总是提高**所有**指标，但通常带来更平坦的 loss minima
4. Loss landscape 分析显示结构约束减少了浅局部最小值

#### 对金融的关键启示

不同 market regime 对模型约束形式的需求不同：
- **趋势状态** → 低耗散、强约束结构（Hamiltonian-like）
- **高噪声/高换手** → 耗散系统，适合加入 entropy balance、流量约束或 regime persistence 约束

不是去写一个"金融 Hamiltonian"，而是承认结构化状态估计优于纯"信号打分"。

---

## 交叉洞察汇总

### 稳健共识（8条）

| # | 共识 | 支撑论文 |
|---|------|---------|
| 1 | 市场里的"熵"最适合被理解为不可逆性代理，而非严格热力学量 | 1, 2 |
| 2 | 真正有价值的不是单变量熵，而是时变耦合结构、截面分布和路径不对称 | 1, 9 |
| 3 | AR(1) 和方差只够做最初级的临界减速检测，不能视为完整预警体系 | 5, 7 |
| 4 | 任何 early warning signal 都必须考虑周期背景和相位偏移 | 7 |
| 5 | 拓扑方法和 reservoir computing 更适合作为实验层和门控层 | 4, 6 |
| 6 | 单一 Lyapunov 风格指标不足以描述高维复杂系统 | 8, 9 |
| 7 | 控制和信息处理本身有成本，高噪声环境下"少做"往往更优 | 3, 10 |
| 8 | gray-box 和结构约束学习通常比纯黑盒时序拟合更靠谱 | 11, 12 |

### 不能直接照搬的地方（6条）

| # | 限制 | 相关论文 |
|---|------|---------|
| 1 | 物理论文中的 EPR 不等于金融里可精确估计的量 | 1, 2 |
| 2 | Hopf、period-doubling、fold 分叉在市场里只能作为近似类比 | 4, 5 |
| 3 | Hamiltonian/Lagrangian/Onsager 结构在金融里没有天然给定形式 | 12 |
| 4 | 很多论文依赖慢参数漂移和局部平稳，而市场存在突发跳变和反身性 | 5, 6, 7 |
| 5 | 复杂最优控制结果未必在真实交易约束下稳定兑现 | 3 |
| 6 | fully connected、大 $N$、白噪声/OU 假设与真实市场有距离 | 1 |

### 对 gp-quant 的优先级排序

| 优先级 | 行动项 | 基于论文 |
|--------|--------|---------|
| 1 | 新增 `path_irreversibility_20`（粗粒化路径不可逆性下界） | 2 |
| 2 | 新增 `dominant_eig_20`（DE-AC 方法），逐步弱化 `ar1_20` | 5 |
| 3 | 新增 `market_phase_state` + 周期去偏，处理 CSD 相位失真 | 7 |
| 4 | 新增 `market_coupling_entropy_20`（行业时变耦合不可逆性） | 1 |
| 5 | 实验层：TDA (persistent homology)、RC tipping score、structure-informed latent model | 4, 6, 11, 12 |
