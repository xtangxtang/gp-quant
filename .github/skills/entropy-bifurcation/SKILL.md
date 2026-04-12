---
name: entropy-bifurcation
description: "运行熵分岔 4 层选股系统。Use when: 熵分岔, entropy bifurcation, 四层系统, 4-layer, 路径不可逆, 主特征值, market coupling, 战略放弃, strategic abandonment, 高级选股, advanced scan。"
argument-hint: "扫描日期或股票代码，例如：20260310"
---

# 熵分岔 4 层选股系统 (Entropy Bifurcation Setup)

基于复杂性理论的 4 层决策系统，整合 12 篇经济物理论文的核心洞见。

## 系统架构

### 4 层决策结构

| 层级 | 功能 | 权重 |
|------|------|------|
| **Market Gate** | 市场环境判断（耦合熵、噪声、regime） | 20% |
| **Stock State** | 个股状态评估（熵、分岔、触发就绪） | 70% |
| **Execution Cost** | 执行成本（滑点、机会成本、控制负担） | 10% |
| **Experimental** | 实验模型（TDA、reservoir、PINN） | 0%（仅研究） |

### 核心增强因子

相比基础多时间框架策略，增加以下因子：

- **path_irreversibility_20** — 路径不可逆性：粗粒化轨迹不对称（来自随机热力学论文）
- **dominant_eig_20** — 主特征值：滚动自相关矩阵的最大特征值（分岔前兆优于 AR(1)）
- **phase_adjusted_ar1_20** — 相位调整 AR(1)：修正星期/月份/季度周期性
- **market_coupling_entropy_20** — 市场耦合熵：行业领先-滞后网络的熵（检测过度相关态）
- **strategic_abandonment** — 战略放弃：噪声/执行成本超阈值时跳过
- **entry_mode / exit_mode / position_scale** — 分阶段建仓/退出

## 使用场景

- 需要比基础共振扫描更精细的选股
- 评估市场耦合度（是否处于过度相关态）
- 检查个股的路径不可逆性（资金是否真在驱动趋势）
- 前瞻回测 4 层系统的历史表现

## 执行步骤

### 1. 运行扫描

```bash
cd /nvme5/xtang/gp-workspace/gp-quant

# 默认扫描
./scripts/run_entropy_bifurcation_setup.sh

# 指定日期
./scripts/run_entropy_bifurcation_setup.sh --scan-date 20260310

# 小样本测试
./scripts/run_entropy_bifurcation_setup.sh \
  --symbols sh600000,sz000001

# 前瞻回测
./scripts/run_entropy_bifurcation_setup.sh \
  --backtest-start-date 20260101 \
  --backtest-end-date 20260331 \
  --hold-days 5
```

参数与多时间框架一致（`--top-n`, `--min-amount`, `--min-turnover`, `--hold-days` 等）。

### 2. 查看结果

输出到 `results/entropy_bifurcation_setup/`。

### 3. 解读增强因子

- `path_irreversibility_20` 高 → 资金流有方向性，非随机波动
- `dominant_eig_20` 接近 -1 → 分岔前兆信号
- `market_coupling_entropy_20` 低 → 市场处于高度同步态，风险上升
- `strategic_abandonment = True` → 建议跳过该股

## 理论基础

来自 12 篇论文的关键洞见：
1. 市场不可逆性在资源流动活跃时增大 → path_irreversibility
2. 分岔前兆 = 主特征值趋近 -1 → dominant_eig
3. 局部观测只能测量熵下界 → 使用粗粒化下界代理
4. 高噪声 → 低频决策最优 → strategic_abandonment
5. 多指标动态整合优于单一指标 → 4 层融合
