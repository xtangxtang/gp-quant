# Entropy Bifurcation Setup

这个目录按 `docs/complexity_research_12_papers_conclusion.md` 的研发优先级重建，核心目标不是堆更多概念，而是把策略明确拆成四层：

1. 市场门控层：`market_phase_state`、`market_noise_cost`、`market_coupling_entropy_20`
2. 个股状态层：`path_irreversibility_20`、`dominant_eig_20`、`phase_adjusted_ar1_20`
3. 执行成本层：`execution_cost_proxy_20`、`execution_cost_state`、`strategic_abandonment`
4. 实验模型层：`experimental_tda_score`、`experimental_reservoir_tipping_score`、`experimental_structure_latent_score`

当前主决策仍以“市场门控 + 个股状态 + 执行成本”为主，但实验层已经不再只是占位符：

1. `experimental_tda_score` 来自 delay embedding loop 的轻量拓扑代理
2. `experimental_reservoir_tipping_score` 来自固定随机 reservoir 的 tipping 敏感度代理
3. `experimental_structure_latent_score` 来自 compression / instability / launch / diffusion 的结构化 latent 状态模型

实验层默认仍不直接进入主决策权重，只作为研究观察层和排序辅助层。

## 权重落地

当前权重按照“市场门控、个股状态、执行成本、实验模型”四层设计显式实现：

1. 个股状态主评分：`entropy_quality=0.32`，`bifurcation_quality=0.48`，`trigger_quality=0.20`
2. 分叉层内部：`dominant_eig_abs_20=0.30`，`path_irreversibility_20=0.24`，`phase_adjusted_ar1_20=0.18`，`recovery_rate_20=0.12`，`var_lift_10_20=0.10`，`ar1_20=0.06`
3. 市场门控层：`low_entropy_share=0.22`，`bifurcation_share=0.26`，`energy_share=0.10`，`breakout_share=0.10`，`market_coupling_entropy_20=0.20`，`low_noise_support=0.12`
4. 最终上下文决策：`stock_state=0.70`，`market_gate=0.20`，`execution_readiness=0.10`，`experimental_model=0.00`

实验层权重明确为 `0.00`，表示结构保留但不进入主策略。

## 当前实现取舍

- 保留低熵压缩 + 分叉启动主线。
- 强化粗粒化不可逆性和主导特征值代理。
- 对 `AR(1)` 做多阶段周期去偏：weekday、月内分段、季度相位，并显式产出 `phase_distortion_20`。
- 把市场耦合从“行业正能量横截面熵”升级成“行业近 20 日 lead-lag returns 网络熵”代理。
- 把高噪声和高执行成本环境下的放弃交易，做成显式门控，而不是隐含在打分里。
- 把执行层从单次选股升级为 `position_scale + entry_mode + staged_entry_days + exit_mode` 的分段计划。
- 在前瞻回测里按实时持仓权重做收益聚合，而不是对活跃仓位简单等权平均。

## 主要字段

- `path_irreversibility_20`: 20 日粗粒化路径不可逆性代理
- `coarse_entropy_lb_20`: 不可逆性下界代理，当前等于 `path_irreversibility_20`
- `dominant_eig_20`: 局部自相关结构的主导特征值代理
- `dominant_eig_abs_20`: 主导特征值绝对值
- `phase_adjusted_ar1_20`: 基于 weekday + 月内分段 + 季度相位去偏的 AR 指标
- `phase_distortion_20`: 周期失真程度代理
- `market_phase_state`: `compression` / `transition` / `expansion` / `distorted` / `neutral` / `abandon`
- `market_coupling_entropy_20`: 行业近 20 日 lead-lag returns 网络的归一化 Shannon 熵
- `execution_cost_state`: `normal` / `cautious` / `blocked`
- `abandonment_score`: 战略放弃强度分数
- `strategic_abandonment`: 最终放弃交易的显式标记
- `position_scale`: 计划仓位权重
- `entry_mode`: `skip` / `probe` / `staged` / `full`
- `staged_entry_days`: 分段建仓天数
- `exit_mode`: `abandon` / `reduce` / `trail`

## 运行方式

```bash
python src/strategy/entropy_bifurcation_setup/run_entropy_bifurcation_scan.py \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv \
  --out_dir results/entropy_bifurcation_setup \
  --scan_date 20260330
```

## 还没有做的部分

- 真正的 persistent homology / mapper 级 TDA
- 可训练或自适应的 reservoir / state-space tipping 模型
- 真正的结构信息先验模型，而不是当前的轻量 latent heuristic

当前版本已经把这些方向做成可运行的轻量实验代理，但仍然保持在主决策权重之外。
