# Bull Hunter v3 — 大牛股预测系统

> 四 Agent LightGBM 流水线，预测 30%/100%/200% 目标涨幅

## 概述

Bull Hunter v3 是基于 LightGBM 的大牛股预测系统，采用 4-Agent 流水线架构。与规则驱动的[[entropy-accumulation-breakout]]不同，v3 采用数据驱动的二分类方法，同时预测三个独立目标（短线 30%、翻倍 100%、大牛 200%），并输出 A/B/C 分层推荐。

## 架构

```
Agent 1 (因子快照) → Agent 2 (训练分类器) → Agent 3 (预测评级) → Agent 4 (监控诊断)
```

### Agent 1: 增量因子生成
- **输入**: feature-cache, scan_date
- **输出**: 全市场因子截面 (~4900 只股票 × 32 因子)
- **逻辑**: 遍历 `feature-cache/daily/` 下所有 CSV，取 ≤ scan_date 的最新行；合并周线因子 (`w_` 前缀)；过滤 ST/退市/北交所/低流动性
- **耗时**: ~60s (IO 密集)

### Agent 2: 分类器训练
- **输入**: feature-cache, scan_date, TrainConfig
- **输出**: 3 个 LightGBM 模型 + meta.json
- **逻辑**:
  1. 回看 12 个月，每 5 天采样，构建 (stock × date) 训练 panel (~10 万~24 万行)
  2. 向量化计算前瞻涨幅：`gain_Nd = (close[t+N] - close[t]) / close[t]`
  3. 标记 label: `gain_Nd >= threshold` → 1
  4. 训练 LightGBM: 800 树, lr=0.03, scale_pos_weight ≤ 5.0, 无 early stopping
  5. F1 最优阈值搜索 [0.01, 0.50]
- **耗时**: ~40s (加载 30s + 训练 3×3s)

### Agent 3: 预测评级
- **输入**: 因子截面, 模型目录
- **输出**: predictions.csv (symbol, prob_30/100/200, grade A/B/C)
- **逻辑**: 全市场预测概率 → 阈值过滤 → A(prob_200 > th) > B(prob_100 > th) > C(prob_30 > th)
- **耗时**: ~30s

### Agent 4: 监控诊断
- **输入**: meta.json (训练指标)
- **输出**: health_report.json + 调参建议
- **逻辑**: 对照健康阈值 (AUC/precision/recall)，输出 SOP 诊断建议
- **耗时**: <1s

## 32 个因子

复用 [[entropy-accumulation-breakout]] 的特征引擎：
- **熵指标** (9): perm_entropy_{s,m,l}, entropy_slope/accel, path_irrev_{m,l}, dom_eig_{m,l}
- **换手率熵** (2): turnover_entropy_{m,l}
- **波动率** (8): volatility_{m,l}, vol_compression, bbw_pctl, vol_ratio_s, vol_impulse, vol_shrink, breakout_range
- **资金流** (9): mf_big_net, mf_big_net_ratio, mf_big_cumsum_{s,m,l}, mf_sm_proportion, mf_flow_imbalance, mf_big_momentum, mf_big_streak
- **密度矩阵** (4): coherence_l1, purity_norm, von_neumann_entropy, coherence_decay_rate

## 关键设计决策

| 决策 | 原因 |
|------|------|
| 不用 early stopping | 极端不平衡下 val loss 从第 1 棵树单调递增 |
| scale_pos_weight ≤ 5.0 | 原始 ~200:1，不截断第一棵树过拟合 |
| 时间分割验证 (非随机) | 避免未来信息泄露 |
| F1 搜索阈值 (非固定 0.5) | 正样本 < 1% 时 0.5 几乎全部预测为负 |

## 回测结果 (2025-03 ~ 2025-12)

11 次滚动扫描, 301 条预测:
- **A 级**: 10d +3.8% (57%胜率), 40d +6.0% (58%), 120d +32.7% (76%)
- **B 级**: 10d +2.1% (60%胜率), 40d +9.2% (63%), 120d +31.7% (74%)
- **C 级**: 10d +2.0% (52%胜率), 40d +7.7% (57%), 120d +21.2% (72%)

## 数据依赖

```
/gp-data/feature-cache/
├── daily/{symbol}.csv       ← 共享日线特征
├── weekly/{symbol}.csv      ← 共享周线特征
└── bull_models/{scan_date}/ ← 本策略模型
```

## 源码

`src/strategy/factor_model_selection/v3_bull_hunter/`

## 关联

- [[entropy-accumulation-breakout]] — 共享特征引擎和缓存
- [[tick-entropy-module]] — 核心熵计算
- [[four-layer-system]] — 四层系统的市场门过滤思路

---

*创建: 2026-04-19*
