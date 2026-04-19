# 个股因子有效性画像 (Per-Stock Factor Profiling)

> 基于特征缓存数据，分析每只股票各因子对未来涨跌的预测能力，发现个股规律。

---

## 1. 动机

策略当前使用全局固定阈值（如 `perm_entropy_m < 0.65`）。但不同股票的因子特征差异很大：

- 有些股票可能 `mf_flow_imbalance` 是强预测因子，有些则完全无效
- 资金流特征对大盘股可能比小盘股更有效
- 不同时期同一因子的有效性也会变化

本分析的目标：**从数据中挖掘每只股票「真正有效」的因子，以及它们的方向和强度**。

## 2. 方法

### 2.1 目标变量

从 `close` 计算前瞻 N 日收益率：

$$r_{t,N} = \frac{\text{close}_{t+N}}{\text{close}_t} - 1$$

默认使用三个窗口：1 日、3 日、5 日。

### 2.2 时间衰减加权

越靠近现在的数据权重越大（指数衰减）：

$$w_t = e^{-\lambda (T - t)}$$

其中 $T$ 是最后交易日序号，$t$ 是当前交易日序号，$\lambda$ 控制衰减速度。

| $\lambda$ | 半衰期 (交易日) | 含义 |
|-----------|----------------|------|
| 0.0035 | ~200 天 | 温和衰减，看 1 年关系 |
| 0.007 | ~100 天 | 适中，看半年关系 |
| 0.014 | ~50 天 | 激进，看近 2 个月 |

默认使用 $\lambda = 0.007$（~100 天半衰期）。

### 2.3 加权 Rank IC

对每个因子 $f$ 和前瞻收益 $r$：

1. 去除 NaN 行
2. 计算因子和收益的秩 (rank)
3. 用时间权重 $w_t$ 计算加权 Spearman 相关系数

$$\text{IC}_f = \text{WeightedCorr}\left(\text{rank}(f_t),\ \text{rank}(r_{t,N}),\ w_t\right)$$

- IC > 0：因子越大 → 越倾向上涨
- IC < 0：因子越大 → 越倾向下跌
- |IC| > 0.05 视为有实际意义

### 2.4 IC 稳定性 (IC_IR)

为检验 IC 是否稳定，而非偶然：

1. 按 60 天滚动窗口计算分段 IC
2. IC_IR = mean(IC) / std(IC)
3. |IC_IR| > 0.5 视为稳定

### 2.5 非线性检测

部分因子可能存在非线性关系（如 U 形）。用分箱分析补充：

1. 将因子值分为 5 档 (quintile)
2. 计算每档的平均前瞻收益
3. 检查是否存在单调或 U/倒 U 形模式

### 2.6 近期有效性（最近 60 天 IC）

单独计算最近 60 个交易日的 IC，与全局 IC 对比：

- 近期 IC 与全局 IC 同向 → 因子一致有效
- 近期 IC 反转 → 因子效果在变化，需谨慎

## 3. 分析因子

从特征缓存中提取以下因子（排除原始 OHLCV 和中间变量）：

### 3.1 核心策略因子 (17 个)

| 因子 | 说明 |
|------|------|
| `perm_entropy_s/m/l` | 短/中/长窗口置换熵 |
| `entropy_slope` | 多尺度熵差异 |
| `entropy_accel` | 熵变化速率 |
| `path_irrev_m/l` | 路径不可逆性 |
| `dom_eig_m/l` | 主特征值 |
| `turnover_entropy_m/l` | 换手率熵 |
| `volatility_m/l` | 波动率 |
| `vol_compression` | 波动率压缩 |
| `bbw_pctl` | 布林带宽分位数 |

### 3.2 量价因子 (4 个)

| 因子 | 说明 |
|------|------|
| `vol_ratio_s` | 短期量比 |
| `vol_impulse` | 量能脉冲 |
| `vol_shrink` | 缩量程度 |
| `breakout_range` | 突破位置 |

### 3.3 资金流因子 (9 个)

| 因子 | 说明 |
|------|------|
| `mf_big_net` | 大单净额 |
| `mf_big_net_ratio` | 大单净额占比 |
| `mf_big_cumsum_s/m/l` | 大单累计（短/中/长） |
| `mf_sm_proportion` | 散户占比 |
| `mf_flow_imbalance` | 流动不平衡度 |
| `mf_big_momentum` | 大单动量 |
| `mf_big_streak` | 大单连续流入天数 |

### 3.4 量子相干因子 (4 个)

| 因子 | 说明 |
|------|------|
| `coherence_l1` | l1 相干性 |
| `purity_norm` | 归一化纯度 |
| `von_neumann_entropy` | 冯诺依曼熵 |
| `coherence_decay_rate` | 退相干速率 |

## 4. 输出

### 4.1 个股画像 JSON

```json
{
  "symbol": "sh600519",
  "name": "贵州茅台",
  "analysis_date": "20260416",
  "data_range": ["20240321", "20260416"],
  "n_rows": 501,
  "forward_days": 3,
  "decay_lambda": 0.007,
  "top_positive_factors": [
    {"factor": "mf_big_cumsum_m", "ic": 0.12, "ic_ir": 0.85, "recent_ic": 0.15, "direction": "越大越涨"},
    ...
  ],
  "top_negative_factors": [
    {"factor": "perm_entropy_m", "ic": -0.08, "ic_ir": 0.62, "recent_ic": -0.11, "direction": "越大越跌"},
    ...
  ],
  "quintile_analysis": {
    "mf_big_cumsum_m": {"Q1": -0.2, "Q2": -0.05, "Q3": 0.01, "Q4": 0.10, "Q5": 0.25}
  }
}
```

### 4.2 全市场汇总 CSV

```
factor, n_positive, n_negative, n_insignificant, median_abs_ic, median_ic_ir, pct_effective
```

哪些因子在多数股票上有效 → 普适因子；只在少数股票上有效 → 个股特异因子。

### 4.3 行业汇总 CSV

按行业聚合，检测行业共性因子模式。

## 5. 使用方式

```bash
# 全市场分析（使用特征缓存）
python -m src.strategy.entropy_accumulation_breakout.factor_profiling \
  --cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --out_dir results/factor_profiling \
  --forward_days 3 \
  --decay_lambda 0.007

# 单只股票
python -m src.strategy.entropy_accumulation_breakout.factor_profiling \
  --cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --out_dir results/factor_profiling \
  --symbols sh600519,sz000001

# 快速验证 (top 50 按成交额)
python -m src.strategy.entropy_accumulation_breakout.factor_profiling \
  --cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --out_dir results/factor_profiling \
  --top_n 50
```

## 6. 后续应用

1. **自适应阈值**：根据个股画像动态调整策略参数
2. **因子组合**：对每只股票选取 top-3 有效因子构建个性化打分
3. **动态权重**：accumulation_quality / bifurcation_quality 的因子权重根据 IC 调整
4. **因子监控**：定期重跑，检测因子漂移（IC 从正变负 → 因子失效预警）
