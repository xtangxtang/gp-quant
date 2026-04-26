# 高维度审视：因子质量 + 模型选择 + 数据管道问题

**日期**: 2026-04-26
**前置**: 全部实验完成后的系统性审查

---

## 一、因子计算的三个问题

### 1. 13 个因子在训练时是 NaN

`train_attention.py` 第 72 行调用 `build_features()` 时**没有传 `data_root` 和 `symbol`**：

```python
result = build_features(df_daily=df, symbol=symbol)
# 缺少 data_root → 资金流、分钟线、周线数据全部无法加载
```

导致以下因子在训练时全部为 NaN：
- 资金流（9 个）：`mf_big_net`, `mf_big_net_ratio`, `mf_big_cumsum_s/m/l`, `mf_sm_proportion`, `mf_flow_imbalance`, `mf_big_momentum`, `mf_big_streak`
- 分钟线（4 个）：`intraday_perm_entropy`, `intraday_path_irrev`, `intraday_vol_concentration`, `intraday_range_ratio`

**模型实际只在用 ~34 个日线技术因子训练。**

### 2. 有效因子高度共线

47 个因子的有效信息维度大概只有 10-15 个：

| 组 | 高度共线的因子对 | 原因 |
|---|---|---|
| 熵 | `perm_entropy_s/m/l` | 同一函数不同窗口（10/20/60），相关系数通常 > 0.8 |
| 熵 | `entropy_slope` = `perm_entropy_s - perm_entropy_l` | 线性组合 |
| 波动率 | `volatility_m` vs `bbw` | 都是收益率标准差变体 |
| 波动率 | `vol_compression` = `volatility_m / volatility_l` | 比值 |
| 资金流 | `mf_cumsum_s/m`, `mf_big_cumsum_s/m/l` | 同一变量不同窗口求和 |
| 特征值 | `dom_eig_m/l` | 同一函数两个窗口 |

冗余因子的后果：增加 embedding 参数、分散 attention、放大过拟合风险。

### 3. 缺少正交信息维度

34 个有效因子全部来自同一个信息源：**日线 OHLCV + 换手率**。数学变换再多（熵、特征值、相干性），输入都是 close/vol/turnover_rate。

缺失的正交维度：

| 维度 | 数据源 | 因子示例 | 为什么正交 |
|------|--------|---------|-----------|
| **基本面** | `tushare-fina_indicator/` | ROE, 营收增速, 现金流/市值 | 财报 vs 交易，信息来源完全不同 |
| **估值** | `tushare-daily-full/` 已有 | PE_TTM, PB, PS | 已在 CSV 中但被 exclude 集过滤掉了 |
| **行业相对** | 行业均值 | 个股 PE/行业 PE, 个股涨幅-行业涨幅 | 剥离行业 beta |
| **事件/情绪** | 外部 | 股东增减持、机构调研 | 非价格信息 |

---

## 二、周线数据的正确用法

之前 multi-scale 失败是因为"周线 = 日线 5 日均值"，信息完全重叠。

但 `feature_engine.py` 里有一套**完整的周线独有特征**（`compute_weekly_extra_features`）：
- `pe_ttm_pctl`：PE 的 52 周分位数（估值位置）
- `weekly_big_net_cumsum`：周线大单净额累计（中期资金趋势）
- `weekly_turnover_shrink`：周线换手率萎缩（中期流动性）

这些是日线算不出来的信息。正确做法：
- 不是双 Transformer 双通道（太多参数）
- 而是把周线特征的**最新值** append 到日线因子里，作为额外 5-8 个截面特征
- 模型仍然是单通道 575K 参数，但输入从 47 个因子变成 52-55 个
- 周线带来的是**正交信息**（估值位置、中期资金），不是日线重复

---

## 三、Transformer 是否真的胜任

**目前没有证据表明 Transformer 优于简单模型。**

### 对比

| 模型 | 典型 IC | 参数量 | 训练时间 |
|------|---------|--------|---------|
| 线性 IC 加权 | 0.03-0.05 | 0 | 秒级 |
| LightGBM（截面） | 0.05-0.08 | ~1K 有效参数 | 分钟级 |
| **当前 Transformer** | **0.066** | **575K** | **小时级** |
| 业界 SOTA Transformer | 0.06-0.10 | 1-10M | GPU 天级 |

575K 参数的 Transformer 得到 IC=0.066，和截面 LightGBM 差不多。而 LightGBM：
- 不需要 60 天序列（只看最新截面）
- 训练快 100 倍
- 过拟合风险小（树深度限制是天然正则化）
- Bull Hunter v3 就是用 LightGBM 做的

### 过拟合比

Val IC = 0.91, Eval IC = 0.066 → **过拟合比 > 13:1**

说明模型更多在记忆训练集的噪声模式，而非学到可泛化的时序规律。

### 该做的验证

用同样的 47 个因子（截面最新值），跑一个 LightGBM 回归（目标 = 未来 10 天收益），用同样的 walk-forward split。如果 LightGBM IC ≥ 0.05，说明 Transformer 的 60 天时序窗口几乎没有额外贡献。

---

## 四、行动计划（按优先级）

| # | 行动 | 投入 | 预期收益 | 说明 |
|---|------|------|---------|------|
| 1 | **修复数据管道** | 低（改 1 行代码） | **最高** | 传入 `data_root` 激活 13 个资金流/分钟线因子 |
| 2 | **因子去冗余** | 低 | 中 | 相关性 > 0.85 的保留一个，47 → 20-25 个 |
| 3 | **纳入已有估值列** | 低 | 中 | PE/PB/total_mv 已在 CSV 中，只需从 exclude 集移除 |
| 4 | **LightGBM 基线对比** | 中 | **决定性** | 如果 IC ≥ 0.05 → Transformer 的复杂度不合理，考虑切到 LightGBM |
| 5 | **周线独有特征** | 中 | 中 | 作为截面特征 append，不做双通道 |
| 6 | **基本面因子** | 中 | 高 | PE/ROE/营收增速，独立于价格信息 |
| 7 | **行业中性化** | 中 | 中 | 在小盘域内做行业中性化 |

### 执行顺序

**1 → 2+3 → 4（关键分支点）→ 5+6 → 7**

第 4 步的结果决定后续路线：
- LightGBM IC < 0.04 → Transformer 的时序窗口有独立贡献，继续优化 Transformer
- LightGBM IC ≈ 0.05-0.07 → Transformer 无额外贡献，切到 LightGBM（训练快 100 倍，迭代效率更高）
- LightGBM IC > 0.07 → Transformer 反而不如 LightGBM，直接切换
