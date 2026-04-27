# 数据管道修复 + 因子去冗余 + 周线特征

**日期**: 2026-04-27
**前置**: 14 文档审查 (问题1/2/3)

---

## 一、问题 1：修复数据管道 (13 个因子 NaN)

### Bug
`train_attention.py` 第 71 行 `build_features(df_daily=df, symbol=symbol)` 没传 `data_root` 参数。

### 修复
- `_compute_one_symbol_factors()` 新增 `data_root` 参数
- 所有 3 个调用点传入 `os.path.dirname(daily_dir)`
- 效果：激活 12 个资金流因子 (mf_*) + 周线数据

### 激活的因子
| 类型 | 数量 | 因子名 |
|------|------|--------|
| 资金流 | 12 | mf_big_net, mf_big_net_ratio, mf_big_cumsum_s/m/l, mf_sm_proportion, mf_flow_imbalance, mf_big_momentum, mf_big_streak, mf_cumsum_s, mf_cumsum_m, mf_impulse, net_mf_vol |
| 订单流 | 2 | buy_sm_amount, buy_sm_vol |
| 周线 | 5 | w_pe_ttm_pctl, w_pb_pctl, w_weekly_big_net_cumsum, w_weekly_turnover_shrink, w_weekly_turnover_ma4 |
| 估值/市值 | 9 | pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, circ_mv, total_mv |

注：4 个分钟线因子 (intraday_*) 因无 trade/ 目录仍为 NaN，已排除。

---

## 二、问题 2：因子去冗余 (76 → 51)

### 方法
基于 30 只股票（共 1800 样本）的因子相关性矩阵，手动去除 |corr| > 0.85 的冗余因子。

### 删除的 25 个因子
| 类型 | 因子 | 原因 |
|------|------|------|
| 镜像 buy/sell | sell_elg_amount/vol, sell_lg_amount/vol, sell_md_amount/vol, sell_sm_amount/vol | 与对应 buy 列 |r|>0.94 |
| 买档间高相关 | buy_elg_amount/vol, buy_lg_amount/vol, buy_md_amount/vol | 与 buy_sm |r|>0.85 |
| 其他高共线 | big_net_ratio, ps_ttm, purity_norm, von_neumann_entropy, entropy_slope, dv_ratio, total_mv | 对应保留列 |r|>0.89 |
| 分钟线(无数据) | intraday_perm_entropy, intraday_path_irrev, intraday_vol_concentration, intraday_range_ratio | 始终 NaN |

### 保留的 51 个因子分组
| 组 | 数量 | 示例 |
|----|------|------|
| entropy | 8 | perm_entropy_s/m/l, path_irrev_m/l, entropy_accel |
| volatility | 8 | volatility_m/l, vol_compression, bbw, bbw_pctl |
| eigenvalue | 2 | dom_eig_m/l |
| coherence | 2 | coherence_l1, coherence_decay_rate |
| money_flow | 12 | mf_big_net, mf_big_cumsum_s/m/l, mf_flow_imbalance |
| order_flow | 2 | buy_sm_amount/vol |
| price | 3 | breakout_range, volume_ratio, purity |
| valuation | 6 | pe, pe_ttm, pb, ps, dv_ttm, circ_mv |
| weekly | 5 | w_pe_ttm_pctl, w_pb_pctl, w_weekly_big_net_cumsum |
| meta | 2 | turnover_rate_f, big_net_ratio_ma |

---

## 三、问题 3b：周线独有特征

### 做法（不是双通道）
- 从 `build_features()` 返回的 `weekly` DataFrame 取最后一行
- 作为静态特征 append 到日线每一行
- 模型不变，输入维度 +5

### 新增的 5 个周线特征
| 特征 | 含义 | 数据来源 |
|------|------|---------|
| w_pe_ttm_pctl | PE 的 52 周分位数 | tushare-weekly-5d/ |
| w_pb_pctl | PB 的 52 周分位数 | tushare-weekly-5d/ |
| w_weekly_big_net_cumsum | 周线大单净额累计 (4周) | tushare-weekly-5d/ |
| w_weekly_turnover_shrink | 周线换手率萎缩 | tushare-weekly-5d/ |
| w_weekly_turnover_ma4 | 周线换手率 4 周均值 | tushare-weekly-5d/ |

### 为什么不是信息重叠
这些特征需要周频采样和周频聚合，日线算不出来：
- PE/PB 分位数需要周频采样 (52 周窗口)
- 大单累计需要周频聚合 (排除日内噪音)

### 覆盖情况
29/29 股票有完整周线特征（万科 pe_ttm 为 NaN 因亏损，但代码仍会添加列填充 NaN 保证因子一致）

---

## 四、代码改动汇总

### train_attention.py
1. `_compute_one_symbol_factors()`: 新增 `data_root` 参数，传入 `build_features()`
2. 新增周线特征追加逻辑：取 weekly 最后一行 append 到日线
3. 新增 `_dedup_factors()`: 手动去冗余函数
4. 在 3 个数据构建入口调用 `_dedup_factors()`

### attention_learner.py
1. `FACTOR_COLUMNS`: 从 37 个 (13 个无效) 更新为 52 个完整因子
2. `FACTOR_GROUPS`: 从 9 组更新为 10 组，新增 weekly 组

---

## 五、基线对比

| 指标 | 旧基线 | 新基线 (待重跑) |
|------|--------|----------------|
| 有效因子数 | 34 (13 个 NaN) | 51 (全部有效) |
| 包含资金流 | 否 | 是 (12 个) |
| 包含周线特征 | 否 | 是 (5 个) |
| 包含估值/市值 | 否 | 是 (9 个) |
| IC 基线 | 0.0656 | ? |
| P&L Sharpe | 1.05 | ? |
