# Agent 1: 因子计算 (Factor Snapshot)

> 代码: `agent_factor.py` → `run_factor_snapshot()`

---

## 职责

从特征缓存中提取截止 `scan_date` 的全市场因子截面。**只负责获取数据，不做选股、不做回测。**

## 输入 / 输出

| 项目 | 内容 |
|------|------|
| **输入** | `cache_dir` (特征缓存根目录), `scan_date`, `basic_path` (股票基本信息) |
| **输出** | `DataFrame`, index=symbol, 每只股票一行，包含所有因子值 + 元信息列 |

## 处理流程

```
1. 加载基本信息 (tushare_stock_basic.csv → {symbol: {name, industry}})
2. 加载日线因子截面 (cache_dir/daily/*.csv, 取 ≤ scan_date 的最后一行)
3. 加载周线因子截面 (cache_dir/weekly/*.csv, 取 ≤ scan_date 的最后一行)
4. 合并: 周线因子加 w_ 前缀并入日线截面 (left join on symbol)
5. 过滤: ST/退市 + 流动性 (近20日均成交额 ≥ 5000万)
6. 附加元信息: _name, _industry
```

## 因子列表

### 日线因子 (32 个, 来自 `feature_engine.py` 的 daily 缓存)

| 类别 | 因子名 | 说明 |
|------|--------|------|
| **置换熵** | `perm_entropy_s/m/l` | 10/20/60 日滚动窗口，时间序列有序度 |
| **熵动态** | `entropy_slope`, `entropy_accel` | 熵的一阶/二阶导数 |
| **路径不可逆** | `path_irrev_m/l` | 20/60 日 KL 散度 (正向 vs 反向回报) |
| **临界减速** | `dom_eig_m/l` | 20/60 日主特征值 (自相关矩阵) |
| **换手熵** | `turnover_entropy_m/l` | 换手率分布的熵 |
| **波动率** | `volatility_m/l` | 20/60 日收益标准差 |
| **波动压缩** | `vol_compression`, `bbw_pctl` | 压缩度、布林带百分位 |
| **量能** | `vol_ratio_s`, `vol_impulse`, `vol_shrink`, `breakout_range` | 量比/缩量/突破幅度 |
| **资金流** | `mf_big_net`, `mf_big_net_ratio` | 大单净额、占比 |
| **资金累计** | `mf_big_cumsum_s/m/l` | 5/20/60 日大单累计 |
| **散户** | `mf_sm_proportion`, `mf_flow_imbalance` | 散户占比、资金失衡 |
| **资金动量** | `mf_big_momentum`, `mf_big_streak` | 大单动量、连续天数 |
| **量子相干** | `coherence_l1`, `purity_norm`, `von_neumann_entropy`, `coherence_decay_rate` | 密度矩阵指标 |

### 周线因子 (30 个, 合并时加 `w_` 前缀)

与日线同名的 22 个 + 6 个周线特有:
- `pe_ttm_pctl`, `pb_pctl` — 估值百分位
- `weekly_big_net`, `weekly_big_net_cumsum` — 周线大单资金流
- `weekly_turnover_ma4`, `weekly_turnover_shrink` — 周线换手

## 加载细节

### `_load_snapshot()` 核心逻辑

```python
for each CSV in cache_dir/{daily,weekly}/:
    1. 读取 CSV, 若行数 < min_rows (日线60, 周线30), 跳过
    2. 按 trade_date 排序
    3. 过滤 trade_date ≤ scan_date
    4. 取最后一行作为该股票的因子快照
    5. 附加: _trade_date = 最新日期, _avg_amount_20 = 近20日均成交额
```

### 元信息列 (以 `_` 开头, 不参与模型训练)

| 列名 | 说明 |
|------|------|
| `_trade_date` | 日线因子对应的最后交易日 |
| `_weekly_trade_date` | 周线因子对应的最后交易日 |
| `_avg_amount_20` | 近 20 日日均成交额 (万元) |
| `_name` | 股票名称 |
| `_industry` | 行业分类 |

## 过滤规则

1. **ST / 退市**: 名称含 "ST" 或 "退" → 排除
2. **流动性**: `_avg_amount_20 < 5000` (万元, 即日均成交额 < 5000 万) → 排除
3. **数据量**: 日线缓存不足 60 行、周线缓存不足 30 行 → 跳过

**典型结果**: ~5500 只全市场 → 过滤后 ~5100 只

## 性能

- 加载时间: ~50-60s (读取 ~5500 个 CSV, 每个 ~2000 行)
- 输出规模: ~5100 行 × ~245 列 (163 日线因子 + 80 周线因子 + 元信息)
