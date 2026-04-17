# 特征缓存生成流程 (Feature Cache Pipeline)

> 记录 `feature-cache/daily/` 和 `feature-cache/weekly/` 的完整生成过程。

---

## 1. 触发方式

### 1.1 策略扫描时自动生成

```bash
python -m src.strategy.entropy_accumulation_breakout.run_entropy_accumulation_breakout \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --feature_cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --scan_date 20260417
```

传入 `--feature_cache_dir` 即启用缓存，扫描过程中对每只股票自动计算并写入。

### 1.2 Supervisor DAG 调度

`src/agents/supervisor.py` 中 `entropy_scan` agent 在 `derived` + `market_data` 完成后自动触发，调用 `src/agents/agent_entropy_scan.py`。

### 1.3 Shell 脚本入口

```bash
./scripts/run_entropy_accumulation_breakout.sh --scan-date 20260417
```

默认路径: `DEFAULT_FEATURE_CACHE_DIR="/nvme5/xtang/gp-workspace/gp-data/feature-cache"`

---

## 2. 数据来源

| 原始数据 | 路径 | 用途 | 列数 |
|---------|------|------|------|
| 日线 OHLCV | `gp-data/tushare-daily-full/{symbol}.csv` | 行情基础 | ~24 列 |
| 资金流 | `gp-data/tushare-moneyflow/{symbol}.csv` | 大单/散户分层 | ~16 列 |
| 周线预计算 | `gp-data/tushare-weekly-5d/{symbol}.csv` | PE/PB/换手估值 | ~40 列 |
| 分钟线 | `gp-data/trade/{symbol}/{YYYY-MM-DD}.csv` | 日内微观结构 | ~10 列 |

---

## 3. 代码调用链

```
run_entropy_accumulation_breakout.py  (CLI 入口)
  → scan_service.run_scan()
    → scan_single_symbol()           (逐只股票)
      → feature_cache.get_cached_daily_features()
        ├── 缓存命中 → 直接读取 CSV 返回
        └── 缓存 miss/过期
            → feature_engine.build_features()
              ├── _load_daily()           → 读 tushare-daily-full
              ├── load_moneyflow()        → 读 tushare-moneyflow，合并
              ├── _compute_entropy()      → 置换熵/路径不可逆/主特征值
              ├── _compute_volatility()   → 波动率/布林带/量比
              ├── _compute_moneyflow()    → 大单行为/散户占比
              ├── _compute_quantum()      → 量子相干性指标
              └── _load_minute_day()      → 分钟线微观结构 (最近几天)
            → feature_cache._write_cache()  → 写入 CSV

      → feature_cache.get_cached_weekly_features()
        ├── 缓存命中 → 直接读取
        └── 缓存 miss/过期
            → load_weekly_precomputed() → 读 tushare-weekly-5d
            → 同上 entropy/volatility/quantum 计算 (窗口改为 4/8/24 周)
            → 追加周线特有列 (pe_ttm_pctl, pb_pctl, weekly_big_net 等)
            → _write_cache()
```

---

## 4. Daily 缓存结构

**路径**: `feature-cache/daily/{symbol}.csv`
**规模**: 83 列 × ~500 行/每股 × 5169 只

### 列分组

| 列号 | 类别 | 列名 | 计算方法 |
|------|------|------|---------|
| 1-26 | 原始行情 | `ts_code, trade_date, open, high, low, close, ...` | 直接来自 tushare-daily-full |
| 27-44 | 资金流原始 | `buy_elg_amount, sell_sm_amount, net_mf_amount, ...` | 来自 tushare-moneyflow 合并 |
| 45-47 | 置换熵 | `perm_entropy_s/m/l` | 10/20/60 日滚动窗口置换熵 |
| 48 | 熵斜率 | `entropy_slope` | `perm_entropy_s - perm_entropy_l` |
| 49 | 熵加速度 | `entropy_accel` | 置换熵 5 日变化率 |
| 50-51 | 路径不可逆 | `path_irrev_m/l` | 20/60 日 KL 散度 (正向 vs 反向回报) |
| 52-53 | 临界减速 | `dom_eig_m/l` | 20/60 日主特征值 (自相关矩阵) |
| 54-55 | 换手率熵 | `turnover_entropy_m/l` | 20/60 日换手率分布熵 |
| 56-57 | 波动率 | `volatility_m/l` | 20/60 日收益标准差 |
| 58 | 波动压缩 | `vol_compression` | `volatility_m / volatility_l` |
| 59-60 | 布林带 | `bbw`, `bbw_pctl` | 布林带宽 + 120 日历史百分位 |
| 61-63 | 量能 | `vol_ratio_s`, `vol_impulse`, `vol_shrink` | 量比、量能脉冲、缩量度 |
| 64 | 突破幅度 | `breakout_range` | 价格相对布林带中轨偏离 |
| 65-69 | 资金流累计 | `mf_cumsum_s/m`, `mf_impulse`, `big_net_ratio`, `big_net_ratio_ma` | 短/中期净流入累积 |
| 70-74 | 量子相干 | `coherence_l1`, `purity`, `von_neumann_entropy`, `coherence_decay_rate`, `purity_norm` | 密度矩阵相干指标 |
| 75-83 | 大单行为 | `mf_big_net`, `mf_big_net_ratio`, `mf_big_cumsum_s/m/l`, `mf_sm_proportion`, `mf_flow_imbalance`, `mf_big_momentum`, `mf_big_streak` | 大单净额/累积/散户占比/连续性 |

---

## 5. Weekly 缓存结构

**路径**: `feature-cache/weekly/{symbol}.csv`
**规模**: 80 列 × ~1180 行/每股 × 5169 只

### 与 Daily 的区别

- **前 74 列**: 与 daily 相同的指标，但滚动窗口改为周级别 (4/8/24 周)
- **额外 6 列** (周线特有):

| 列 | 说明 | 来源 |
|----|------|------|
| `pe_ttm_pctl` | PE_TTM 历史百分位 | tushare-weekly-5d |
| `pb_pctl` | PB 历史百分位 | tushare-weekly-5d |
| `weekly_big_net` | 周度大单净额 | 周内资金流聚合 |
| `weekly_big_net_cumsum` | 大单净额累计 | 逐周滚动累加 |
| `weekly_turnover_ma4` | 4 周平均换手率 | 周换手移动平均 |
| `weekly_turnover_shrink` | 周换手萎缩度 | 近期/远期换手比 |

---

## 6. 增量更新机制

```
缓存检查:
  cached_last_date >= raw_last_date?
  ├── YES → 缓存命中，直接返回 (0 计算)
  └── NO  → 计算新增行数 num_new
            ├── num_new ≤ 50 (daily) / 20 (weekly)
            │   → 增量更新:
            │     1. 取 cached_end - lookback 到末尾 (overlap 区间)
            │     2. 计算 overlap + new 区间的特征 (滚动窗口需要回溯)
            │     3. 只提取 new 行
            │     4. concat [cached_old ... new_rows] 写入
            └── num_new > 50 / 20
                → 全量重算整个序列
```

**Lookback 常量**:
- Daily: `_MAX_LOOKBACK = 130` (bbw_pctl 需要 120 日)
- Weekly: `lookback = 30`

**性能**:
- 首次全量: ~10 分钟 (5500 只, 16 进程)
- 后续增量: ~17 倍加速 (仅计算 1-5 天新数据)

---

## 7. 缓存管理工具

```python
from src.strategy.entropy_accumulation_breakout.feature_cache import (
    invalidate_cache,  # 清除缓存 (全量或单只)
    cache_stats,       # 统计 {'daily': {'count': 5169, 'size_mb': 2340}}
)

# 清除单只股票缓存 (强制下次重新计算)
invalidate_cache('/path/to/feature-cache', symbol='sh600519')

# 清除全部缓存
invalidate_cache('/path/to/feature-cache')
```

---

## 8. 文件命名规则

- Daily: `feature-cache/daily/{exchange}{code}.csv` — 如 `sh600519.csv`, `sz000001.csv`, `bj920000.csv`
- Weekly: `feature-cache/weekly/{exchange}{code}.csv` — 同上
- 小写前缀 (`sh/sz/bj`)，与 tushare-daily-full 目录一致
