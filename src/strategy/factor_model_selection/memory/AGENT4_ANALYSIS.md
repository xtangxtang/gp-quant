# Agent 4: 策略分析 (Effectiveness & Bull Stock Analysis)

> 代码: `agent_analysis.py` → `run_analysis()`

---

## 职责

**4 件事**:
1. 评判策略是否有效 (胜率/收益是否达标)
2. 发现全市场同期大牛股 (涨幅 Top 30)
3. 分析策略是否抓住了大牛股; 没抓住的原因是什么
4. 量化被选中股票的收益贡献分布

## 输入 / 输出

| 项目 | 内容 |
|------|------|
| **输入** | `factor_snapshot` (Agent 1), `selections` (Agent 2), `validation_results` (Agent 3), `data_dir`, `scan_date` |
| **输出** | 结构化分析报告 dict + 控制台打印 |

## 输出结构

```python
{
    "effectiveness": {                    # 全局有效性
        "effective": bool,                # 是否超半数 horizon 达标
        "effective_horizons": int,
        "total_horizons": int,
        "ratio": float,
    },
    "per_horizon": {                      # 逐 horizon 分析
        "3d": {
            "metrics": {...},             # Agent 3 的回测指标
            "effectiveness": {...},       # 是否达标 + 原因
            "entry_date": "20250103",
            "exit_date": "20250108",
            "hold_days": 3,
            "bull_stocks": [...],         # 全市场涨幅 Top 30
            "catch_analysis": {...},      # 大牛股捕获分析
            "contribution": {...},        # 收益贡献分析
        },
        ...
    },
    "summary": "策略有效 (3/5 个 horizon 达标); 3d: 捕获 0/30 只大牛股 (0%)"
}
```

---

## 模块 1: 策略有效性判定

### 阈值 (EFFECTIVENESS_THRESHOLDS)

| 指标 | 阈值 | 说明 |
|------|------|------|
| `min_win_rate` | 50% | 胜率必须过半 |
| `min_avg_pnl` | 0.3% | 平均收益必须为正且有意义 |
| `min_profit_loss` | 1.0 | 盈亏比 > 1 |
| `max_worst_trade` | -15% | 单笔最差不超过 -15% |

### 判定逻辑

```python
checks = {
    "win_rate":             win_rate >= 50%,
    "avg_pnl":              avg_pnl >= 0.3%,
    "worst_ok":             worst_trade >= -15%,
    "positive_expectation": avg_pnl > 0,
}
score = sum(checks.values()) / len(checks)  # 0~1
effective = (score >= 0.5)                   # 至少一半条件通过
```

### 全局判定

```python
overall_effective = (effective_horizons > total_horizons / 2)
```

---

## 模块 2: 大牛股发现

### `_find_bull_stocks()` 逻辑

```
输入: data_dir, entry_date, exit_date, top_n=30
处理:
  for each CSV in data_dir (~5500 只股票):
    1. 找 entry_date 的 open 价
    2. 找 exit_date 的 close 价
    3. 计算收益率 pnl = (close - open) / open * 100
    4. 过滤: 期间日均成交额 < 5000万 → 排除
  排序: 按 pnl 降序取 Top 30
输出: DataFrame[symbol, entry_price, exit_price, pnl_pct]
```

**关键**: 这一步遍历全市场 ~5500 个 CSV, 是 Agent 4 的主要耗时来源 (~30-40s/horizon)。

---

## 模块 3: 大牛股捕获分析

### `_analyze_bull_catch()` 逻辑

```
for each bull stock in Top 30:
    if stock in selected_stocks:
        → caught (抓住了), 记录实际交易收益
    else:
        → missed (错过了), 调用 _diagnose_miss() 分析原因
```

### `_diagnose_miss()` 诊断原因

按优先级依次检查:

| 优先级 | 检查 | 诊断结果 |
|--------|------|----------|
| 1 | 股票不在 factor_snapshot 中 | "不在因子截面中 (数据缺失/流动性过低/ST)" |
| 2 | 关键因子缺失 | "因子缺失: mf_sm_proportion" (最常见于北交所) |
| 3 | 成交额不足 | "成交额不足(3000)" |
| 4 | 以上都不是 | "因子值在截面中排名不够高 (模型评分低)" |

**检查的关键因子**: `perm_entropy_m`, `path_irrev_m`, `vol_shrink`, `mf_sm_proportion`

### 输出

```python
{
    "caught": [{"symbol": ..., "bull_pnl": ..., "actual_pnl": ...}],
    "missed": [{"symbol": ..., "bull_pnl": ..., "miss_reason": ...}],
    "catch_rate": 0.0,        # caught / total
    "n_bull": 30,
    "n_caught": 0,
    "caught_total_pnl": 0.0,
}
```

---

## 模块 4: 收益贡献分析

### `_analyze_contribution()` 逻辑

```python
sorted_trades = sorted(trades, by pnl_pct descending)
output:
  total_pnl: 所有交易收益之和
  top_contributor: 收益最高的股票 {symbol, name, pnl_pct}
  worst_performer: 收益最低的股票 {symbol, name, pnl_pct}
  pnl_distribution: {">5%": n, "2~5%": n, "0~2%": n, "-2~0%": n, "<-2%": n}
```

---

## 报告打印格式

```
======================================================================
  策略分析报告 — scan_date: 20250102
======================================================================

  总结: 策略有效 (3/5 个 horizon 达标); 3d: 捕获 0/30 只大牛股 (0%)

  策略有效性: ✓ 有效 (3/5)

  ──────────────────────────────────────────────────
  3d (持有3天, 20250103→20250108)
  ──────────────────────────────────────────────────
    交易: 5 笔, 胜率 80.0%, 均收益 +0.19%, ✓

    大牛股捕获: 0/30 (0%)
    ✗ 错过的 (Top 5):
      sz300927: 涨幅 +75.5%, 原因: 因子值在截面中排名不够高 (模型评分低)
      ...

    最大贡献: sh600018 上港集团 +2.5%
    最差表现: sh600026 中远海能 -2.0%
    收益分布: >5%=0, 2~5%=1, 0~2%=2, -2~0%=2, <-2%=0
```

---

## 性能

- **主要瓶颈**: `_find_bull_stocks()` 遍历全市场 ~5500 CSV
  - 每个 horizon 调用一次: ~30-40s
  - 5 个 horizon: ~150-200s
  - 占 Agent 4 总耗时的 ~95%
- **优化空间**: 不同 horizon 若 entry_date/exit_date 相同 (如 5d 和 1w 都是持有 5 天), 可共享结果; 或者做一次全市场预加载

## 2025Q1 回测发现

| 指标 | 结果 |
|------|------|
| 大牛股捕获率 | **0%** (全部 horizon, 全部扫描日) |
| 错过原因分布 | ~80% "模型评分低", ~15% "因子缺失 (mf_sm_proportion)", ~5% "不在截面中" |
| 短期有效率 | 3d: 10/12 (83%), 5d: 8/12 (67%) |
| 长期有效率 | 1w: 4/12 (33%), 3w: 2/4 (50%), 5w: 1/3 (33%) |
