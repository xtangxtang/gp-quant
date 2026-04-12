---
name: market-scan
description: "运行多时间框架共振扫描选股。Use when: 选股, 扫描市场, 共振候选, resonance scan, market scan, 找股票, 今日推荐, top candidates, 多时间框架, multitimeframe。"
argument-hint: "扫描日期或股票代码，例如：20260310 或 sh600000,sz000001"
---

# 多时间框架共振扫描 (Market Scan)

对全市场进行日线/周线/月线共振扫描，筛选趋势共振候选股票。

## 使用场景

- 每日收盘后扫描全市场，获取共振候选股票 Top N
- 指定日期回顾历史共振信号
- 小样本验证（指定股票列表）
- 前瞻回测（指定起止日期）

## 执行步骤

### 1. 确认参数

与用户确认以下关键参数（括号内为默认值）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--scan-date` | 扫描日期 YYYYMMDD | 自动推断最新交易日 |
| `--top-n` | 输出前 N 只候选 | 30 |
| `--symbols` | 逗号分隔股票列表（小样本） | 全市场 |
| `--min-amount` | 最低成交额 | 500000 |
| `--min-turnover` | 最低换手率 | 1.0 |
| `--include-st` | 是否包含 ST | 默认排除 |
| `--hold-days` | 回测持有天数 | 5 |
| `--max-positions` | 最大持仓数 | 10 |
| `--max-positions-per-industry` | 每行业最大持仓 | 2 |
| `--backtest-start-date` | 前瞻回测起始日 | 无 |
| `--backtest-end-date` | 前瞻回测结束日 | 无 |

### 2. 运行扫描

```bash
cd /nvme5/xtang/gp-workspace/gp-quant
./scripts/run_multitimeframe_resonance_scan.sh [参数]
```

**常用示例**：

```bash
# 默认扫描（最新交易日，Top 30）
./scripts/run_multitimeframe_resonance_scan.sh

# 指定日期扫描
./scripts/run_multitimeframe_resonance_scan.sh --scan-date 20260310

# 小样本验证
./scripts/run_multitimeframe_resonance_scan.sh \
  --symbols sh600000,sz000001 \
  --out-dir /tmp/gp_quant_resonance_smoke

# 前瞻回测
./scripts/run_multitimeframe_resonance_scan.sh \
  --backtest-start-date 20260201 \
  --backtest-end-date 20260310 \
  --hold-days 5
```

### 3. 查看结果

结果默认输出到 `results/multitimeframe_resonance/live_market_scan/`：

- `market_scan_snapshot_<date>.csv` — 全市场状态快照
- `resonance_candidates_<date>_all.csv` — 所有共振候选
- `resonance_candidates_<date>_top30.csv` — Top 30 候选
- `selected_portfolio_<date>_top30.csv` — 经过 ST/流动性/行业过滤后的组合
- `forward_backtest_daily_<date>.csv` — 回测每日统计
- `forward_backtest_trades_<date>.csv` — 回测逐笔交易
- `forward_backtest_summary_<date>.csv` — 回测汇总指标（总收益、Sharpe、最大回撤、胜率）

### 4. 解读结果

帮助用户分析 CSV 中的关键列：
- `resonance_score` — 共振得分（日/周/月综合）
- `d_score`, `w_score`, `m_score` — 各时间框架物理得分
- 前瞻回测关注：收益率、Sharpe 比率、最大回撤、胜率
