---
name: backtest-analyze
description: "运行回测或分析回测结果。Use when: 回测, backtest, 分析结果, analyze results, 收益率, Sharpe, 最大回撤, 胜率, equity curve, 净值曲线, 前瞻回测, forward backtest, 绩效分析。"
argument-hint: "回测类型或结果目录，例如：多时间框架回测、分析 entropy_backtest_v2"
---

# 回测与结果分析 (Backtest & Analyze)

运行前瞻回测并分析策略绩效指标。

## 使用场景

- 对策略进行前瞻滚动回测（指定日期范围 + 持有天数）
- 分析已有回测结果的绩效
- 比较不同策略/参数组合的表现
- 诊断信号质量（因子分布、入场条件）

## 回测方式

### 1. 内置前瞻回测（推荐）

多时间框架和熵分岔策略均支持 `--backtest-start-date` / `--backtest-end-date` 参数：

```bash
# 多时间框架共振回测
./scripts/run_multitimeframe_resonance_scan.sh \
  --backtest-start-date 20260101 \
  --backtest-end-date 20260331 \
  --hold-days 5 \
  --max-positions 10

# 熵分岔 4 层系统回测
./scripts/run_entropy_bifurcation_setup.sh \
  --backtest-start-date 20260101 \
  --backtest-end-date 20260331 \
  --hold-days 5
```

回测逻辑：每个扫描日滚动选股 → 次日开盘买入 → 持有 N 天卖出 → 记录收益。

### 2. 独立回测脚本

```bash
# 四层系统专用回测
python scripts/backtest_four_layer_2025.py

# 熵因子回测
python tests/backtest_entropy_factors.py

# 熵回测 v2
python tests/backtest_entropy_v2.py

# 多日回测
python tests/backtest_multi_day.py

# 分岔 v3 回测
python tests/backtest_bifurcation_v3.py
```

## 结果分析

### 分析脚本

```bash
# 通用回测结果分析
python tests/analyze_backtest_results.py --results-dir <dir>

# 熵回测专项分析
python tests/analyze_entropy_backtest.py

# 因子分布分析
python tests/analyze_factor_distribution.py

# 终版回测分析
python tests/analyze_final_backtest.py

# 快速因子分析
python tests/quick_factor_analysis.py

# 验证熵因子
python tests/validate_entropy_factors.py
```

### 结果目录说明

回测结果存放在 `results/` 下：

| 目录 | 说明 |
|------|------|
| `multitimeframe_resonance/` | 多时间框架共振扫描 & 回测 |
| `entropy_bifurcation_setup/` | 熵分岔 4 层系统 |
| `four_layer_backtest_2025/` | 四层系统 2025 回测 |
| `four_layer_backtest_2025_v2/` | 四层 v2 (改进参数) |
| `four_layer_backtest_2025_v3_entropy_exit/` | v3 熵退出 |
| `four_layer_backtest_2025_v4_pure_entropy/` | v4 纯熵 |
| `entropy_backtest/` | 熵因子回测 |
| `entropy_backtest_v2/` | 熵因子 v2 |
| `bifurcation_v2_backtest/` | 分岔 v2 |
| `bifurcation_v3/` | 分岔 v3 |

### 关键绩效指标

分析时关注以下指标：
- **总收益率** — 累计收益
- **年化收益** — 年化后收益率
- **Sharpe 比率** — 风险调整后收益（>1 较好，>2 优秀）
- **最大回撤** — 最大峰谷回撤幅度
- **胜率** — 盈利交易占比
- **盈亏比** — 平均盈利 / 平均亏损
- **交易次数** — 信号频率评估

### 交互式分析

用 Python 读取 CSV 结果文件进行自定义分析：

```python
import pandas as pd

# 读取回测交易记录
trades = pd.read_csv('results/multitimeframe_resonance/live_market_scan/forward_backtest_trades_<date>.csv')

# 读取汇总
summary = pd.read_csv('results/multitimeframe_resonance/live_market_scan/forward_backtest_summary_<date>.csv')
```
