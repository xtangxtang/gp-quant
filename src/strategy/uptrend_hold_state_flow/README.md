# Uptrend Hold State Flow

## 描述

一句话概括：把“上升趋势里是否继续持有”的三套判断放进同一张状态图里，输入买点和评估日后，判断这段持有路径经历了哪些状态切换，以及当前是否还适合继续持有。

这个总策略内部包含三个子策略目录：

- `entropy_hold_judgement`：熵秩序持有
- `rapid_expansion_hold`：快速扩张持有
- `rapid_expansion_exhaustion_exit`：快速扩张衰竭退出

状态图按下面的流转理解：

| 状态 | 含义 | 主要看什么 |
| --- | --- | --- |
| `observation` | 当前还不属于这三类明确持有状态 | 三套状态都没有充分激活 |
| `entropy_hold_judgement` | 仍属于低熵有序持有 | `hold_score`、`entropy_reserve`、`disorder_pressure` |
| `rapid_expansion_hold` | 已进入快速扩张持有区 | `expansion_thrust`、`directional_persistence`、`acceptance_score` |
| `rapid_expansion_exhaustion_exit` | 进入快速扩张末端衰竭退出区 | `peak_extension_score`、`deceleration_score`、`fragility_score` |

## 主要参数

- `symbol_or_name`：股票代码或股票名称，例如 `sh688268`、`688268.SH` 或 `华特气体`
- `start_date`：买点日期，从该日开始评估持有路径
- `scan_date`：路径评估终点，默认自动推断最新交易日
- `data_dir`：日线 CSV 目录
- `basic_path`：`tushare_stock_basic.csv` 路径，用于名称和代码映射
- `lookback_years`：特征回看年数，默认 5

## 输出

策略会输出：

- 当前股票从买点以来经历的状态路径与切换摘要
- 当前路径状态的解释、继续持有/退出/重新评估结论以及下一步观察重点
- 三套子策略在扫描日时点的得分，以及它们在整段路径中是否被真正激活过
- 一份用于 web 页面高亮状态图的候选状态表，以及一份逐日路径诊断表
