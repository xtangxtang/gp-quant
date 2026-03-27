# Multi-Timeframe Resonance Analysis

## 描述

这个策略本质上不是高频交易，也不是简单均值回归，而是一个偏研究型的 A 股多周期共振选股策略。它先在日线、周线、月线三个周期上分别做状态识别，再去找“日线先启动、周线和月线同步支持”的股票，最后输出候选池、入选组合和前瞻回测结果。

从特征设计上看，它采用了经济物理风格的状态评分框架，把价格、成交额、换手率和资金流映射成五类特征：

- `energy`：资金流和成交额强度
- `temperature`：换手与波动的相对状态
- `order`：均线结构、突破状态
- `phase`：熵、Hurst 指数等秩序变化
- `switch`：从混沌向有序切换的触发项

这些特征会被合成为单周期 `score`，并要求持续若干 bar 后才认定为有效状态。随后策略再把周线和月线映射回日线时间轴，计算多周期 `support_count` 和 `resonance_score`。默认要求日线状态成立、日周月分数都过阈值、至少两个周期同时支持、共振分数达标，并连续满足若干天，才会认定为真正的 `resonance_state`。

落地到选股时，它还会叠加实盘约束，所以更像一个“共振选股器”而不是单纯的信号打分器。默认会过滤 ST 股票、低成交额股票、低换手率股票，并支持大盘状态过滤、组合最大持仓数限制以及单行业持仓上限限制。

一句话概括，这个策略是在找“日线率先有序化，且周线、月线同步确认”的股票，属于中短周期的多周期共振趋势选股策略。

## 主要参数

- `--scan_date`：扫描日期，格式 `YYYYMMDD`；不传则自动推断最新交易日
- `--top_n`：输出前 N 只共振候选，默认 `30`
- `--min_amount`：最低成交额过滤，默认 `500000`
- `--min_turnover`：最低换手率过滤，默认 `1.0`
- `--exclude_st` / `--include_st`：是否排除 ST 股票，默认排除
- `--hold_days`：前瞻回测持有天数，默认 `5`
- `--max_positions`：组合最大持仓数，默认 `10`
- `--max_positions_per_industry`：单一行业最大持仓数，默认 `2`
- `--backtest_start_date` / `--backtest_end_date`：启用滚动前瞻回测的日期区间
- `--entry_threshold`：日线首信号阈值，默认 `0.18`
- `--resonance_threshold`：多周期共振阈值，默认 `0.22`
- `--resonance_min_count`：至少多少个周期同时支持，默认 `2`
- `--gate_index`：是否启用指数 regime 过滤

这一组脚本把“特征计算 / 评估 / CLI”拆开，针对 A 股日线数据做多周期共振扫描，并直接输出扫描日可见信息下的全市场候选股列表。

## 模块结构

- `multitimeframe_feature_engine.py`
  - 负责日 / 周 / 月 K 线聚合
  - 负责物理态特征计算：energy、temperature、order、phase、switch、score
- `multitimeframe_evaluation.py`
  - 负责单周期首信号评估
  - 负责多周期共振信号构建与首信号评估
- `multitimeframe_physics_utils.py`
  - 负责 entropy、hurst、rolling z-score、指数月度 regime 等底层工具
- `run_multitimeframe_resonance_scan.py`
  - CLI 入口
  - 只负责参数解析和调用 service
- `multitimeframe_scan_service.py`
  - 负责全股票池收益排序和批量扫描编排
- `multitimeframe_report_writer.py`
  - 负责结果聚合和 CSV 落盘

## 输入数据

- 个股日线目录：`gp-data/tushare-daily-full`
- 股票名称映射：`gp-data/tushare_stock_basic.csv`
- 可选指数数据：通过 `--index_path` 传入，用于 `--gate_index` 开启后做大盘 regime 过滤

个股 CSV 至少需要这些列：

- `trade_date`
- `open`
- `close`
- `amount`
- `turnover_rate`
- `net_mf_amount`

## 正式全量扫描

在 `gp-quant` 根目录运行：

```bash
/root/miniforge3/bin/conda run -p /root/miniforge3 --no-capture-output python \
  /root/.vscode-server/extensions/ms-python.python-2026.2.0-linux-x64/python_files/get_output_via_markers.py \
  src/strategy/multitimeframe/run_multitimeframe_resonance_scan.py \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --out_dir /nvme5/xtang/gp-workspace/gp-quant/results/multitimeframe_resonance/live_market_scan \
  --scan_date 20260309 \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv
```

当前正式扫描结果默认建议放在：`results/multitimeframe_resonance/`，避免代码目录混入产物文件。若不传 `--scan_date`，程序会自动推断数据中的最新交易日。

也可以直接使用 `scripts/` 目录下的一键脚本：`./scripts/run_multitimeframe_resonance_scan.sh`

```bash
./scripts/run_multitimeframe_resonance_scan.sh

# 指定扫描日
./scripts/run_multitimeframe_resonance_scan.sh --scan-date 20260309

# 仅跑小样本
./scripts/run_multitimeframe_resonance_scan.sh --symbols sh600000,sz000001 --out-dir /tmp/gp_quant_resonance_smoke
```

## 小样本验证

```bash
/root/miniforge3/bin/conda run -p /root/miniforge3 --no-capture-output python \
  /root/.vscode-server/extensions/ms-python.python-2026.2.0-linux-x64/python_files/get_output_via_markers.py \
  src/strategy/multitimeframe/run_multitimeframe_resonance_scan.py \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --out_dir /tmp/gp_quant_resonance_smoke \
  --scan_date 20260309 \
  --symbols sh600000,sz000001,sh601398 \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv
```

## 输出文件

- `market_scan_snapshot_<scan_date>.csv`
  - 扫描日全市场状态快照
- `resonance_candidates_<scan_date>_all.csv`
  - 扫描日全部共振候选股
- `resonance_candidates_<scan_date>_top<top_n>.csv`
  - 扫描日共振候选前 N
- `selected_portfolio_<scan_date>_top<top_n>.csv`
  - 叠加流动性 / ST / 组合容量 / 行业上限约束后的最终入选名单
- `resonance_summary_<scan_date>.csv`
  - 扫描统计汇总
- `forward_backtest_daily_<scan_date>.csv`
  - 按扫描日滚动生成的每日候选 / 入选统计，同时包含 `nav` 与 `strategy_daily_return`
- `forward_backtest_trades_<scan_date>.csv`
  - 前瞻回测逐笔交易结果
- `forward_backtest_summary_<scan_date>.csv`
  - 前瞻回测摘要指标

扫描 / 候选 / 入选 / 回测结果里会一并保留：`industry`、`market`、`area` 字段，供 Web 页面或后续二次分析直接使用。

## Web 浏览

可以直接启动新的 dashboard 来浏览上述 CSV，并在页面里手动调整成交额、换手率、ST、分数和支撑数过滤：

```bash
/root/miniforge3/bin/conda run -p /root/miniforge3 --no-capture-output python \
  /root/.vscode-server/extensions/ms-python.python-2026.2.0-linux-x64/python_files/get_output_via_markers.py \
  web/app.py \
  --port 5050
```
