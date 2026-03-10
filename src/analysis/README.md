# Multi-Timeframe Resonance Analysis

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
  src/analysis/run_multitimeframe_resonance_scan.py \
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
  src/analysis/run_multitimeframe_resonance_scan.py \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --out_dir /tmp/gp_quant_resonance_smoke \
  --scan_date 20260309 \
  --symbols sh600000,sz000001,sh601398 \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv
```

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
  src/web/app.py \
  --scan-output-dir /nvme5/xtang/gp-workspace/gp-quant/results/multitimeframe_resonance/live_market_scan \
  --port 5050
```
