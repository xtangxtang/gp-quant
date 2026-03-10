# Multi-Timeframe Resonance Analysis

这一组脚本把“特征计算 / 评估 / CLI”拆开，针对 A 股日线数据做多周期共振扫描。

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
  --out_dir /nvme5/xtang/gp-workspace/gp-quant/results/multitimeframe_resonance/out_2025_multitimeframe_fullscan \
  --test_year 2025 \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv
```

当前正式扫描结果默认建议放在：`results/multitimeframe_resonance/`，避免代码目录混入产物文件。

也可以直接使用 `scripts/` 目录下的一键脚本：`./scripts/run_multitimeframe_resonance_scan.sh`

```bash
./scripts/run_multitimeframe_resonance_scan.sh

# 指定年份
./scripts/run_multitimeframe_resonance_scan.sh --test-year 2024

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
  --test_year 2025 \
  --symbols sh600000,sz000001,sh601398 \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv
```

## 主要参数

- `--top_n`：从全市场年内涨幅排序后，取前 N 只进入详细扫描，默认 `300`
- `--entry_threshold`：日线首信号阈值，默认 `0.18`
- `--resonance_threshold`：多周期共振阈值，默认 `0.22`
- `--resonance_min_count`：至少多少个周期同时支持，默认 `3`
- `--gate_index`：是否启用指数 regime 过滤

## 输出文件

- `bull_stocks_<year>_all.csv`
  - 全股票池年内涨幅排序结果
- `bull_stocks_<year>_top<top_n>.csv`
  - 进入详细扫描的股票列表
- `multitimeframe_entry_eval_<year>.csv`
  - 日 / 周 / 月首信号评估明细
- `multitimeframe_entry_eval_<year>_agg.csv`
  - 日 / 周 / 月评估汇总
- `multitimeframe_resonance_eval_<year>.csv`
  - 多周期共振首信号评估明细
- `multitimeframe_resonance_eval_<year>_agg.csv`
  - 多周期共振汇总
- `multitimeframe_resonance_signals_<year>.csv`
  - 年内所有共振信号明细
