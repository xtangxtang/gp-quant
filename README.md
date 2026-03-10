# gp-quant

`gp-quant` 当前是一套面向 A 股的量化研究工作区，核心分成两部分：

- `src/downloader/`：基于 Tushare 拉取全市场日线、股票列表和扩展基本面数据
- `src/analysis/`：基于日 / 周 / 月多周期共振的物理风格选股与评估

旧的 `src/strategy/` 相变策略已经移除，当前仓库的正式分析主线是 `src/analysis/` 下的 multi-timeframe resonance 扫描器。

## 当前目录

```text
gp-quant/
├── scripts/
│   ├── run_get_tushare_all.sh
│   ├── run_get_tushare_daily_full.sh
│   ├── run_get_tushare_extended.sh
│   └── run_multitimeframe_resonance_scan.sh
├── src/
│   ├── analysis/
│   ├── downloader/
│   └── web/
├── results/
│   └── multitimeframe_resonance/
├── complexity_theory_notes.md
├── requirements.txt
└── README.md
```

## 核心思路

当前分析逻辑来自 `complexity_theory_notes.md` 的复杂系统 / 经济物理学视角，用日线先识别局部有序化，再用周线、月线做更高层级的共振确认。

特征上主要对应为：

- `energy`：资金流和成交额强度
- `temperature`：换手与波动的相对状态
- `order`：均线结构、突破状态
- `phase`：熵、Hurst 指数等秩序变化
- `switch`：从混沌向有序切换的触发项

最终输出的核心是“扫描日市场快照 + 当日共振候选股列表”：

- 市场快照：对全市场逐只股票给出扫描日当天的日 / 周 / 月状态与共振分数
- 候选股列表：仅保留扫描日当天满足多周期共振条件的股票，并按共振强度排序

## 环境依赖

建议使用现有 conda 环境，或至少保证 Python 3.12 左右可用。

安装依赖：

```bash
pip install -r requirements.txt
```

如果需要拉取 Tushare 数据，需要准备有效的 Tushare token。当前下载脚本支持显式用 `--token` 传入。

## 数据下载

### 1. 一键拉取完整 Tushare 数据

会依次完成：

- 获取当前股票列表
- 下载全市场日线历史到 `tushare-daily-full/`
- 下载扩展数据，例如复权因子、财务、分红、停复牌等

```bash
./scripts/run_get_tushare_all.sh -o /nvme5/xtang/gp-workspace/gp-data --token <your_tushare_token>
```

### 2. 只下载全市场日线

```bash
./scripts/run_get_tushare_daily_full.sh -o /nvme5/xtang/gp-workspace/gp-data --token <your_tushare_token>
```

常见补充参数：

- `--start-date 20200101`
- `--end-date 20251231`
- `--symbols sh600000,sz000001`
- `--list-file tushare_gplist.json`

### 3. 只下载扩展数据

```bash
./scripts/run_get_tushare_extended.sh -o /nvme5/xtang/gp-workspace/gp-data --token <your_tushare_token>
```

默认会下载：

- `trade_cal`
- `adj_factor`
- `stk_limit`
- `suspend_d`
- `income`
- `balancesheet`
- `cashflow`
- `fina_indicator`
- `dividend`

## 输入数据约定

多周期共振扫描默认使用：

- 日线目录：`/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full`
- 股票名称映射：`/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv`

个股 CSV 至少应包含这些列：

- `trade_date`
- `open`
- `close`
- `amount`
- `turnover_rate`
- `net_mf_amount`

## 运行正式扫描

推荐直接使用 `scripts/` 里的入口脚本：

```bash
./scripts/run_multitimeframe_resonance_scan.sh
```

默认行为：

- 自动推断数据中的最新交易日作为 `scan_date`
- 输出目录：`results/multitimeframe_resonance/live_market_scan`
- 对全市场所有股票直接做扫描日可见信息下的多周期共振排序
- 默认推荐参数：`top_n=30`、`min_amount=500000`、`min_turnover=1.0`、`resonance_min_count=2`、`hold_days=5`、`max_positions=10`、`max_positions_per_industry=2`

常见用法：

```bash
# 指定扫描日
./scripts/run_multitimeframe_resonance_scan.sh --scan-date 20260309

# 追加流动性 / ST / 组合层约束
./scripts/run_multitimeframe_resonance_scan.sh \
    --scan-date 20260309 \
    --min-amount 500000 \
    --min-turnover 1.0 \
    --hold-days 5 \
    --max-positions 10 \
    --max-positions-per-industry 2 \
    --resonance_min_count 2

# 做一个短窗口前瞻回测
./scripts/run_multitimeframe_resonance_scan.sh \
    --backtest-start-date 20260201 \
    --backtest-end-date 20260309 \
    --hold-days 5 \
    --max-positions 10

# 小样本 smoke test
./scripts/run_multitimeframe_resonance_scan.sh \
    --symbols sh600000,sz000001 \
    --out-dir /tmp/gp_quant_resonance_smoke

# 指定自定义数据目录
./scripts/run_multitimeframe_resonance_scan.sh \
    --data-dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
    --basic-path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv
```

如果你要直接运行 Python 入口：

```bash
python src/analysis/run_multitimeframe_resonance_scan.py \
    --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
    --out_dir /tmp/gp_quant_resonance_smoke \
    --scan_date 20260309 \
    --symbols sh600000,sz000001,sh601398 \
    --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv
```

## 扫描输出

正式结果会落到：

- `results/multitimeframe_resonance/`

常见输出文件：

- `market_scan_snapshot_<scan_date>.csv`
- `resonance_candidates_<scan_date>_all.csv`
- `resonance_candidates_<scan_date>_top<top_n>.csv`
- `selected_portfolio_<scan_date>_top<top_n>.csv`
- `resonance_summary_<scan_date>.csv`
- `forward_backtest_daily_<scan_date>.csv`
- `forward_backtest_trades_<scan_date>.csv`
- `forward_backtest_summary_<scan_date>.csv`

含义大致为：

- `market_scan_snapshot_*`：全市场扫描日状态快照，包含日 / 周 / 月与共振分数
- `resonance_candidates_*_all`：扫描日满足共振条件的全部候选股
- `resonance_candidates_*_topN`：按共振强度排序后的前 N 只候选股
- `selected_portfolio_*_topN`：叠加流动性 / ST / 组合容量约束后的最终入选名单
- `max_positions_per_industry > 0` 时，会额外限制单一行业入选数量，降低主题扎堆
- `resonance_summary_*`：扫描日统计汇总
- `forward_backtest_daily_*`：按扫描日滚动生成的每日选股数量统计
- `forward_backtest_trades_*`：前瞻回测逐笔交易结果
- `forward_backtest_summary_*`：前瞻回测摘要指标

现在扫描和回测结果里还会携带：

- `industry` / `market` / `area`：来自 `tushare_stock_basic.csv` 的行业、板块与地域信息
- `nav` / `strategy_daily_return`：前瞻回测日度净值与当日组合收益

## analysis 模块说明

`src/analysis/` 目前已经拆成较清晰的职责边界：

- `multitimeframe_feature_engine.py`：日 / 周 / 月 K 线聚合与物理态特征计算
- `multitimeframe_evaluation.py`：单周期与共振信号评估
- `multitimeframe_physics_utils.py`：熵、Hurst、z-score、指数 regime 工具
- `multitimeframe_scan_service.py`：全市场排序和扫描编排
- `multitimeframe_report_writer.py`：结果 CSV 落盘
- `run_multitimeframe_resonance_scan.py`：CLI 入口

更细的使用说明见 [src/analysis/README.md](/nvme5/xtang/gp-workspace/gp-quant/src/analysis/README.md)。

## Web 模块

`src/web/` 现在提供一个轻量 dashboard，用来直接浏览扫描 CSV、手动加过滤条件，以及联动查看前瞻回测摘要和净值曲线。页面支持：

- 按行业 / 市场 / 地域过滤
- 按成交额 / 换手率 / 共振分 / 支撑数过滤
- 查看入选组合、候选池、全市场快照和逐笔交易

当前正式研究与生产输出，默认都走：

- 数据下载：`scripts/run_get_tushare_*.sh`
- 分析扫描：`scripts/run_multitimeframe_resonance_scan.sh`

启动 Web：

```bash
/root/miniforge3/bin/conda run -p /root/miniforge3 --no-capture-output python \
    /root/.vscode-server/extensions/ms-python.python-2026.2.0-linux-x64/python_files/get_output_via_markers.py \
    src/web/app.py \
    --scan-output-dir /nvme5/xtang/gp-workspace/gp-quant/results/multitimeframe_resonance/live_market_scan \
    --port 5050
```

## 建议工作流

```bash
# 1. 更新数据
./scripts/run_get_tushare_all.sh -o /nvme5/xtang/gp-workspace/gp-data --token <your_tushare_token>

# 2. 小样本验证
./scripts/run_multitimeframe_resonance_scan.sh \
    --symbols sh600000,sz000001 \
    --out-dir /tmp/gp_quant_resonance_smoke

# 3. 全量正式扫描
./scripts/run_multitimeframe_resonance_scan.sh
```

## 备注

- 根目录旧策略脚本和 `src/strategy/` 已经移除，不再维护
- 输出文件已经从代码目录分离到 `results/` 下，便于版本管理
- 如果你只想理解策略本身，优先阅读：`complexity_theory_notes.md` 和 `src/analysis/README.md`
