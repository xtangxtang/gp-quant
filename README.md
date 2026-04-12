# gp-quant

`gp-quant` 当前是一套面向 A 股的量化研究工作区，核心分成两部分：

- `src/downloader/`：基于 Tushare 拉取全市场日线、股票列表和扩展基本面数据
- `src/strategy/multitimeframe/`：基于日 / 周 / 月多周期共振的物理风格选股与评估

当前仓库的正式策略主线是 `src/strategy/multitimeframe/` 下的 multi-timeframe resonance 扫描器。

## 当前目录

```text
gp-quant/
├── scripts/
│   ├── run_get_tushare_all.sh
│   ├── run_get_tushare_daily_full.sh
│   ├── run_get_tushare_extended.sh
│   └── run_multitimeframe_resonance_scan.sh
├── src/
│   ├── strategy/
│   │   └── multitimeframe/
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

### 2.5. 增量同步 A 股 1 分钟数据

默认落盘到：`gp-data/trade/<symbol>/<YYYY-MM-DD>.csv`

默认行为：

- 优先使用 Tushare 分钟源
- 自动跳过已经完整落盘的日文件
- 未显式指定日期时，只同步最近 `3` 个交易日，便于日常增量更新和补漏
- 默认使用原始分钟价 `fqt=0`，避免在缺少额外复权辅助模块时出现语义偏差
- 现在会自动读取 `failed_tasks.json`，优先续跑历史失败任务，并在初始轮结束后对失败项做多轮自动续跑
- 当主源是免费 `tx` 时，失败续跑轮次会在 Tencent 和 Tushare 之间自动切换主源

```bash
./scripts/run_sync_a_share_1m.sh -o /nvme5/xtang/gp-workspace/gp-data
```

常见用法：

```bash
# 只更新最近 1 个交易日
./scripts/run_sync_a_share_1m.sh \
    -o /nvme5/xtang/gp-workspace/gp-data \
    --recent-open-days 1

# 只续跑失败队列，适合免费网页源偶发网络抖动后的补漏
./scripts/run_sync_a_share_1m.sh \
    -o /nvme5/xtang/gp-workspace/gp-data \
    --recent-open-days 3 \
    --retry-failed-only

# 初始轮先走 Tushare，失败续跑轮次自动切到 Tencent
./scripts/run_sync_a_share_1m.sh \
    -o /nvme5/xtang/gp-workspace/gp-data \
    --recent-open-days 3 \
    --threads 3

# 只拉少量股票做 smoke test
./scripts/run_sync_a_share_1m.sh \
    -o /nvme5/xtang/gp-workspace/gp-data \
    --symbols sh600000,sz000001 \
    --recent-open-days 1

# 如果你有 Tushare 分钟权限，可以切到 ts 源
./scripts/run_sync_a_share_1m.sh \
    -o /nvme5/xtang/gp-workspace/gp-data \
    --source ts \
    --token <your_tushare_token> \
    --start-date 20260327 \
    --end-date 20260330
```

限制说明：

- 免费 Tencent 分钟接口通常只覆盖最近几个交易日，不适合补很久以前的 1 分钟历史
- 如果要做更长历史区间回补，通常需要 `--source ts` 且你的 Tushare 账户具备分钟权限
- 免费网页源批量抓取时仍可能出现代理或限流抖动；现在脚本会把失败项写入 `failed_tasks.json` 并自动续跑，但不能保证一次性全量成功

### 2.6. 交易日 16:00 自动定时更新

如果你希望在每个交易日下午 `4:00` 自动更新全量增量数据，可以直接运行：

```bash
./scripts/run_eod_data_scheduler.sh -o /nvme5/xtang/gp-workspace/gp-data
```

默认行为：

- 每个交易日本地时间 `16:00` 触发
- 先运行 `fast_sync_tushare_latest.py`，增量更新股票列表、交易日历、日线、复权、停复牌、分红和财务数据
- 再运行当天的 `1` 分钟同步，只抓当前交易日，并默认使用 `Tencent` 作为主源
- 同一天只会成功跑一次；如果当天任务失败，达到冷却时间后会再次尝试
- 调度器会在 `gp-data/.eod_data_scheduler_state.json` 记录状态，并在 `gp-data/.eod_data_scheduler.lock` 上锁，避免重复启动
- 调度器 stdout 会持续打印当前状态：启动时已加载状态、空闲等待心跳、`fast_sync` / `minute_sync` 分阶段进度，以及最终 `all_done=yes/no`

常见用法：

```bash
# 立即执行一次并退出，适合 smoke test
./scripts/run_eod_data_scheduler.sh \
    -o /nvme5/xtang/gp-workspace/gp-data \
    --run-now \
    --run-once

# 只打印会执行哪些命令，不真正下载
./scripts/run_eod_data_scheduler.sh \
    -o /nvme5/xtang/gp-workspace/gp-data \
    --run-now \
    --run-once \
    --dry-run

# 后台常驻运行
nohup ./scripts/run_eod_data_scheduler.sh \
    -o /nvme5/xtang/gp-workspace/gp-data \
    > /tmp/gp_quant_eod_scheduler.log 2>&1 &

# 把空闲状态心跳调成每 60 秒打印一次
./scripts/run_eod_data_scheduler.sh \
    -o /nvme5/xtang/gp-workspace/gp-data \
    --status-log-interval-seconds 60
```

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
python src/strategy/multitimeframe/run_multitimeframe_resonance_scan.py \
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

## Strategy 模块说明

`src/strategy/multitimeframe/` 目前已经拆成较清晰的职责边界：

- `multitimeframe_feature_engine.py`：日 / 周 / 月 K 线聚合与物理态特征计算
- `multitimeframe_evaluation.py`：单周期与共振信号评估
- `multitimeframe_physics_utils.py`：熵、Hurst、z-score、指数 regime 工具
- `multitimeframe_scan_service.py`：全市场排序和扫描编排
- `multitimeframe_report_writer.py`：结果 CSV 落盘
- `run_multitimeframe_resonance_scan.py`：CLI 入口

更细的使用说明见 [src/strategy/multitimeframe/README.md](/nvme5/xtang/gp-workspace/gp-quant/src/strategy/multitimeframe/README.md)。

## Web 模块

`web/` 现在提供一个策略控制台：首页会自动发现 `src/strategy` 下的策略，展示 README 中的一句话概括；进入策略页后可查看 README 的“描述”和“主要参数”，填写参数并直接应用策略执行选股。页面支持：

- 首页按策略目录自动展示策略卡片
- 策略页展示 README 的“描述”和“主要参数”
- 根据 CLI 参数自动生成表单并直接执行策略
- 展示入选股票、候选池、摘要指标和执行日志

当前正式研究与生产输出，默认都走：

- 数据下载：`scripts/run_get_tushare_*.sh`
- 分析扫描：`scripts/run_multitimeframe_resonance_scan.sh`

启动 Web：

```bash
/root/miniforge3/bin/conda run -p /root/miniforge3 --no-capture-output python \
    /root/.vscode-server/extensions/ms-python.python-2026.2.0-linux-x64/python_files/get_output_via_markers.py \
    web/app.py \
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

- 多周期策略代码目前统一收敛在 `src/strategy/multitimeframe/`
- 输出文件已经从代码目录分离到 `results/` 下，便于版本管理
- 如果你只想理解策略本身，优先阅读：`complexity_theory_notes.md` 和 `src/strategy/multitimeframe/README.md`
