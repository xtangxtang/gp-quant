# gp-quant

这是一个用于抓取 A 股自选股每日成交明细的 Python 脚本。

当前数据源为东方财富的 1 分钟行情（分钟线），脚本会将其转换为“tick-like”的 CSV 结构（每分钟一行），用于后续分析流程兼容。

分钟数据默认使用**前复权（qfq）**价格，如需不复权/后复权可通过参数 `--adj none` / `--adj hfq` 指定（wrapper 中同样支持 `-a/--adj`）。

> 重要限制：东方财富的分钟历史接口（`trends2/get`）**只支持最近约 5 个交易日**的分钟数据。
> 
> - 当你请求的 `date` 不在最近 5 个交易日范围内时，脚本会明确报错并记录失败任务，避免把“最新一天”的分钟数据写入到历史日期文件里。
> - 如果你需要更长周期的历史行情，请考虑改用日线（`klt=101`）或其他数据源。

> 复权说明：分钟数据的 qfq/hfq 是通过“日线复权因子”实现的：对目标日期取日线 `close_adj / close_raw` 作为缩放因子，将分钟的价格与成交额做同倍率缩放；成交量保持不变。

## 功能特性

1. **多线程并发抓取**：支持多线程同时下载多只股票的数据，提高下载效率。
2. **自定义股票列表**：通过 `self_gplist.json` 文件灵活配置需要抓取的股票代码。
3. **历史数据回溯**：支持通过命令行参数指定日期范围，自动下载历史数据。
4. **智能节假日过滤**：内置中国法定节假日和周末过滤逻辑，只在真正的交易日进行数据抓取。
5. **反爬虫与断点续传**：
    - 每个任务会加入 2～5 秒的随机延迟，降低请求节奏。
    - 当前实现不做“指数退避自动重试”（避免在被限流时持续打点）。
   - 如果某天的数据彻底下载失败或程序被意外中断，失败的任务会被记录到 `failed_tasks.json` 中。
    - 自选股模式下（`src/get_selflist_daily.py`），下次启动脚本时会自动读取并将这些失败任务追加到任务列表中重试。
    - 全市场模式下（`src/get_total_daily.py`），同样会记录 `failed_tasks.json`，但当前入口脚本不会自动读取该文件（可按需自行重跑日期范围）。
6. **自定义存储目录**：支持将下载的数据保存到指定的文件夹中。

> 说明：为避免起始日期早于上市日期导致大量无效请求，自选股/全市场模式都会缓存上市日期并跳过上市日前的任务（自选股缓存文件 `self_listing_dates.json`，全市场缓存文件 `total_listing_dates.json`）。如需关闭可加参数 `--no_ipo_filter`（wrapper 中为 `--no-ipo-filter`）。

## 环境依赖

请确保已安装 Python 3，并安装以下依赖库：

```bash
pip install -r requirements.txt
```

## 配置文件

`self_gplist.json` 的读取规则如下：

- 如果运行时显式传了 `--output_dir`，则从该目录读取/创建 `self_gplist.json`
- 如果没有传 `--output_dir`，则从当前运行目录读取/创建 `self_gplist.json`

文件内容填入你需要抓取的股票代码（带 `sh` 或 `sz` 前缀）：

```json
[
    "sz002409",
    "sz301323",
    "sh688114",
    "sh688508"
]
```
*(如果该文件不存在，脚本首次运行时会自动生成一个包含默认股票的示例文件。)*

## 启动方式

### 1. 下载当天的最新数据（默认行为）
如果不带任何参数运行，脚本会自动判断今天是否为交易日，如果是，则下载今天的数据。
```bash
python src/get_selflist_daily.py
```

### 2. 下载指定日期范围内的数据
使用 `--start_date` 和 `--end_date` 参数（格式为 `YYYY-MM-DD`）。脚本会自动跳过期间的周末和法定节假日。
```bash
python src/get_selflist_daily.py --start_date 2023-08-01 --end_date 2023-08-05
```

### 3. 从指定日期一直下载到今天
如果只提供 `--start_date`，脚本会默认将结束日期设置为今天。
```bash
python src/get_selflist_daily.py --start_date 2023-08-01
```

### 4. 指定数据保存目录
使用 `--output_dir` 参数可以将数据保存到指定路径（默认为当前目录下的 `gp_daily`）。
```bash
python src/get_selflist_daily.py --start_date 2023-08-01 --output_dir my_stock_data
```

### 5. 下载全市场股票（Total）
使用 `run_get_total.sh` 可以自动获取截止上个交易日的全市场股票列表（保存为 `total_gplist.json`），并按日期范围下载。

`total_gplist.json` 的保存位置规则与 `self_gplist.json` 一致：显式传了 `--output_dir` 则保存在该目录；否则保存在当前运行目录。

为避免“起始日期早于上市日期”的股票产生大量无效请求，全市场模式会自动拉取并缓存各股票的上市日期（文件 `total_listing_dates.json`），并跳过上市日前的任务。如需关闭该行为，可加参数 `--no_ipo_filter`。
```bash
./run_get_total.sh -s 2023-08-01 -o /tmp/my_total_data
```

## 输出目录结构

下载的数据会按照股票代码进行分类，每天的数据保存为一个独立的 CSV 文件。

假设你运行了 `python src/get_selflist_daily.py --output_dir my_data`，目录结构如下（注意：此时 `self_gplist.json` 会读取/创建在 `my_data` 目录内）：

```text
当前工作目录/
│
└── my_data/                  # 你指定的输出目录 (默认是 gp_daily)
    │
    ├── self_gplist.json      # 你的自选股配置文件（显式传了 --output_dir 时放这里）
    ├── failed_tasks.json     # (如果发生错误) 记录下载失败任务的文件，下次启动会自动读取
    │
    ├── sz002409/             # 以股票代码命名的子文件夹
    │   ├── 2023-08-01.csv    # 该股票当天的分钟级“tick-like”数据
    │   └── 2023-08-02.csv
    │
    └── sh688114/             # 另一只股票的子文件夹
        ├── 2023-08-01.csv
        └── 2023-08-02.csv
```

### CSV 数据格式说明
每个 CSV 文件包含以下列（与东方财富分钟线含义一致）：
- 时间 (例如: 2026-02-13 09:31)
- 开盘
- 收盘
- 最高
- 最低
- 成交量(手)
- 成交额(元)
- 均价
