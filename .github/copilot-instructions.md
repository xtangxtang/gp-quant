# Project Guidelines — gp-quant

A-股量化研究平台：复杂系统理论驱动的多策略选股、回测与数据管道。

## Architecture

```
src/
├── agents/          # 数据管道 Agent (supervisor → 6 agents DAG 调度)
├── core/            # 共享模块: tick_entropy (置换熵/路径不可逆/主特征值)
├── downloader/      # 数据源: Tushare (日线/1分钟/财务), Xueqiu (Wiki 爬取)
└── strategy/        # 策略模块 (每个策略 5 文件模板，见下)
docs/                # 理论基础 (复杂性理论笔记, 12 篇论文综述, 研报分析)
wiki/                # 项目知识库 (Karpathy LLM-Wiki: concepts/entities/experiments/decisions)
web/                 # Flask 可视化面板 (自动发现策略, port 5050)
scripts/             # Shell 入口脚本
results/             # 策略输出 (CSV + JSON)
```

- 数据根目录: `/nvme5/xtang/gp-workspace/gp-data/`
- 每只股票一个 CSV: `tushare-daily-full/{symbol}.csv`
- 衍生数据: `tushare-weekly-5d/`, `tushare-1m-free/`, `tushare-extended/`
- Supervisor DAG: `stock_list → daily_financial/market_data → minute → derived → market_trend → entropy_scan`
  详见 [src/agents/supervisor.py](src/agents/supervisor.py)

## Strategy Module Pattern

每个策略目录 `src/strategy/{name}/` 遵循 5 文件模板:

| File | Purpose |
|------|---------|
| `feature_engine.py` | 计算因子 (熵、Hurst、波动率、资金流等) |
| `signal_detector.py` / `market_regime.py` | 状态机 / 信号逻辑 |
| `scan_service.py` | 编排: 加载数据 → 计算因子 → 检测信号 → 过滤 |
| `backtest_fast.py` | 前向滚动回测引擎 |
| `run_{name}.py` | CLI 入口 (argparse), 搭配 `scripts/run_{name}.sh` |

添加新策略时复制此结构。`web/app.py` 会自动发现含 `README.md` 的策略目录。

> 部分策略用 `config.py` 替代散落的常量、或有嵌套子目录，这是允许的变体。

## Build and Test

```bash
# 环境
source .venv/bin/activate          # Python 3.12, conda-based

# 安装依赖
pip install -r requirements.txt

# 数据同步 (推荐用 supervisor 统一调度)
./scripts/run_agent_supervisor.sh

# 运行策略扫描 (示例)
python -m src.strategy.entropy_accumulation_breakout.run_entropy_accumulation_breakout \
    --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
    --feature_cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
    --scan_date 20260414

# 回测
python -m src.strategy.entropy_accumulation_breakout.run_entropy_accumulation_breakout \
    --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
    --feature_cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
    --backtest_start_date 20250101 --backtest_end_date 20250630

# Web 面板
python web/app.py --port 5050
```

测试为诊断脚本 (`tests/`)，无 pytest 组织。

## Conventions

- **数据格式**: 纯 CSV，无数据库。日期列 `trade_date` 格式 `YYYYMMDD` (字符串，非 datetime)。
- **结果输出**: `results/{strategy_name}/scan_{date}.csv` + `backtest_*_summary.json`。
- **CLI 标准参数**: `--data_dir`, `--scan_date`, `--top_n`, `--backtest_start_date`, `--backtest_end_date`。
- **模块运行**: 用 `python -m src.strategy.{name}.run_{name}` 而非直接执行脚本。
- **环境变量**: `GP_DATA_DIR` (数据根), `TUSHARE_TOKEN` / `GP_TUSHARE_TOKEN`, `DASHSCOPE_API_KEY` (LLM)。
- **Agent 状态持久化**: `.agent_*_state.json` (JSON 文件，supervisor 依赖图管理)。
- **语言**: 代码用英文变量名 + 中文注释 / docstring 均可，README 用中文。
- **CSV 写入**: 结构化写盘统一用 `src.downloader.csv_utils.write_rows_csv()`，避免手动 DictWriter 导致列序不稳定。

## Import & Data Loading

```python
# 策略内部 — 相对导入
from .feature_engine import build_features

# 跨模块 — 绝对导入
from src.core.tick_entropy import permutation_entropy
from src.downloader.csv_utils import write_rows_csv
```

数据加载统一模式 — 文件缺失或损坏时返回 `None`，调用方检查后跳过:

```python
def load_daily(data_dir, symbol):
    fpath = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath)
    if "trade_date" not in df.columns:
        return None
    return df.sort_values("trade_date").reset_index(drop=True)
```

## Pitfalls

- `trade_date` 是字符串 `YYYYMMDD`，不要 `pd.to_datetime()` 后做比较。
- 策略 `@dataclass` 配置的 list 字段必须用 `field(default_factory=list)`。
- 诊断脚本在 `tests/`，但不是 pytest — 无断言，纯运行+打印/存 CSV。

## Wiki Maintenance

项目知识库 `wiki/` 采用 Karpathy LLM-Wiki 三层架构: `concepts/`, `entities/`, `experiments/`, `decisions/`, `sources/`。
结构定义见 [wiki/WIKI.md](wiki/WIKI.md)，索引见 [wiki/index.md](wiki/index.md)。
跨页面引用用 `[[wikilink]]` 语法，新增内容须同步更新 `index.md`。

## Key References

- 理论基础: [docs/complexity_theory_notes.md](docs/complexity_theory_notes.md)
- 12 篇论文综述: [docs/complexity_research_12_papers_conclusion.md](docs/complexity_research_12_papers_conclusion.md)
- Core 熵模块文档: [src/core/TICK_ENTROPY_MODULE_README.md](src/core/TICK_ENTROPY_MODULE_README.md)
- Supervisor DAG: [src/agents/supervisor.py](src/agents/supervisor.py)
- 各策略 README: `src/strategy/{name}/README.md`
- 已有 Skills (13 个): `.github/skills/*/SKILL.md`
- 已有 Agents (3 个): `.github/agents/*.agent.md`
- 特征缓存: `/nvme5/xtang/gp-workspace/gp-data/feature-cache/` (熵惜售策略增量计算缓存)
