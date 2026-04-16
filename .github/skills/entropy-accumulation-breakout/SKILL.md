---
name: entropy-accumulation-breakout
description: "运行熵惜售分岔突破策略（三阶段状态机：惜售吸筹→分岔突破→结构崩塌）。Use when: 熵惜售, 惜售吸筹, 分岔突破, entropy accumulation, breakout, 筹码集中, 临界减速, critical slowing, 路径不可逆, path irreversibility, 主特征值, dominant eigenvalue, 结构崩塌, collapse, 三阶段, 状态机, 熵突破扫描, 熵选股, 资金流, 分钟线, 周线, 多源, 特征缓存。"
argument-hint: "扫描日期或股票代码，例如：20260410 或 sh600000"
---

# 熵惜售分岔突破策略 (Entropy-Accumulation-Breakout)

基于信息熵、路径不可逆性、临界减速理论的三阶段交易系统，融合日线+周线+分钟线+资金流多源数据：

```
惜售吸筹 (Accumulation)  →  分岔突破 (Bifurcation)  →  结构崩塌 (Collapse)
     ↓                          ↓                          ↓
   买入准备                     买入执行                     卖出退出
```

## 核心逻辑

### 阶段1: 惜售吸筹 (Accumulation)
- 置换熵 < 0.65（筹码集中、交易有序化）
- 路径不可逆性 > 0.05（定向资金力量）
- 资金流: 大单净额累计为正 + 大单买散户卖不平衡度 > 0.3
- 需连续 ≥ 5 天满足条件
- 质量评分 AQ = 25%熵 + 20%不可逆 + 15%纯度 + 15%大单连续天数 + 15%资金流不平衡 + 10%大单净额

### 阶段2: 分岔突破 (Bifurcation)
- 主特征值 > 0.85（临界减速 → 即将突破）
- 量脉冲 > 1.8×（成交量暴增）
- 突破时熵仍 < 0.75（有序突破，非混乱暴涨）
- 资金流: 大单动量为正（大单加速流入）
- 周线确认: 周线置换熵 < 0.75 + 周线大单净额累计为正
- 分钟确认: 日内置换熵 < 0.70（日内价格有序）
- 质量评分 BQ = 20%特征值 + 20%量脉冲 + 15%退相干速率 + 15%大单动量 + 10%熵 + 10%不可逆 + 10%大单净额占比
- 综合得分 = 40% AQ + 60% BQ

### 阶段3: 结构崩塌退出 (Collapse)
- 5个信号中满足3个即退出:
  1. 熵飙升 > 0.90（无序化）
  2. 路径不可逆骤降 < 0.01（主力撤离）
  3. 熵加速 > 0.05（结构加速瓦解）
  4. 量能衰竭（< 峰值×0.3）
  5. 密度矩阵纯度 < 0.3（共识瓦解）
- 硬止损: -10%

## 执行步骤

### 1. 全市场扫描（自动调度 — supervisor 每天17:00运行）

supervisor 会在数据同步完成后自动执行 `entropy_scan` agent，结果写入 `results/entropy_accumulation_breakout/`。

手动运行：

```bash
cd /nvme5/xtang/gp-workspace/gp-quant

python -m src.strategy.entropy_accumulation_breakout.run_entropy_accumulation_breakout \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --out_dir results/entropy_accumulation_breakout \
  --feature_cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv \
  --scan_date 20260416
```

### 2. 指定日期扫描

```bash
python -m src.strategy.entropy_accumulation_breakout.run_entropy_accumulation_breakout \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --out_dir results/entropy_accumulation_breakout \
  --feature_cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --scan_date 20260410 --top_n 30
```

### 3. 扫描特定股票

```bash
python -m src.strategy.entropy_accumulation_breakout.run_entropy_accumulation_breakout \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --out_dir results/entropy_accumulation_breakout \
  --feature_cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --symbols sh600000,sz000001,sh600036
```

### 4. 前瞻回测

```bash
python -m src.strategy.entropy_accumulation_breakout.run_entropy_accumulation_breakout \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --out_dir results/entropy_accumulation_breakout \
  --feature_cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --backtest_start_date 20260101 --backtest_end_date 20260331 \
  --hold_days 5 --max_positions 10
```

### 5. 通过 supervisor 手动触发

```bash
cd /nvme5/xtang/gp-workspace/gp-quant/src/agents
python supervisor.py --data-dir /nvme5/xtang/gp-workspace/gp-data run --agent entropy_scan
```

## 输出文件

结果输出到 `results/entropy_accumulation_breakout/`:

| 文件 | 内容 |
|------|------|
| `market_snapshot_<date>.csv` | 全市场扫描快照（所有股票状态） |
| `breakout_candidates_<date>.csv` | Top N 突破候选（已排序+行业约束） |
| `backtest_trades_<date>.csv` | 回测逐笔交易记录 |
| `backtest_equity_<date>.csv` | 回测每日净值曲线 |
| `backtest_summary_<date>.csv` | 回测绩效汇总 |

## 策略代码

```
src/strategy/entropy_accumulation_breakout/
├── run_entropy_accumulation_breakout.py  # CLI 入口
├── scan_service.py                       # 主服务层（扫描+回测）
├── signal_detector.py                    # 三阶段状态机信号检测
├── feature_engine.py                     # 多源特征工程 (日线+周线+分钟+资金流)
├── feature_cache.py                      # 增量计算缓存 (首次全量→后续只算新行)
├── market_regime.py                      # 大盘状态门控
├── backtest_fast.py                      # 高速回测引擎
└── README.md                             # 完整文档
```

- Supervisor agent: `src/agents/agent_entropy_scan.py`
- Agent 配置: `.github/agents/entropy-accumulation-breakout.agent.md`

## 关键参数

| 参数 | 阈值 | 阶段 | 说明 |
|------|------|------|------|
| `perm_entropy_low` | 0.65 | 惜售 | 排列熵低于此值=有序 |
| `path_irrev_high` | 0.05 | 惜售 | 路径不可逆高于此值=定向力量 |
| `mf_flow_imbalance_min` | 0.3 | 惜售 | 大单买散户卖不平衡阈值 |
| `mf_big_streak_min` | 3 | 惜售 | 大单连续流入最少天数 |
| `accum_min_days` | 5 | 惜售 | 最少连续满足天数 |
| `dom_eig_threshold` | 0.85 | 突破 | 主特征值→1=临界减速 |
| `vol_impulse_threshold` | 1.8 | 突破 | 成交量脉冲倍数 |
| `intraday_entropy_low` | 0.70 | 突破 | 日内置换熵<此值=日内有序确认 |
| `weekly_perm_entropy_max` | 0.75 | 突破 | 周线熵<此值=周线有序确认 |
| `weekly_big_net_positive` | True | 突破 | 周线大单净额累计为正 |
| `perm_entropy_collapse` | 0.90 | 崩塌 | 熵飙升退出阈值 |
| `path_irrev_collapse` | 0.01 | 崩塌 | 不可逆骤降退出 |
| `purity_collapse_max` | 0.3 | 崩塌 | 纯度骤降退出 |
| `max_hold_days` | 20 | 崩塌 | 最大持仓天数 |

## 市场状态门控

策略内置大盘状态检测（基于上证综指），不同市场状态对应不同仓位比例:

| 市场状态 | 仓位比例 | 说明 |
|----------|----------|------|
| DECLINING | 0% | 禁止开仓 |
| DECLINE_ENDED | 30% | 轻仓试探 |
| CONSOLIDATION | 100% | 理想建仓环境 |
| RISING | 80% | 正常建仓 |
| RISE_ENDING | 0% | 禁止新建仓 |

## 数据需求

| 数据源 | 目录 | 用途 | 必需 |
|--------|------|------|------|
| 日线 K 线 | `tushare-daily-full/` | 熵/分岔/量价特征 (500天回溯) | ✅ |
| 预计算周线 | `tushare-weekly-5d/` | PE/PB估值分位 + 周级别资金流确认 | 推荐 |
| 分钟线 | `trade/{symbol}/` | 日内微观结构 (置换熵/成交量集中度) | 推荐 |
| 资金流向 | `tushare-moneyflow/` | 大单/散户分层资金流 (吸筹/派发检测) | 推荐 |
| 指数日线 | `tushare-index-daily/` | 大盘状态门控 | ✅ |
| 股票信息 | `tushare_stock_basic.csv` | ST过滤 + 行业分类 | 推荐 |

`data_root` 从 `data_dir` 父目录自动推断，包含以上子目录。

## 特征缓存

使用 `--feature_cache_dir` 启用增量计算：
- 首次运行: 全量计算 → 缓存到 `{cache_dir}/daily/{symbol}.csv` + `weekly/`
- 后续运行: 只计算新增行 → 追加到缓存，热缓存命中 ~17x 加速
- 缓存目录: `/nvme5/xtang/gp-workspace/gp-data/feature-cache/`

## Supervisor 集成

`entropy_scan` agent 已注册到 supervisor DAG (priority=5)，每天 17:00 自动运行：
```
stock_list → daily_financial/market_data → minute → derived → market_trend → entropy_scan
```
Agent 代码: `src/agents/agent_entropy_scan.py`

## Wiki 参考

- `wiki/entities/entropy-accumulation-breakout.md` — 策略实体页
- `wiki/concepts/entropy.md` — 熵的理论基础
- `wiki/concepts/bifurcation.md` — 分岔理论
- `wiki/concepts/path-irreversibility.md` — 路径不可逆性
- `wiki/sources/fan-2025-irreversibility.md` — KLD 不可逆性论文
