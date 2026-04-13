---
name: entropy-accumulation-breakout
description: "运行熵惜售分岔突破策略（三阶段状态机：惜售吸筹→分岔突破→结构崩塌）。Use when: 熵惜售, 惜售吸筹, 分岔突破, entropy accumulation, breakout, 筹码集中, 临界减速, critical slowing, 路径不可逆, path irreversibility, 主特征值, dominant eigenvalue, 结构崩塌, collapse, 三阶段, 状态机, 熵突破扫描, 熵选股。"
argument-hint: "扫描日期或股票代码，例如：20260410 或 sh600000"
---

# 熵惜售分岔突破策略 (Entropy-Accumulation-Breakout)

基于信息熵、路径不可逆性、临界减速理论的三阶段交易系统：

```
惜售吸筹 (Accumulation)  →  分岔突破 (Bifurcation)  →  结构崩塌 (Collapse)
     ↓                          ↓                          ↓
   买入准备                     买入执行                     卖出退出
```

## 核心逻辑

### 阶段1: 惜售吸筹 (Accumulation)
- 置换熵 < 0.65（筹码集中、交易有序化）
- 路径不可逆性 > 0.05（定向资金力量）
- 量缩（5日/20日量比 < 0.7）+ 波动率压缩（短/长比 < 0.8）
- 需连续 ≥ 5 天满足条件
- 质量评分 AQ = 30%熵 + 25%不可逆 + 20%量缩 + 15%BBW分位 + 10%大单比

### 阶段2: 分岔突破 (Bifurcation)
- 主特征值 > 0.85（临界减速 → 即将突破）
- 量脉冲 > 1.8×（成交量暴增）
- 突破位置 > 0.8（接近20日高点）
- 突破时熵仍 < 0.75（有序突破，非混乱暴涨）
- 质量评分 BQ = 35%特征值 + 30%量脉冲 + 20%熵 + 15%不可逆
- 综合得分 = 40% AQ + 60% BQ

### 阶段3: 结构崩塌退出 (Collapse)
- 4个信号中满足3个即退出:
  1. 熵飙升 > 0.90（无序化）
  2. 路径不可逆骤降 < 0.01（主力撤离）
  3. 熵加速 > 0.05（结构加速瓦解）
  4. 量能衰竭（< 峰值×0.3）
- 硬止损: -10%

## 执行步骤

### 1. 全市场扫描（默认最近交易日）

```bash
cd /nvme5/xtang/gp-workspace/gp-quant

./scripts/run_entropy_accumulation_breakout.sh
```

### 2. 指定日期扫描

```bash
./scripts/run_entropy_accumulation_breakout.sh \
  --scan-date 20260410 \
  --top-n 30
```

### 3. 扫描特定股票

```bash
./scripts/run_entropy_accumulation_breakout.sh \
  --symbols sh600000,sz000001,sh600036
```

### 4. 前瞻回测

```bash
./scripts/run_entropy_accumulation_breakout.sh \
  --backtest-start-date 20260101 \
  --backtest-end-date 20260331 \
  --hold-days 5 \
  --max-positions 10
```

### 5. 直接用 Python 运行

```bash
python src/strategy/entropy_accumulation_breakout/run_entropy_accumulation_breakout.py \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv \
  --out_dir ./results/entropy_accumulation_breakout \
  --scan_date 20260410 \
  --top_n 30
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
├── feature_engine.py                     # 25+ 特征工程
├── market_regime.py                      # 大盘状态门控
├── backtest_fast.py                      # 高速回测引擎
└── README.md                             # 完整文档
```

## 关键参数

| 参数 | 阈值 | 阶段 | 说明 |
|------|------|------|------|
| `perm_entropy_low` | 0.65 | 惜售 | 排列熵低于此值=有序 |
| `path_irrev_high` | 0.05 | 惜售 | 路径不可逆高于此值=定向力量 |
| `vol_shrink_threshold` | 0.7 | 惜售 | 量缩比 |
| `accum_min_days` | 5 | 惜售 | 最少连续满足天数 |
| `dom_eig_threshold` | 0.85 | 突破 | 主特征值→1=临界减速 |
| `vol_impulse_threshold` | 1.8 | 突破 | 成交量脉冲倍数 |
| `breakout_range_min` | 0.8 | 突破 | 突破位置 [0,1] |
| `perm_entropy_collapse` | 0.90 | 崩塌 | 熵飙升退出阈值 |
| `path_irrev_collapse` | 0.01 | 崩塌 | 不可逆骤降退出 |
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

- 日线 K 线（含资金流）: `tushare-daily-full/`（至少 500 天回溯）
- 股票基本信息: `tushare_stock_basic.csv`（ST 过滤 + 行业分类）

## Wiki 参考

- `wiki/entities/entropy-accumulation-breakout.md` — 策略实体页
- `wiki/concepts/entropy.md` — 熵的理论基础
- `wiki/concepts/bifurcation.md` — 分岔理论
- `wiki/concepts/path-irreversibility.md` — 路径不可逆性
- `wiki/sources/fan-2025-irreversibility.md` — KLD 不可逆性论文
