# Holding Position Model — 持仓决策模型

> 选股模型决定"买哪个"，持仓模型决定"拿还是走"

---

## 1. 问题定义

**现有模型的盲区**：adaptive_state_machine 做截面排序选股票，但它不知道"这只股票我已经持有 3 天、浮盈 +6%、从最高点回撤了 4%"。同样的因子模式，对刚入场和已持仓的决策完全不同。

**本策略解决的问题**：

```
输入: 一只股票最近 60 天因子序列 + 持仓特征
输出: 拿住 / 减仓 / 走人 + 置信度
```

这是**时序决策**，不是截面排序。

---

## 2. 核心设计

### 2.1 三个输出头（不是 5 类分类）

不用 softmax 做 5 类互斥分类，而是三个独立头：

| 头 | 类型 | 含义 | 决策阈值 |
|---|------|------|---------|
| `stay_prob` | sigmoid | 当前上涨趋势延续的概率 | > 0.7 → 拿住 |
| `collapse_risk` | sigmoid | 未来 5 天内崩塌风险 | > 0.3 → 走人 |
| `expected_days` | linear | 预期还能持有多少天 | < 3 → 减仓 |

为什么不做 5 类分类：
- 5 个状态之间有重叠（hold 和 accumulation 边界模糊）
- softmax 强制互斥不合理（一只股票可能同时处于 hold + breakout 过渡）
- 决策需要的是"风险有多大"和"还能拿多久"，不是"属于哪一类"

### 2.2 持仓特征（5 维）

这是当前所有模型都没有的：

| 特征 | 计算方式 | 为什么重要 |
|------|---------|-----------|
| `days_since_entry` | 当前日 - 入场日 | 不同持有阶段风险不同 |
| `unrealized_pnl` | (current - entry) / entry | +30% 和 -5% 时决策完全不同 |
| `max_pnl_since_entry` | (peak - entry) / entry | 入场后最高到过多少 |
| `drawdown_from_peak` | (current - peak) / peak | 从最高点回撤多少是关键退出信号 |
| `entry_price_position` | entry 在 60 天价格中的分位 | 高位追入 vs 低位建仓，风险不同 |

### 2.3 模型架构

```
Input A: 因子时序 (60 × n_factors)
  ↓
  Transformer Encoder (复用现有架构)
  ↓ Linear
  summary_vector (128 维)

Input B: 持仓特征 (5 维)
  ↓ Linear
  holding_vector (32 维)

Concat(summary_vector, holding_vector) → 160 维
  ↓
  三个头:
    stay_prob:     Linear(160 → 1) → sigmoid
    collapse_risk: Linear(160 → 1) → sigmoid
    expected_days: Linear(160 → 1) → relu (约束 > 0)
```

参数量：~600K，和现有模型相当。

### 2.4 因子复用

不重新造轮子：因子计算完全复用 `entropy_accumulation_breakout.feature_engine.build_features()`，和 `adaptive_state_machine/feature.py` 的 `FactorCalculator`。

---

## 3. 训练数据构造

### 3.1 模拟持仓轨迹

不是"每只股票每天一个样本"，而是**模拟在不同时点买入后的持有过程**：

```
对每只股票:
  对历史每一天作为 entry_day:
    entry_price = 当天 open
    对后续 1-20 天（每个持有日）:
      current_price = 当天 close
      peak_price = 从 entry 到 current 的最高价
      计算持仓 5 维特征
      用未来走势打标签
      因子窗口 = current 往前 60 天
      → 一个训练样本
```

### 3.2 标签定义（用未来走势，不是用当前规则）

不用硬阈值切 5 类，用连续标签：

| 标签 | 类型 | 定义 |
|------|------|------|
| `stay_label` | 0/1 | 未来 10 天涨跌幅 > 0 且最大回撤 < 5% → 1 |
| `collapse_label` | 0/1 | 未来 5 天跌幅 > 5% 或从当前回撤 > 10% → 1 |
| `days_label` | 回归 | 到未来第一次满足 collapse 条件过了多少天（上限 20） |

注意：训练时可以看到未来（回标），推理时只看过去。

### 3.3 样本量估算

```
500 只股票 × 500 天 × 20 个持有日 ≈ 5M 样本
实际过滤后约 3-4M（去掉 NaN、停牌、涨跌停等）
```

575K 参数的模型，数据量足够。

### 3.4 防泄漏措施

- **按时间 split**：前 80% 交易日训练，后 20% 测试（不是随机 split）
- entry_day 不全量枚举，随机采样 30% 的 entry_day
- 同一只股票的不同 entry_day 样本虽然因子窗口有重叠，但持仓特征不同，且时序 split 避免训练集和测试集有相同日期的样本

---

## 4. 文件结构

```
src/strategy/holding_position_model/
├── README.md                    # 使用说明
├── __init__.py
├── __main__.py                  # python -m 入口
├── config.py                    # 配置：因子列表、模型超参、阈值
├── feature.py                   # 因子计算（复用 adaptive_state_machine/feature.py）
├── model.py                     # Transformer + 持仓特征双分支架构
├── data_builder.py              # 模拟持仓轨迹 + 标签构造
├── train.py                     # 训练脚本
├── inference.py                 # 单只股票持仓决策
├── run_holding_position_model.py  # CLI 入口
└── memory/
    └── 00_strategy_plan.md      # 本文件（策略总纲领）
```

---

## 5. 使用方式

### 5.1 训练

```bash
python -m src.strategy.holding_position_model.run_holding_position_model \
    --train \
    --data_dir /path/to/tushare-daily-full \
    --data_root /path/to/gp-data \
    --output_model src/strategy/holding_position_model/models/holding_model.pt \
    --max_stocks 500
```

### 5.2 持仓决策

```bash
python -m src.strategy.holding_position_model.run_holding_position_model \
    --symbol sz000001 \
    --entry-price 15.20 \
    --entry-date 20260424 \
    --current-date 20260427 \
    --data_dir /path/to/tushare-daily-full \
    --data_root /path/to/gp-data \
    --model src/strategy/holding_position_model/models/holding_model.pt
```

输出：
```
Symbol: sz000001
Entry:  15.20 (20260424)
Current: 16.10 (+5.9%)
Peak:   16.30 (drawdown -1.2%)
Held:   3 days

stay_prob:     0.72  → 拿住
collapse_risk: 0.08  → 风险低
expected_days: 7.3   → 预期还能持 7 天

建议: 继续持有
```

### 5.3 批量扫描已持仓

```bash
python -m src.strategy.holding_position_model.run_holding_position_model \
    --scan-positions positions.csv \
    --data_dir /path/to/tushare-daily-full \
    --data_root /path/to/gp-data \
    --model src/strategy/holding_position_model/models/holding_model.pt
```

positions.csv 格式：
```csv
symbol,entry_price,entry_date
sz000001,15.20,20260424
sh600519,1800.00,20260420
...
```

---

## 6. 与 adaptive_state_machine 的关系

| | adaptive_state_machine | holding_position_model |
|---|---|---|
| 问题 | 截面选股（全市场选 Top N） | 持仓决策（手里这只该不该走） |
| 输入 | 全市场截面 | 单只股票 + 持仓信息 |
| 输出 | 5 状态（idle/accum/breakout/hold/collapse） | stay_prob / collapse_risk / expected_days |
| 模型 | Transformer 分位数回归 | Transformer + 持仓特征双分支 |
| 配合方式 | 每天选股 → 买入候选 | 每天检查已持仓 → 卖出决策 |

**两者配合的工作流**：

```
每天收盘后:
  1. adaptive_state_machine 扫描全市场 → 选出 Top 10 breakout/accumulation
  2. 对比当前持仓 → 新信号加入
  3. holding_position_model 检查每只已持仓 → stay_prob / collapse_risk
  4. collapse_risk > 0.3 的 → 卖出
  5. stay_prob > 0.7 的 → 继续持有
  6. 新信号填补卖出后的仓位
```

---

## 7. 实施步骤

| Phase | 内容 | 优先级 |
|-------|------|--------|
| Phase 1 | `data_builder.py` — 模拟持仓轨迹 + 标签构造 | P0 |
| Phase 2 | `model.py` — Transformer + 持仓特征双分支 | P0 |
| Phase 3 | `train.py` — 训练脚本 + 验证 | P0 |
| Phase 4 | `inference.py` + CLI — 单只/批量持仓决策 | P0 |
| Phase 5 | 回测 — 和选股模型联调 | P1 |

Phase 1-4 是最小可用版本，Phase 5 是完整工作流验证。

---

## 8. 风险和注意事项

1. **标签泄漏**：训练时用未来走势定义标签，推理时只看过去。必须严格按时间 split。
2. **样本重叠**：同一只股票不同 entry_day 的样本因子窗口有重叠。通过时序 split 和时间衰减缓解。
3. **过拟合风险**：575K 参数 vs 3-4M 样本，数据量足够但需要关注分布（行业、市值）是否均衡。
4. **持仓特征只在推理时有意义**：训练时需要模拟 entry_price，模拟的 entry_price 和真实用户的 entry_price 分布可能不同。
