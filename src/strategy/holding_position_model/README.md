# Holding Position Model — 持仓决策模型

> 选股模型决定"买哪个"，持仓模型决定"拿还是走"

---

## 状态

**Research** — Transformer + 持仓特征双分支架构，输出 stay_prob / collapse_risk / expected_days

---

## 架构

```
Input A: 因子时序 (60 × n_factors) → Transformer Encoder → summary (128 维)
Input B: 持仓特征 (5 维) → Linear → holding_vector (32 维)
  ↓
Concat → 三个头:
  stay_prob:     sigmoid → 趋势延续概率 (> 0.7 → 拿住)
  collapse_risk: sigmoid → 崩塌风险 (> 0.3 → 走人)
  expected_days: relu   → 预期持有天数 (< 3 → 减仓)
```

## 持仓特征 (5 维)

| 特征 | 说明 |
|------|------|
| `days_since_entry` | 入场后第几天 |
| `unrealized_pnl` | 当前浮盈/浮亏比例 |
| `max_pnl_since_entry` | 入场后最高浮盈 |
| `drawdown_from_peak` | 从持仓最高点的回撤 |
| `entry_price_position` | 入场价在近 60 天的分位 |

## 文件结构

| 文件 | 职责 |
|------|------|
| `config.py` | 配置：模型超参、训练参数、决策阈值 |
| `feature.py` | 因子计算（复用 feature_engine） |
| `data_builder.py` | 模拟持仓轨迹 + 标签构造 |
| `model.py` | Transformer + 持仓特征双分支 |
| `train.py` | 训练脚本 |
| `inference.py` | 单只/批量持仓决策 |
| `run_holding_position_model.py` | CLI 入口 |

## 使用方法

### 训练

```bash
python -m src.strategy.holding_position_model.run_holding_position_model \
    --train \
    --data_dir /path/to/tushare-daily-full \
    --data_root /path/to/gp-data \
    --output_model src/strategy/holding_position_model/models/holding_model.pt \
    --max_stocks 500
```

### 单只持仓决策

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

### 批量持仓扫描

```bash
python -m src.strategy.holding_position_model.run_holding_position_model \
    --scan-positions positions.csv \
    --data_dir /path/to/tushare-daily-full \
    --data_root /path/to/gp-data \
    --model src/strategy/holding_position_model/models/holding_model.pt
```

positions.csv:
```csv
symbol,entry_price,entry_date
sz000001,15.20,20260424
sh600519,1800.00,20260420
```

## 与 adaptive_state_machine 的配合

```
每天收盘后:
  1. adaptive_state_machine 扫描全市场 → 选出 Top 10 breakout/accumulation
  2. holding_position_model 检查已持仓 → stay_prob / collapse_risk
  3. collapse_risk > 0.3 → 卖出
  4. stay_prob > 0.7 → 继续持有
  5. 新信号填补卖出后的仓位
```

## 训练数据

- 模拟持仓轨迹：每只股票随机采样 30% entry_day，模拟持有 1-20 天
- 标签：用未来走势定义（训练时看未来，推理时只看过去）
- 时间 split：前 80% 训练，后 20% 测试

详见 [memory/00_strategy_plan.md](memory/00_strategy_plan.md)
