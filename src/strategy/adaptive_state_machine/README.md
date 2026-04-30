# Adaptive State Machine — 自适应状态机策略

> 因子计算 → Transformer 预测 → 截面排序 → 状态判定

---

## 状态

**Research** — Transformer 分位数回归 + 5 状态生命周期建模

---

## 架构

```
feature.py (因子计算)
    ↓ 复用 entropy_accumulation_breakout.feature_engine
attention_learner.py (Transformer 推理)
    ↓ 分位数回归 (9 分位) + 收益预测 + 涨跌分类
截面排序 (Sharpe-like: pred_return / risk_IQR)
    ↓
状态判定: idle / accumulation / breakout / hold / collapse
```

## 5 个状态

| 状态 | 含义 | 判定逻辑 |
|------|------|---------|
| `idle` | 无信号 | 默认状态 |
| `accumulation` | 低熵蓄力 | 截面排名 > 70% + q50 > 50% + pred_return > 0 |
| `breakout` | 分岔突破 | 截面排名 > 90% + q50 > 80% + pred_return > 0 |
| `hold` | 持仓延续 | breakout 后跨扫描日继承 (最多 3 周期) |
| `collapse` | 结构崩塌 | 截面排名 < 10% + q50 < 20% |

## 文件结构

| 文件 | 职责 |
|------|------|
| `config.py` | 状态枚举、因子列表、默认阈值、AdaptiveConfig 数据类 |
| `feature.py` | 全市场因子计算（复用 feature_engine.py，ProcessPoolExecutor 并行） |
| `attention_learner.py` | Transformer 模型: 分位数回归 + 因子重要性提取 |
| `state_evaluator.py` | hold 状态继承管理 |
| `strategy.py` | 统一策略类: scan / backtest 入口 |
| `pipeline.py` | 兼容层: run_scan / run_backtest 委托给 AdaptiveStateMachine |
| `train_attention.py` | Transformer 模型训练脚本 |
| `run_adaptive_state_machine.py` | CLI 入口 |
| `models/` | 训练好的模型权重 |
| `memory/` | 设计文档 + 回测报告 |

## 使用方法

### 单日扫描

```bash
python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
    --scan_date 20260423 \
    --attention_model src/strategy/adaptive_state_machine/models/attention_model.pt
```

### 历史回测

```bash
python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
    --backtest \
    --start_date 20250101 \
    --end_date 20251231 \
    --interval_days 5 \
    --attention_model src/strategy/adaptive_state_machine/models/attention_model.pt
```

### 指定股票

```bash
python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
    --scan_date 20260423 \
    --symbols sh600519,sz000001 \
    --attention_model src/strategy/adaptive_state_machine/models/attention_model.pt
```

### 自定义路径

```bash
python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
    --scan_date 20260423 \
    --data_root /path/to/gp-data \
    --daily_dir /path/to/tushare-daily-full \
    --cache_dir /path/to/feature-cache \
    --output_dir /path/to/output \
    --attention_model /path/to/model.pt
```

## Transformer 模型

### 架构

```
输入: (seq_len=60, n_factors) 每只股票的因子时序
  ↓
Linear(n_factors → d_model=128) + 可学习位置编码
  ↓
Transformer Encoder: 4 layers, 8 heads, d_model=128
  ↓
多任务头:
  * 回归: 未来收益率 (MSE)
  * 分类: 涨/跌 (CrossEntropy)
  * 分位数回归: 9 个分位点 (Quantile Loss, 10/20/.../90%)
  ↓
输出: pred_return + up_prob + quantiles → Sharpe-like 评分 → 截面排序
```

### 训练

```bash
python -m src.strategy.adaptive_state_machine.train_attention \
    --data_root /path/to/gp-data \
    --daily_dir /path/to/tushare-daily-full \
    --output_model /path/to/model.pt
```

## 与 entropy_accumulation_breakout 的区别

| 维度 | entropy_accumulation_breakout | adaptive_state_machine |
|------|------------------------------|----------------------|
| 状态判定 | 固定阈值 (DetectorConfig) | Transformer 分位数回归 + 截面排序 |
| 因子 | 硬编码权重 | 模型学习因子表示 |
| 预测 | 无 | 收益率 + 涨跌 + 分位数分布 |
| 学习 | 无 | 离线训练，在线推理 |

## 回测结果

详见 [memory/backtest_report.md](memory/backtest_report.md)

---

## 输出文件

```
results/adaptive_state_machine/
├── signals_{scan_date}.csv    ← 单日扫描信号
├── backtest_signals.csv       ← 回测所有信号
├── backtest_summary.csv       ← 按日期汇总
└── config/
    └── adaptive_config.json   ← 当前配置 (持久化)
```
