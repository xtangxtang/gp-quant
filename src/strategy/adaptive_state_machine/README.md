# Adaptive State Machine — 自适应状态机策略

> 4 Agent 闭环架构：因子计算 → 权重学习 → 状态判定 → 验证反馈
> 目标：用多 Agent 交互动态更新因子权重和状态阈值，替代硬编码参数

---

## 状态

**Research** — 4 Agent 闭环已跑通全量回测 (2025-01 ~ 2026-03, 15 期扫描)
- 验证准确率 46.8%，与随机相当，需优化因子判别力与阈值
- 详见 [memory/backtest_report.md](memory/backtest_report.md)

---

## 架构

```
Agent 1 (Factor Calculator) → 37 因子截面 (7 类)
         ↓
Agent 2 (Weight Learner) → IC 权重 + 阈值搜索 + Agent 4 奖励/惩罚
         ↓
Agent 3 (State Evaluator) → idle / accumulation / breakout / hold / collapse
         ↓
Agent 4 (Validator) → 验证历史预测 → 奖励/惩罚 → 反馈给 Agent 2
```

## 5 个状态

| 状态 | 含义 | 验证周期 |
|------|------|---------|
| `idle` | 无信号 | — |
| `accumulation` | 惜售吸筹中 | 未来 10 天是否涨 |
| `breakout` | 分岔突破 | 未来 5 天是否突破 |
| `hold` | 持仓中（突破后未退出） | 未来 10 天是否涨 |
| `collapse` | 结构崩塌（退出信号） | 未来 5 天是否跌 |

## 文件结构

| 文件 | Agent | 职责 |
|------|-------|------|
| `config.py` | — | 状态枚举、因子列表 (37)、默认阈值、AdaptiveConfig 数据类 |
| `agent1_factor.py` | Agent 1 | 全市场因子计算（复用 feature_engine.py，ProcessPoolExecutor 并行） |
| `agent2_weight.py` | Agent 2 | 截面 IC 计算 + 坐标下降阈值搜索 + 权重更新 |
| `agent3_state.py` | Agent 3 | 动态状态判定（替代 signal_detector.py） |
| `agent4_validator.py` | Agent 4 | 历史预测验证 + 奖励/惩罚信号生成（指数衰减） |
| `pipeline.py` | 编排 | 4 Agent 循环 + 单日扫描 + 历史回测 |
| `run_adaptive_state_machine.py` | CLI | argparse 入口 |
| `memory/` | — | 设计文档 + 回测报告 |

## 使用方法

### 单日扫描

```bash
python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
    --scan_date 20260423
```

### 历史回测

```bash
python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
    --backtest \
    --start_date 20250101 \
    --end_date 20251231 \
    --interval_days 5
```

### 指定股票

```bash
python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
    --scan_date 20260423 \
    --symbols sh600519,sz000001
```

### 自定义路径

```bash
python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
    --scan_date 20260423 \
    --data_root /path/to/gp-data \
    --daily_dir /path/to/tushare-daily-full \
    --cache_dir /path/to/feature-cache \
    --output_dir /path/to/output
```

## 与 entropy_accumulation_breakout 的区别

| 维度 | entropy_accumulation_breakout | adaptive_state_machine |
|------|------------------------------|----------------------|
| 阈值 | 硬编码 (DetectorConfig 默认值) | 动态学习，每更新 |
| 因子权重 | 固定 (AQ: 25+20+15+15+15+10) | IC 驱动 + Agent 4 奖励/惩罚 |
| 验证 | 无 | Agent 4 滚动验证历史预测 |
| 学习 | 无 | 闭环反馈：正确→奖励，错误→惩罚 |
| 参数约束 | 无 | 阈值限定在基准值 ±20%，平滑过渡 |

## 参数自适应机制

### 因子权重更新公式

```
IC(f) = SpearmanRankCorrelation(factor_f, future_return)
delta_w(f) = lr * IC(f) * (1 + reward(f))
w_{new}(f) = w_{old}(f) + delta_w(f)
```

### 阈值搜索

在 `THRESHOLD_SEARCH_RANGES` 范围内做坐标下降优化：
- `perm_entropy_acc`: [0.52, 0.78] (基准 0.65)
- `dom_eig_breakout`: [0.68, 1.02] (基准 0.85)
- `vol_impulse_breakout`: [1.44, 2.16] (基准 1.8)
- `collapse_need_n`: [2, 4] (基准 3)

### 平滑过渡

```
threshold_{new} = 0.8 * threshold_{old} + 0.2 * threshold_{searched}
```

## 回测结果 (2025-01-02 ~ 2026-03-05, 15 期)

| 指标 | 数值 |
|------|------|
| 每期评估股票 | ~5,300 只 |
| 总信号数 | 4,588 |
| Config 版本 | v0 → v15 (每期递增) |
| 验证准确率 | 46.8% (1,964/4,200) |
| accumulation 准确率 | 50.9% (412/809) |
| collapse 准确率 | 45.8% (1,544/3,374) |
| breakout 准确率 | 47.1% (8/17, 样本不足) |
| 平均 accumulation/期 | 56 只 (1.1%) |
| 平均 breakout/期 | 1.1 只 (0.02%) |
| 平均 collapse/期 | 248 只 (4.7%) |

### 主要发现

- **Agent 2 正常工作**: 每期 5,200-5,300+ 有效远期收益，配置持续更新
- **collapse 信号泛滥**: 占 80% 已验证预测，准确率低于随机
- **breakout 极度稀缺**: 15 期仅 17 次，阈值过严
- **IC 值极低**: 截面 Spearman 相关 ~1e-5，权重更新主要靠 Agent 4 奖励驱动

### 改进方向

1. 放宽 breakout 阈值，收紧 collapse 阈值
2. 改用多日滚动 IC 替代单日 IC
3. 引入基本面因子增强判别力
4. 对 accumulation 加入持续时间约束

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
