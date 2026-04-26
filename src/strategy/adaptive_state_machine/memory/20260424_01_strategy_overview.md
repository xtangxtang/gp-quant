# 自适应状态机策略 — 完整逻辑总结

**文档日期**: 2026-04-24
**策略版本**: v4
**回测准确率**: 50.9%（2,606 正确 / 5,117 已验证）
**回测区间**: 2025-01-02 ~ 2026-03-05，15 个扫描日，每 20 个交易日一次

---

## 一、它是什么

一套 A 股日线量化扫描策略。每天对全市场 5000+ 只股票做四件事：

```
算因子 → 更新权重/阈值 → 判定每只股票的状态 → 验证历史预测
```

代码结构：

```
adaptive_state_machine/
├── config.py              # 共享配置: 状态枚举、因子列表、默认阈值、AdaptiveConfig
├── feature.py             # 因子计算 (复用 feature_engine.py:build_features)
├── weight.py              # IC 权重更新 + 全量阈值网格搜索 + 分布惩罚
├── state_evaluator.py     # 分数竞争状态判定 (非级联)
├── validator.py           # 历史验证反馈
├── strategy.py            # 统一入口: AdaptiveStateMachine.scan() / .backtest()
├── pipeline.py            # 薄包装: run_scan() / run_backtest() (向后兼容)
├── run_adaptive_state_machine.py  # CLI 入口
└── memory/                # 设计文档与回测报告
```

---

## 二、输入数据

**数据源**: `gp-data/tushare-daily-full/*.csv`，每只股票一个 CSV，包含日线 OHLCV + 资金流。

**复用**: 因子计算直接调用 `entropy_accumulation_breakout` 的 `feature_engine.py:build_features()`，不重复造轮子。它输出日线 + 周线两个带 37 个衍生因子的 DataFrame。

---

## 三、四个内部模块

### 1. Feature Calculator (`feature.py`)

**做什么**: 并行调用 `build_features()` 为每只股票计算 37 个因子。

**关键设计**:
- 多进程池（默认 4 workers），每只股票独立计算
- 过滤：数据不足 80 行或解析失败的股票直接跳过
- 最终输出一个截面 DataFrame：index=symbol，columns=37 个因子

**37 个因子分 7 类**:

| 类别 | 数量 | 例子 |
|------|------|------|
| 熵值 | 9 | perm_entropy_m, path_irrev_m |
| 波动率 | 8 | volatility_m, vol_compression, vol_impulse |
| 特征值 | 2 | dom_eig_m, dom_eig_l |
| 相干性 | 4 | purity_norm, coherence_decay_rate |
| 资金流 | 9 | mf_big_net_ratio, mf_flow_imbalance, mf_big_streak |
| 价格 | 1 | breakout_range |
| 分钟 | 4 | intraday_perm_entropy, intraday_range_ratio |

---

### 2. Weight Learner (`weight.py`)

**做什么**: 两件事 — 算因子 IC（预测力），用网格搜索优化状态判定阈值。

#### IC 计算（核心创新）

不是单日截面 IC（~1e-5，几乎为 0），而是**多日滚动累积**:
- 每个扫描日的特征矩阵和前向收益率都保存到历史缓冲
- 累积满 5 天后，用 `pd.concat` 拼接所有日期的数据
- 样本量从 ~5000 提升到 26000~53000
- IC 从 0.00001 提升到 0.08~0.21

公式: `Spearman 秩相关(因子值, 未来10天收益率)`

权重更新:
```
delta_w = lr × IC × (1 + validator_rewards)
```
更新后归一化，使权重总和不变。

#### 阈值搜索

不是逐个阈值独立优化（旧版的致命缺陷），而是**全量状态模拟**:

1. 对一组完整的阈值组合，模拟全市场所有股票的状态判定
2. 计算各状态信号的平均收益率（信号正确的股票赚了多少）
3. 加入**分布惩罚**: collapse 超过 30% 强扣分，其他状态超过 40% 扣分
4. 用坐标下降法：一次只调一个阈值，但评估的是完整状态分布

```python
# 搜索范围（10 个阈值）
THRESHOLD_SEARCH_RANGES = {
    "perm_entropy_acc": (0.52, 0.78),           # accumulation 熵
    "path_irrev_acc": (0.04, 0.06),             # accumulation 方向性
    "dom_eig_breakout": (0.68, 1.02),           # breakout 主特征值
    "vol_impulse_breakout": (1.44, 2.16),       # breakout 量脉冲
    "perm_entropy_breakout_max": (0.60, 0.90),  # breakout 熵上限
    "perm_entropy_collapse": (0.72, 1.00),      # collapse 熵
    "path_irrev_collapse": (0.005, 0.02),       # collapse 方向性
    "entropy_accel_collapse": (0.03, 0.08),     # collapse 熵加速
    "collapse_need_n": (3, 4),                  # collapse 需要几个信号
    ...
}
```

#### AQ/BQ 子权重

同时更新 Accumulation Quality 的 6 个内部权重和 Breakout Quality 的 7 个内部权重，按各因子 IC 绝对值比例分配。

---

### 3. State Evaluator (`state_evaluator.py`)

**做什么**: 对每只股票计算三个状态的得分，分数竞争决定最终状态。

#### 状态判定逻辑（分数竞争制，不是 if/elif 级联）

对每只股票同时计算三个分数（0~1）:

- **accumulation_score** = 满足 accumulation 条件的条件数 / 总条件数（4 个条件）
- **breakout_score** = 满足 breakout 条件的条件数 / 总条件数（4 个条件）
- **collapse_score** = 触发的 collapse 信号数 / collapse_need_n

然后按优先级决定状态:

```
1. collapse_score >= 1.0  → COLLAPSE（4 个信号全触发，直接退出）
2. breakout_score > 0.5 且 ≥ accum_score  → BREAKOUT（入场）
3. 已在持仓中 → 检查超时：超过 3 个扫描日未 breakout → IDLE，否则 HOLD
4. accum_score > 0.5  → ACCUMULATION（蓄力）
5. 其他  → IDLE（无信号）
```

**关键设计**:
- Breakout 不再依赖 accumulation 前置条件（旧版双重封锁导致 breakout 几乎不触发）
- Hold 状态跨扫描日期继承，但有 3 个扫描日（~60 交易日）超时自动释放
- collapse 优先级最高，触发就退出持仓

#### 5 个状态

| 状态 | 含义 | 验证周期 |
|------|------|----------|
| idle | 无信号，不满足任何条件 | — |
| accumulation | 蓄力：低熵 + 高不可逆性 + 资金流入 | 未来 10 天是否涨 |
| breakout | 突破：高主特征值 + 放量 + 低熵 | 未来 5 天是否突破 |
| hold | 持仓：突破后维持，超时自动释放 | 未来 10 天是否继续涨 |
| collapse | 崩塌：高熵 + 低不可逆性 + 纯度下降 | 未来 5 天是否跌 |

#### 置信度计算

- AQ（Accumulation Quality）：用 6 个 AQ 子因子加权，归一化到 [0,1]
- BQ（Breakout Quality）：用 7 个 BQ 子因子加权，归一化到 [0,1]
- composite = 0.4 × AQ + 0.6 × BQ

AQ/BQ 的子权重由 Weight Learner 动态更新（不是写死的）。

---

### 4. Validator (`validator.py`)

**做什么**: 回顾历史预测，对比实际走势，生成奖励/惩罚信号给 Weight Learner。

**流程**:
1. 接收 State Evaluator 的预测，加入队列
2. 等验证周期到了（accumulation 等 10 天，breakout 5 天，collapse 5 天，hold 10 天）
3. 对比实际价格：
   - accumulation/breakout/hold → 涨了算对，跌了算错
   - collapse → 跌了算对，涨了算错
4. 计算奖励幅度 = 实际收益率绝对值 × 正确性（+1 或 -1）
5. 按因子贡献度分配奖励（哪个因子对判定影响大，谁分得多）
6. 指数衰减：越远的预测权重越低（λ = 0.05）

最终输出：`{factor: avg_reward}` 传给 Weight Learner，用于更新因子权重。

---

## 四、单日扫描的完整流程

```python
strategy.scan("20260423")
```

1. **因子计算**: 5000+ 只股票 × 37 个因子 → 截面 DataFrame
2. **Validator 验证历史**: 检查队列中到期的预测，对比实际价格
3. **Weight Learner 更新**:
   - 计算多日滚动 IC（如果累积满 5 天）
   - 全量状态模拟 + 网格搜索最优阈值
   - 更新因子权重（IC × validator_rewards）
   - 更新 AQ/BQ 子权重
4. **State Evaluator 判定**: 每只股票计算三个状态分数 → 竞争决定状态
5. **收集信号 + 入库**: validator 接收本次预测，加入队列
6. **保存结果**: 信号 CSV + 配置 JSON

---

## 五、回测的完整流程

```python
strategy.backtest("20250101", "20260331", interval_days=20)
```

1. 加载交易日历，过滤日期范围，每隔 20 天取一个扫描日（15 个）
2. 加载价格矩阵（一次性，用于 forward return 计算和 validator 验证）
3. 初始化三个**持久化组件**（跨扫描日期复用）：
   - **Validator**: 积累历史预测，跨日期验证
   - **StateEvaluator**: 维护 _hold_state（持仓记录），跨日期继承
   - **WeightLearner**: 维护滚动 IC 历史缓冲，跨日期累积
4. 循环扫描每个日期，复用上述组件
5. 输出：backtest_signals.csv（所有信号）+ backtest_summary.csv（逐日摘要）

**持久化是关键**: 如果每次扫描都创建新实例，Hold 状态不会继承，滚动 IC 历史会丢失，Validator 预测队列会清空。

---

## 六、自适应配置（AdaptiveConfig）

一个 dataclass，包含:

| 字段 | 含义 | 谁更新 |
|------|------|--------|
| `factor_weights` | 32 个因子的权重 | Weight Learner |
| `thresholds` | 10 个状态判定阈值 | Weight Learner |
| `learning_rate` | 学习率，固定 0.10 | — |
| `factor_scores` | 每因子绩效分 | Weight Learner |
| `aq_weights` | AQ 内部 6 个权重 | Weight Learner |
| `bq_weights` | BQ 内部 7 个权重 | Weight Learner |
| `aq_bq_weight` | composite = 0.4*AQ + 0.6*BQ | — |
| `version` / `last_updated` | 元数据 | 每期递增 |

每期扫描后版本递增，持久化到 JSON。下次扫描自动加载。

---

## 七、验证结果（v4 回测）

| 指标 | 数值 |
|------|------|
| 扫描日期 | 15 个（每 20 个交易日） |
| 总信号 | 5,856 条 |
| 累计验证 | 5,117 条 |
| **整体准确率** | **50.9%** |
| collapse 信号 | 1,091 条（50.0% 准确） |
| breakout 信号 | 1,275 条（50.1% 准确） |
| hold 信号 | 1,884 条（51.8% 准确） |
| accumulation 信号 | 867 条（51.3% 准确） |

### 状态分布变化（v1 → v4）

| 状态 | v1 占比 | v4 占比 | 变化 |
|------|---------|---------|------|
| collapse | 80.3% | 21.3% | **-59pp** |
| breakout | 0.4% | 24.9% | **+24.5pp** |
| hold | 0% | 36.8% | **+36.8pp** |
| accumulation | 19.2% | 17.0% | -2.2pp |

### 优化历程

| 版本 | 改动 | 准确率 |
|------|------|--------|
| v1 (基线) | 单日截面 IC + 级联判定 + 独立阈值搜索 | 46.8% |
| v2 | 多日滚动 IC (5 日累积, ~26000+ 样本) | 46.8% |
| v3 | 分数竞争判定 + 分布惩罚 + breakout 独立检测 + Hold 跨日期继承 | 50.1% |
| v4 | collapse_need_n 3→4 + collapse 专属分布惩罚 (30%) | 50.9% |

---

## 八、输出文件

- `results/adaptive_state_machine/backtest_signals.csv` — 5,856 条信号记录
- `results/adaptive_state_machine/backtest_summary.csv` — 15 期逐日摘要
