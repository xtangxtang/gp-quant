# Adaptive State Machine — 架构图

## 组件关系

```
                    ┌─────────────────────────────────────────────────────┐
                    │              run_adaptive_state_machine.py          │
                    │                   CLI 入口                          │
                    │         (--scan_date / --backtest)                 │
                    └─────────────────────────┬───────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────────┐
                    │                   pipeline.py                       │
                    │          run_scan() / run_backtest()               │
                    └─────────────────────────┬───────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────────┐
                    │                strategy.py                          │
                    │          AdaptiveStateMachine 类                    │
                    │        scan() / backtest() 统一入口                 │
                    └─────┬────────────┬──────────────┬───────────┬──────┘
                          │            │              │           │
                          │            │              │           │
          ┌───────────────▼──┐  ┌─────▼──────────┐  │    ┌──────▼─────────┐
          │   feature.py     │  │  weight.py     │  │    │  validator.py  │
          │ FactorCalculator │  │ WeightLearner  │  │    │   Validator    │
          │                  │  │                │  │    │                │
          │ _compute_one_    │  │ update()       │  │    │ verify()       │
          │ symbol()         │  │  ├─ Attention  │◄─┘    │ ingest_        │
          │ build_cross_     │  │  │  模式(默认) │       │ predictions()  │
          │ section()        │  │  └─ IC 模式    │       │ get_           │
          └──────┬───────────┘  └──────┬─────────┘       │ performance()  │
                 │                     │                 └────────────────┘
                 │                     │                        ▲
                 ▼                     ▼                        │
          ┌─────────────────────────────────────────┐          │
          │           state_evaluator.py            │          │
          │          StateEvaluator                 │          │
          │                                         │          │
          │  evaluate_all() ────────────────────────┼──────────┘
          │    │                                    │
          │    ├─ _check_accumulation() → AQ 分数   │
          │    ├─ _check_breakout()   → BQ 分数     │
          │    ├─ _check_collapse()   → collapse    │
          │    ├─ _resolve_state()    → 最终状态    │
          │    │     ▲                              │
          │    │     │ model_up_prob 校准           │
          │    └─────┼──────────────────────────────┘
          │          │
          └──────────┼──────────────────────────────┘
                     │ StateResult (每只股票)
                     │ {state, confidence, aq_score,
                     │  bq_score, pred_return, pred_up_prob}
                     ▼
              输出: signals_YYYYMMDD.csv
              回测: backtest_signals.csv


  ┌──────────────────────────────────────────────────────────────────┐
  │                    Attention 模型链路                             │
  │                                                                  │
  │  ┌─────────────────────┐                                         │
  │  │ train_attention.py  │  一次性训练                             │
  │  │  输入: 历史因子 + 价格                                      │
  │  │  输出: attention_model.pt                                    │
  │  └─────────┬───────────┘                                         │
  │            │                                                     │
  │            ▼                                                     │
  │  ┌─────────────────────┐                                         │
  │  │ attention_learner.py│                                         │
  │  │                     │                                         │
  │  │ FactorAttentionModel│  Transformer (76K 参数)                 │
  │  │   ├─ embedding      │  Linear(47→64) + 位置编码              │
  │  │   ├─ transformer    │  2 层 encoder, 4 heads                 │
  │  │   ├─ summary_token  │  可学习聚合向量                        │
  │  │   ├─ regression     │  预测未来收益率                        │
  │  │   ├─ classification │  预测涨/跌                             │
  │  │   └─ factor_proj    │  投影到因子重要性                      │
  │  │                     │                                         │
  │  │ AttentionTrainer    │  训练循环 (多任务 loss)                 │
  │  │ AttentionLearner    │  推理: extract_weights() + predict()   │
  │  └─────────┬───────────┘                                         │
  │            │                                                     │
  │            │ 每次扫描: (权重, 预测)                              │
  │            ▼                                                     │
  │  strategy.py → _extract_attention_from_cross_section()          │
  │            │                                                     │
  │            ├──→ weight.py: attention_weights → config            │
  │            │                                                     │
  │            └──→ state_evaluator.py: model_predictions 校准       │
  └──────────────────────────────────────────────────────────────────┘


  ┌──────────────────────────────────────────────────────────────────┐
  │                    配置文件                                       │
  │                                                                  │
  │  ┌─────────────────────┐                                         │
  │  │    config.py        │                                         │
  │  │                     │                                         │
  │  │  StockState 枚举    │  idle/accumulation/breakout/hold/      │
  │  │                     │  collapse                               │
  │  │                     │                                         │
  │  │  因子列表 (37 个)   │  AQ_FACTORS / BQ_FACTORS / ALL_FACTORS │
  │  │                     │                                         │
  │  │  DEFAULT_THRESHOLDS │  10+ 个阈值初值                        │
  │  │                     │                                         │
  │  │  AdaptiveConfig     │  运行时配置 (序列化到 JSON)             │
  │  │   ├─ factor_weights │  全量因子权重                          │
  │  │   ├─ attention_     │  Attention 动态权重 (注入打分)         │
  │  │   │    weights      │                                         │
  │  │   ├─ aq_weights     │  AQ 内部 6 因子权重                    │
  │  │   ├─ bq_weights     │  BQ 内部 7 因子权重                    │
  │  │   ├─ thresholds     │  动态阈值 (搜索 + 平滑)                │
  │  │   ├─ factor_scores  │  每因子绩效分                          │
  │  │   └─ learning_rate  │  学习率                                │
  │  └─────────────────────┘                                         │
  │            ▲                                                     │
  │            │ 更新 / 读取                                         │
  │  ┌─────────┴──────────────────────────┐                         │
  │  │ adaptive_config.json               │  持久化                  │
  │  └────────────────────────────────────┘                         │
  └──────────────────────────────────────────────────────────────────┘


  ┌──────────────────────────────────────────────────────────────────┐
  │                    数据流向                                       │
  │                                                                  │
  │  tushare-daily-full/*.csv (5300+ 日线)                          │
  │       │                                                          │
  │       ▼                                                          │
  │  ┌─────────────┐     ┌──────────────────┐                       │
  │  │ feature.py  │────▶│ cross_section    │                       │
  │  │             │     │ (5300×79 矩阵)   │                       │
  │  └─────────────┘     └──┬───────────────┘                       │
  │                         ├─────────────────────────────────────┐ │
  │                         ▼                                     │ │
  │  ┌──────────────────────────┐  ┌────────────────────────────┐ │ │
  │  │ attention_learner.py     │  │ price_series (查未来收益)  │ │ │
  │  │  ├─ 因子权重 {f: w}      │  │                            │ │ │
  │  │  └─ 模型预测 {s: {r, p}} │  └──────────────┬─────────────┘ │ │
  │  └──────────┬───────────────┘                 │               │ │
  │             │                                 │               │ │
  │             ▼                                 ▼               │ │
  │  ┌─────────────────────────────────────────────────────┐     │ │
  │  │ weight.py: update()                                 │     │ │
  │  │  ├─ config.factor_weights = attention_weights       │     │ │
  │  │  ├─ config.attention_weights = attention_weights    │     │ │
  │  │  ├─ 阈值搜索 (10 个阈值)                             │     │ │
  │  │  ├─ 平滑过渡 (alpha=0.2)                            │     │ │
  │  │  └─ AQ/BQ 权重更新                                  │     │ │
  │  └──────────────────────────┬──────────────────────────┘     │ │
  │                             ▼                                │ │
  │  ┌──────────────────────────────────────────────────────────┘ │
  │  ▼                                                            │
  │  ┌──────────────────────────────────────────────────────────┐ │
  │  │ state_evaluator.py: evaluate_all()                       │ │
  │  │                                                          │ │
  │  │  for each stock:                                         │ │
  │  │    accum_score = _check_accumulation(last_row, cfg)     │ │
  │  │    breakout_score = _check_breakout(last_row, cfg)      │ │
  │  │    collapse_score = _check_collapse(last_row, cfg)      │ │
  │  │    state = _resolve_state(scores, model_up_prob, cfg)   │ │
  │  │    aq = _calc_aq(last_row, cfg)  ← attention_weights    │ │
  │  │    bq = _calc_bq(last_row, cfg)  ← attention_weights    │ │
  │  └──────────────────────────────────────────────────────────┘ │
  │                             │                                  │
  │                             ▼                                  │
  │              results/adaptive_state_machine/                   │
  │                signals_YYYYMMDD.csv                            │
  │                backtest_signals.csv                            │
  │                backtest_summary.csv                            │
  │                adaptive_config.json                            │
  └──────────────────────────────────────────────────────────────────┘
```

## 调用时序 (单次 scan)

```
scan(date)
 │
 ├─ 1. FactorCalculator.compute_all(date)
 │     → 5300 只股票 × 79 因子, 并行 28 进程
 │
 ├─ 2. FactorCalculator.build_cross_section()
 │     → 5300×79 截面矩阵
 │
 ├─ 3. Validator.verify(price_df, date)
 │     → 验证 10 天前的预测, 生成奖励信号
 │
 ├─ 4. AttentionLearner.extract_weights()
 │     → 因子权重 {47 factors}
 │     → 每只股票 {pred_return, pred_up_prob}
 │
 ├─ 5. WeightLearner.update()
 │     → 用 attention 权重更新 config
 │     → 搜索最优阈值
 │
 ├─ 6. StateEvaluator.evaluate_all(predictions)
 │     → 每只股票: 4 个检测器 + 模型校准 → 最终状态
 │
 ├─ 7. Validator.ingest_predictions(results)
 │     → 保存预测, 等待 10 天后验证
 │
 └─ 8. 输出 signals + 保存 config
```
