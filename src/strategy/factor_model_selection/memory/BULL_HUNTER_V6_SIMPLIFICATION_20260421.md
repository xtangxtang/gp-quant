# Bull Hunter v6 — 代码简化与重构 (2026-04-21)

> 在 v6 三层架构（选股层 / 交易层 / 监控层）落定后，对 `v3_bull_hunter/` 进行
> 一次彻底的死代码清理 + 重复代码提取。本文档记录这次简化的全部内容，作为
> 后续维护的依据。

---

## 0. 背景

v6 架构最终形态：

```
选股层  Agent 1 (factor) → Agent 2 (train) → Agent 3 (predict) → Agent 8 (buy_signal)
交易层  Agent 5 (portfolio)   ←──  Agent 6 (exit_signal)
监控层  统一复盘 run_unified_review (Agent 4 双回路 + Agent 6/8 自适应权重)
```

> **完整 Agent 交互图**（Mermaid 流程图 + 交互矩阵 + 时序图）:
> - [PIPELINE_OVERVIEW.md § v6 三层架构](PIPELINE_OVERVIEW.md#v6-三层架构)
> - [v3_bull_hunter/README.md § 核心思路](../v3_bull_hunter/README.md#核心思路)
> - [BULL_HUNTER_V4_AGENT4_BACKTEST_20260420.md § Agent 完整交互图](BULL_HUNTER_V4_AGENT4_BACKTEST_20260420.md#零agent-完整交互图) (ASCII 版)

v4/v5 时期遗留下两类负担：

1. **Agent 7 Supervisor**：原本设计为"中央大脑"做权重调参，但 Agent 6/8 的
   `auto_adjust_*_weights()` 已经基于相关性自适应，Agent 7 的权重调参分支
   永远不被触发，状态持久化也无人消费。
2. **LLM Advisor**：`use_llm` 链路从 CLI → pipeline → Agent 4 → llm_advisor
   贯穿 5 个文件，但实际效果不稳定，已被规则引擎+auto_adjust 替代。
3. **Agent 6/8 重复代码**：两个文件互为镜像（买/卖方向相反），各自实现
   了 ~200 行几乎相同的 IO/相关性计算/快照逻辑。

---

## 1. 改动清单

| # | 内容 | 影响 |
|---|------|------|
| 1 | 删除 `agent7_supervisor.run_supervisor` / `SupervisorState` / `_load_state` / `_save_state` / `_log_evaluation` / `_extract_factor_from_reason` / `DEFAULT_RULE_WEIGHTS_KEYS` / `apply_directives` 中权重调参分支 | 716 → 422 行 |
| 2 | 删除 `llm_advisor.py` 整文件；移除 `pipeline.py` / `agent4_monitor.py` / `run_bull_hunter.py` 中所有 `use_llm` 引用、`--use-llm` CLI、`PipelineConfig.use_llm`、`_apply_llm_final_filter` | -719 行整文件 + 多点清理 |
| 3 | 清理 `_generate_agent6_directives` 中已被 auto_adjust 替代的权重分支，只保留诊断建议 | 约 -40 行 |
| 4 | **新建 `_signal_common.py`**（295 行）：抽取 Agent 6/8 共享的 8 个 helper：`get_post_trade_return` / `load_recent_prices` / `load_recent_volumes` / `load_rule_weights` / `save_rule_weights_to` / `apply_model_override` / `save_daily_snapshot` / `auto_adjust_signal_weights` | 替代约 400 行重复 |
| 5 | 修复 `pipeline.py` 中 `apply_directives` 被调用两次的 bug（拆出 `(new_p_cfg, new_e_cfg)` tuple，单次返回） | 行为正确性 |
| 6 | `run_review` 收敛为 `run_unified_review` 的 thin wrapper（保留旧 CLI `--review` 兼容性） | 控制流统一 |
| 7 | docstring / README / CLI help 全部刷新到 v6 措辞 | 文档同步 |

---

## 2. `_signal_common.py` 设计

核心抽象：**买卖方向只是符号反转**。`auto_adjust_signal_weights()` 接受
`direction` 参数：

| direction | sign | 含义 |
|-----------|------|------|
| `"buy"`   | +1 | 因子与 forward return 正相关 → 加权 |
| `"sell"`  | -1 | 因子与 forward return 负相关 → 加权 |

调用约定：

```python
# Agent 6 (sell)
auto_adjust_exit_weights = lambda **kw: _sc.auto_adjust_signal_weights(
    direction="sell",
    factors=EXIT_FACTORS,
    default_weights=DEFAULT_RULE_WEIGHTS,
    feature_prefix="exit_",
    snapshot_subdir="sell_weights",
    model_subdir="exit_models",
    weights_filename="rule_weights.json",
    label="Agent 6",
    **kw,
)

# Agent 8 (buy)
auto_adjust_buy_weights = lambda **kw: _sc.auto_adjust_signal_weights(
    direction="buy",
    factors=BUY_FACTORS,
    default_weights=DEFAULT_BUY_WEIGHTS,
    feature_prefix="buy_",
    snapshot_subdir="buy_quality",
    model_subdir="buy_models",
    weights_filename="buy_rule_weights.json",
    label="Agent 8",
    **kw,
)
```

参数：Spearman 相关性，lr=0.3，权重 clip [0.01, 0.25]，最少 10 个样本。

`apply_model_override` 对 Agent 8 需额外排除 `("buy_quality", "buy_reason")`
非特征列，通过 `extra_exclude_cols` 参数传入。

---

## 3. 验证

```bash
# 模块导入烟测
python -c "from src.strategy.factor_model_selection.v3_bull_hunter import (
    pipeline, agent6_exit_signal, agent7_supervisor, agent8_buy_signal, _signal_common
); print('OK')"
# → All imports OK

# CLI help
python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter --help
# → Bull Hunter v6 — 大牛股预测系统

# Live backtest 端到端
python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
    --live-backtest --start_date 20250101 --end_date 20251230 \
    --retrain_interval 0 --monitor_interval 20
# → 通过（详见终端日志）
```

---

## 4. 最终行数

```
   315 agent1_factor.py
   732 agent2_train.py
   159 agent3_predict.py
  1115 agent4_monitor.py
   333 agent5_portfolio.py
   542 agent6_exit_signal.py
   422 agent7_supervisor.py    # 原 716，-294
   523 agent8_buy_signal.py
  1311 pipeline.py
   417 portfolio.py
   477 run_bull_hunter.py
   295 _signal_common.py       # 新增
   333 tracker.py
  6974 total
```

整体节省（含删除的 `llm_advisor.py` 719 行）约 1500 行。

---

## 5. 后续维护要点

- **新增买/卖因子**：只需在 Agent 6/8 的 `*_FACTORS` / `DEFAULT_*_WEIGHTS`
  列表里加项，`_signal_common` 自动接管 IO + auto-adjust。
- **Agent 7 当前定位**：只是 `run_unified_review` 调用的纯函数集合（评估、
  生成指令、判断干预），不再持有任何状态；apply_directives 只处理 Agent 5
  动作。
- **复盘统一入口**：`pipeline.run_unified_review`（`run_review` 是兼容 wrapper）。
- **不要再加 `use_llm` 类参数**：规则 + auto_adjust 已是稳定方案。

---

## 6. 关联文档

- 架构总览：[BULL_HUNTER_V4_ARCHITECTURE_20260420.md](BULL_HUNTER_V4_ARCHITECTURE_20260420.md)
  （v6 架构与 v4 一致，本次仅做实现层简化）
- 上一次迭代：[BULL_HUNTER_V4_FULL_SUMMARY_20260420.md](BULL_HUNTER_V4_FULL_SUMMARY_20260420.md)
- Pipeline 总览：[PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md)
- 策略 README：[../v3_bull_hunter/README.md](../v3_bull_hunter/README.md)
