---
name: hold-exit-decision
description: "持有/退出决策：评估已持仓股票应继续持有还是卖出。Use when: 持有判断, 退出判断, hold or sell, exit decision, 上升趋势持有, entropy hold, 快速膨胀, rapid expansion, 衰竭退出, exhaustion exit, 状态流, state flow, 该不该卖。"
argument-hint: "股票代码和买入日期，例如：sh688268 20260301"
---

# 持有/退出决策 (Hold/Exit Decision)

评估已买入的股票当前应继续持有还是卖出。状态流转路径：熵持有 → 快速膨胀持有 → 衰竭退出。

## 三个子策略

| 策略 | 脚本 | 功能 |
|------|------|------|
| **上升趋势持有状态流** | `run_uptrend_hold_state_flow.sh` | 综合状态路径评估 |
| **熵持有判断** | `run_entropy_hold_judgement.sh` | 低熵有序态继续持有 |
| **快速膨胀持有** | `run_rapid_expansion_hold.sh` | 动量膨胀阶段继续持有 |
| **衰竭退出** | `run_rapid_expansion_exhaustion_exit.sh` | 检测动量衰竭，建议退出 |

## 必填参数

- `--symbol-or-name` — 股票代码或名称（如 `sh688268` / `688268.SH` / `华特气体`）
- `--start-date` — 买入日期 YYYYMMDD

## 执行步骤

### 1. 综合状态流评估（推荐首选）

```bash
cd /nvme5/xtang/gp-workspace/gp-quant

./scripts/run_uptrend_hold_state_flow.sh \
  --symbol-or-name sh688268 \
  --start-date 20260301
```

输出到 `results/uptrend_hold_state_flow/`，包含：
- 状态路径历史（每日状态转换记录）
- 当前状态评估
- 继续持有 / 卖出建议

### 2. 单独评估各阶段

**熵持有判断**（是否仍在低熵有序态）：
```bash
./scripts/run_entropy_hold_judgement.sh \
  --symbol-or-name sh603507 \
  --start-date 20260301 \
  --exit-persist-days 3
```

**快速膨胀持有**（是否仍在动量膨胀期）：
```bash
./scripts/run_rapid_expansion_hold.sh \
  --symbol-or-name sh688268 \
  --start-date 20260301 \
  --exit-persist-days 2
```

**衰竭退出检测**（动量是否已耗尽）：
```bash
./scripts/run_rapid_expansion_exhaustion_exit.sh \
  --symbol-or-name sh688268 \
  --start-date 20260301 \
  --exit-persist-days 2
```

### 3. 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--scan-date` | 评估截止日 | 自动推断最新交易日 |
| `--lookback-years` | 特征回看年数 | 5 |
| `--exit-persist-days` | 连续多少天触发退出才确认 | 2-3 |

### 4. 解读结果

状态流转路径说明：
1. **entropy_hold** — 熵压缩态持有，价格有序收敛，继续持有
2. **rapid_expansion** — 进入快速膨胀期，动量爆发，继续持有
3. **exhaustion_exit** — 动量衰竭，建议卖出
4. **强制退出** — 连续 N 天触发退出信号

`exit_persist_days` 用于过滤噪声：要求连续多天满足退出条件才确认离场。
