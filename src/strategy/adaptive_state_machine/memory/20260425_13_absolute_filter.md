# P&L 验证后发现：信号分布机械固定问题 + 绝对过滤建议

> 日期: 2026-04-25
> 前置: P&L 验证 Sharpe=2.16（优秀），但信号分布存在结构性问题

## 一、问题：信号比例完全由截面排名固定

每个日期的状态分布精确由股票总数决定，与模型预测无关：

| 日期 | 总数 | BREAKOUT | ACCUM | HOLD | COLLAPSE |
|------|------|----------|-------|------|----------|
| 20250102 | 5261 | 526 (10.0%) | 1052 (20.0%) | 2104 (40.0%) | 526 (10.0%) |
| 20260305 | 5358 | 536 (10.0%) | 1072 (20.0%) | 2142 (40.0%) | 536 (10.0%) |

### 根因

状态判定完全由两个截面百分位交叉决定：

```python
pct = sharpe_score 的截面排名 (0~1)
q50_pctile = q50 的截面排名 (0~1)

if pct > 0.9 and q50_pctile > 0.8:    # top 10% × top 20% ≈ 固定 10%
elif pct > 0.7 and q50_pctile > 0.5:   # 10~30% × top 50% ≈ 固定 20%
elif pct < 0.1 and q50_pctile < 0.2:   # bottom 10% × bottom 20% ≈ 固定 10%
```

`pct` 和 `q50_pctile` 高度相关（都来自同一模型输出），交叉条件几乎就是取固定百分位。

### 后果

1. 模型无法区分"今天有好机会" vs "今天没机会"（每天永远 ~536 个 BREAKOUT）
2. 系统性下跌期（如 2025-06）强行推荐 top 10%，是最差月份 -7.86% 的来源
3. 信号含义是"相对最不差"，不是"模型认为会涨"

### 对 Sharpe 的影响

Sharpe 2.16 说明排序能力真实存在（前 10% 确实跑赢），但固定选股数量会在熊市期间产生不必要的亏损交易。

## 二、建议改动：加入绝对信号过滤

在现有截面排名后加一层绝对过滤，让 BREAKOUT/ACCUMULATION 在熊市自然减少：

```python
# strategy.py — _predict_states_from_model()
# 现有:
if pct > 0.9 and q50_pctile > 0.8:
    state = StockState.BREAKOUT

# 建议:
if pct > 0.9 and q50_pctile > 0.8 and pred_return > 0:
    state = StockState.BREAKOUT
elif pct > 0.7 and q50_pctile > 0.5 and pred_return > 0:
    state = StockState.ACCUMULATION
```

`pred_return > 0` 是回归头（15% MSE loss）预测的绝对收益。虽然回归头不如分位数头准确，但作为"模型认为会涨"的最低门槛足够了。

### 预期效果

- 牛市：大部分股票 pred_return > 0，信号数量接近现有
- 熊市：多数股票 pred_return < 0，BREAKOUT/ACCUMULATION 自然减少甚至为 0
- 震荡市：信号数量随市场情绪自适应调整
- 避免 2025-06 类型的强制推荐亏损

### 改动范围

仅 `strategy.py` `_predict_states_from_model()` 中 2 个 if 条件加 `and pred_return > 0`，1~2 行代码。

## 三、优先级

此改动应在因子分组嵌入**之前**完成：
1. 改动极小（2 行）
2. 直接提升回测 Sharpe（消除熊市期间的强制亏损交易）
3. 让信号语义更合理（"会涨" vs "相对不差"）
