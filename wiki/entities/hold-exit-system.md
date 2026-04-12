---
title: 持有/退出决策系统
tags: [strategy, hold, exit, state-flow]
confidence: high
status: active
created: 2026-04-12
updated: 2026-04-12
---

# 持有/退出决策系统 (Hold-Exit System)

**源码**: `src/strategy/uptrend_hold_state_flow/`  
**运行**: `scripts/run_uptrend_hold_state_flow.sh`

## 功能

买入后的持仓管理。回答核心问题：**该不该继续持有？何时卖出？**

## 三个子模块

### 1. 熵储备持有判断 (Entropy Hold)
- 如果熵储备充足（低置换熵 + 高路径不可逆），继续持有
- 熵储备耗尽 → 准备退出

### 2. 快速膨胀持有 (Rapid Expansion Hold)
- 检测爆发性行情（成交量急剧放大 + 价格加速）
- 在爆发期间持有甚至加仓

### 3. 衰竭退出 (Exhaustion Exit)
- 检测爆发后的衰竭信号
- 成交量萎缩 + 涨幅收窄 + 熵上升 → 立即退出

## 状态流图

```
买入 → 正常持有 → 熵储备评估
                    ├── 储备充足 → 继续持有
                    └── 储备耗尽 → 退出
              → 快速膨胀
                    ├── 膨胀中 → 持有/加仓
                    └── 衰竭信号 → 立即退出
```

## 输入

- 入场日期 + 评估日期
- 日线 K 线数据

## 输出

- 状态路径历史
- 持有/卖出建议

## 相关实体

- [[tick-entropy-module]] — 熵储备计算
- [[four-layer-system]] — 买入信号来源
- [[multitimeframe-scanner]] — 买入信号来源

## 相关概念

- [[entropy]] — 熵储备的含义
- [[permutation-entropy]] — 通过置换熵判断趋势状态
- [[path-irreversibility]] — 主力是否还在
