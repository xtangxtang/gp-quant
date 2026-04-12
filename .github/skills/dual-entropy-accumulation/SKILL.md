---
name: dual-entropy-accumulation
description: "运行双熵共振策略（日线压缩+日内集中化）。Use when: 双熵, dual entropy, 日线压缩, 日内集中, 吸筹, accumulation, 悄然建仓, stealth, 派发检测, distribution, 买卖信号, 熵扩散, entropy diffusion。"
argument-hint: "操作模式和参数，例如：买入扫描、卖出检测 sh600000"
---

# 双熵共振策略 (Dual Entropy Accumulation)

日线熵压缩 + 日内成交集中化 → 检测"悄然吸筹"信号。同时提供卖出信号检测（熵扩散 + 暗中派发）。

## 核心逻辑

### 买入信号（双熵共振）
1. **日线压缩态**: 排列熵百分位 < 40%，绝对值 < 0.85，短期比长期更有序
2. **日内集中化**: 成交量熵 < 0.88，路径不可逆 > 0.01，振幅 < 6%
3. **辅助确认**: 有效分钟占比 > 70%，量能集中度 > 0.008

### 卖出信号
1. **熵扩散**: 日线排列熵加速上升 → 结构瓦解
2. **暗中派发**: 日内成交分散化 + 尾盘比早盘更无序
3. **衰竭**: 日线 PE > 0.92 或日内 PE > 0.96
4. **量能异常**: 成交模式异变

### 得分权重

买入: 日线压缩 30% + 日内集中 30% + 方向性 20% + 量能形态 20%  
卖出: 熵扩散 30% + 暗中派发 35% + 衰竭 20% + 量能异常 15%

## 执行步骤

### 1. 买入信号扫描

```bash
cd /nvme5/xtang/gp-workspace/gp-quant

# 默认扫描（最近交易日，前 200 只）
python -m src.strategy.dual_entropy_accumulation.run_scan

# 指定参数
python -m src.strategy.dual_entropy_accumulation.run_scan \
  --daily_data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --minute_data_dir /nvme5/xtang/gp-workspace/gp-data/trade \
  --scan_date 2026-04-03 \
  --max_stocks 100 \
  --workers 8
```

### 2. 卖出信号扫描（已持仓检测）

```bash
python -m src.strategy.dual_entropy_accumulation.run_scan \
  --mode sell \
  --watchlist sh600000,sz000001,sh600036
```

### 3. 回测

```bash
python src/strategy/dual_entropy_accumulation/backtest.py
```

### 4. 查看结果

输出到 `results/dual_entropy/`。

## 策略代码

```
src/strategy/dual_entropy_accumulation/
├── config.py            # 全部配置（日线/日内/融合/卖出/扫描器）
├── daily_entropy.py     # 日线熵计算（PE20/PE60/路径不可逆/主特征值）
├── intraday_entropy.py  # 日内熵计算（分钟级滚动窗口）
├── fusion_signal.py     # 双熵融合买入信号
├── sell_signal.py       # 卖出信号检测
├── bifurcation.py       # 分岔检测
├── scanner.py           # 扫描器
├── run_scan.py          # CLI 入口
└── backtest.py          # 回测
```

## 关键阈值

| 参数 | 买入值 | 卖出值 |
|------|--------|--------|
| 日线排列熵百分位 | < 0.40 | 穿越 0.70 |
| 日线排列熵绝对值 | < 0.85 | > 0.92 |
| 日内成交量熵 | < 0.88 | 上升 > 0.02 |
| 日内路径不可逆 | > 0.01 | 下降 > 0.005 |
| 买入/卖出得分 | > 0.55 | > 0.50 |
| 观察/预警得分 | > 0.40 | > 0.35 |

## 数据需求

- 日线 K 线: `tushare-daily-full/`（至少 60 天）
- 分钟数据: `trade/<symbol>/`（最近 10 天）

## Wiki 参考

- `wiki/concepts/entropy.md` — 熵的理论基础
- `wiki/concepts/permutation-entropy.md` — 置换熵详解
- `wiki/concepts/path-irreversibility.md` — 路径不可逆性
- `wiki/decisions/why-daily-not-minute.md` — 日线 vs 分钟线选择
