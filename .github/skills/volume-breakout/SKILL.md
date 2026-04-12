---
name: volume-breakout
description: "运行量价突破策略。Use when: 量价突破, volume breakout, 放量突破, 缩量横盘, 布林带收窄, 波动率压缩, 量能放大, 突破检测。"
argument-hint: "扫描日期或股票代码，例如：20260310"
---

# 量价突破策略 (Volume Breakout)

低波动压缩后的放量突破检测：窄幅横盘（布林带收窄 + 低波动率）→ 成交量急剧放大 → 突破信号。

## 核心逻辑

### 压缩条件（必须同时满足）
- 20 日实现波动率 < 3.5%
- 20 日价格区间 (max-min)/mean < 25%
- 布林带宽度 < 14%

### 突破条件
- 5 日均量 / 20 日均量 > 1.2（量能放大）
- 当日量 / 20 日均量 > 阈值（单日放量）

## 执行步骤

### 运行回测

```bash
cd /nvme5/xtang/gp-workspace/gp-quant

python src/strategy/volume_breakout/backtest.py \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv
```

### 查看结果

输出到 `results/volume_breakout/`。

## 策略代码

```
src/strategy/volume_breakout/
├── config.py      # 配置（波动率/布林带/量比阈值）
├── detector.py    # 突破检测器
└── backtest.py    # 回测脚本
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_realized_vol_20` | 0.035 | 20 日波动率上限 |
| `max_price_range_20` | 0.25 | 20 日价格区间上限 |
| `max_bb_width` | 0.14 | 布林带宽度上限 |
| `min_vol_surge` | 1.2 | 5 日/20 日量比下限 |
