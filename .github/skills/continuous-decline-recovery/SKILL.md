---
name: continuous-decline-recovery
description: "运行连续下跌恢复买入策略。Use when: 连续下跌, 恢复买入, recovery buy, decline recovery, 抄底, 修复买点, 行业轮动, sector recovery, 跌后反弹, 板块修复, 市场修复。"
argument-hint: "扫描日期或股票代码，例如：20260310"
---

# 连续下跌恢复买入策略 (Continuous Decline Recovery)

核心理念：**不抄最低点，买最先修复的一段**。当市场经历连续下跌后，寻找率先恢复的行业和个股。

## 三层架构

| 层级 | 功能 |
|------|------|
| **市场层** | 判断是否真的发生过连续下跌 + 当前是否进入修复窗口 |
| **行业层** | 筛选"先受伤、再修复、相对市场更早转强"的行业 |
| **个股层** | 在恢复行业中选：受损过 + 正在恢复 + 早期窗口 + 未过热 |

## 市场状态机

```
no_setup → selloff → repair_watch → buy_window → rebound_crowded
```

只在 `buy_window` 状态时出信号。

## 执行步骤

### 1. 运行扫描

```bash
cd /nvme5/xtang/gp-workspace/gp-quant

# 扫描最新交易日
python src/strategy/continuous_decline_recovery/run_continuous_decline_recovery_scan.py \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv

# 指定日期
python src/strategy/continuous_decline_recovery/run_continuous_decline_recovery_scan.py \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv \
  --scan_date 20260310
```

### 2. 诊断模式

详细输出每层的判断细节：

```bash
python src/strategy/continuous_decline_recovery/run_continuous_decline_recovery_diagnostics.py \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv
```

### 3. 查看结果

输出到 `results/continuous_decline_recovery/`。

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 市场回望天数 | 6 | 检测连续下跌的窗口 |
| 最小行业成员数 | 4 | 行业至少有几只股票 |
| 最小反弹幅度 | 3% | 从低点至少反弹多少 |
| 最大反弹幅度 | 15% | 超过则认为已过热 |
| 前哨行业数 | 6 | 选几个领先恢复的行业 |

## 策略代码

```
src/strategy/continuous_decline_recovery/
├── continuous_decline_recovery_feature_engine.py   # 特征计算
├── continuous_decline_recovery_scan_service.py     # 扫描服务
├── continuous_decline_recovery_diagnostics.py      # 诊断
├── continuous_decline_recovery_report_writer.py    # 报告生成
├── run_continuous_decline_recovery_scan.py         # 扫描入口
└── run_continuous_decline_recovery_diagnostics.py  # 诊断入口
```

## Wiki 参考

- `wiki/entities/continuous-decline-recovery.md` — 实体详情
- `wiki/concepts/bifurcation.md` — 从熊市到恢复的分岔转变
