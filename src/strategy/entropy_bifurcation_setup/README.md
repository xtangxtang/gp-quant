# Entropy Bifurcation Setup

## 描述

基于日线交易数据构建的低熵压缩 + 分叉启动策略。

这套策略把信号拆成三层：

- `entropy_quality`：当前是否处于低熵有序压缩态
- `bifurcation_quality`：旧平衡是否开始失稳，是否接近临界切换
- `trigger_quality`：是否已经出现向上突破和资金确认

核心字段包括：

- `perm_entropy_20_norm` / `perm_entropy_60_norm`
- `entropy_gap`
- `entropy_percentile_120`
- `ar1_20` / `recovery_rate_20`
- `var_lift_10_20`
- `breakout_10` / `breakout_20`
- `volume_impulse_5_20` / `flow_impulse_5_20`

## 信号逻辑

策略默认要求：

- 当前短周期排列熵处于过去 120 日低位
- 短周期比中周期更有序
- `ar1_20` 较高，恢复率较低，说明系统接近临界点
- 日线开始向上突破
- 股价站上 20 日线，且 20 日线不弱于 60 日线

## 运行

```bash
python src/strategy/entropy_bifurcation_setup/run_entropy_bifurcation_scan.py \
    --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
    --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv \
    --out_dir results/entropy_bifurcation_setup
```

## 主要参数

- `scan_date`：扫描日，默认自动推断最新交易日
- `top_n`：候选与组合容量上限
- `min_amount` / `min_turnover`：流动性过滤
- `backtest_start_date` / `backtest_end_date` / `hold_days`：前瞻回测窗口
