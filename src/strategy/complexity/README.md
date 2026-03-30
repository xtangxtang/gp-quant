# Complexity Strategies

## 描述

一句话概括：把复杂系统笔记里的“熵、分岔、自组织、分形、能量流”压成四类可执行的日线选股策略。

基于 `complexity_theory_notes.md` 落地的四类日线策略集合：

- `compression_breakout`：低熵压缩后的放量突破
- `self_organized_trend`：资金持续注入后的趋势跟随
- `fractal_pullback`：大级别趋势中的分形回踩续涨
- `market_energy_flow`：横截面资金能量流与相对强度领先

| 策略 | 主要抓手 | 必须满足的直观条件 | 打分时重点看什么 |
| --- | --- | --- | --- |
| `compression_breakout` | 先压缩后突破 | 最近一段时间波动明显收敛；当前有效突破近 20 日高点；成交额和主力资金不能太弱；股价站上 20 日线且 20 日线在 60 日线上方 | 压缩是否充分、突破是否有力度、资金能量是否跟上、结构是否开始转顺 |
| `self_organized_trend` | 趋势已经形成 | 收盘价在 20 日线上方；20 日线在 60 日线上方，60 日线在 120 日线上方；20 日和 60 日涨幅足够；趋势斜率持续向上 | 趋势强度、均线和区间结构是否一致、能量项和相位项是否仍支持趋势延续、离阶段高点有多近 |
| `fractal_pullback` | 上升趋势里的二次启动 | 先有明确母趋势；最近 3% 到 12% 的适中回踩；回踩过程缩量且波动没有失控；重新站回短整理高点；收盘仍在 20 日线上方 | 母趋势质量、回踩深度是否健康、缩量是否充分、再启动是否明确、回踩后资金是否还在 |
| `market_energy_flow` | 横截面资金流聚集 | 主力净流入、成交额、换手率都不弱；最近 20 日已有领先收益；股价在 20 日线上方；不仅个股强，还要求行业里同时出现多只能量增强个股 | 能量脉冲强不强、个股是不是价格领先者、换手是否活跃、所在行业是否出现共振 |

| 策略 | 更适合的市场环境 | 最容易失效的情况 |
| --- | --- | --- |
| `compression_breakout` | 震荡末端转强、个股开始脱离盘整区间 | 假突破很多的高噪声震荡市 |
| `self_organized_trend` | 主线明确、强者恒强、趋势持续扩散 | 情绪高潮末端或趋势衰竭阶段 |
| `fractal_pullback` | 主升浪中的中继整理和二次上车窗口 | 母趋势本身不成立，所谓回踩其实是转弱开端 |
| `market_energy_flow` | 板块轮动清晰、资金集中进攻某些行业或主题 | 资金四散、没有主线、单点异动但行业不跟 |

## 运行

```bash
python src/strategy/complexity/run_complexity_scan.py \
    --strategy_name compression_breakout \
    --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
    --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv \
    --out_dir results/complexity/compression_breakout
```

## 主要参数

- `strategy_name`：四选一
- `scan_date`：扫描日，默认自动推断最新交易日
- `top_n`：候选与组合容量上限
- `min_amount` / `min_turnover`：流动性过滤
- `backtest_start_date` / `backtest_end_date` / `hold_days`：前瞻回测窗口

## 输出

每个策略会输出：

- 全市场扫描快照
- 全量候选池
- TopN 候选池
- 组合入选结果
- 前瞻回测逐日结果
- 前瞻回测逐笔交易
- 前瞻回测汇总
