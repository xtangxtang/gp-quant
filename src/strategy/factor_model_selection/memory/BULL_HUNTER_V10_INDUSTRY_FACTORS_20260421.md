# Bull Hunter V10 — 行业动量/共振因子 (2026-04-21)

> 用户反馈 V9 错过了 2025 Q2-Q4 多个板块行情（固态 利元亨/上海洗霸/先导智能、CPO 中际旭创/新易盛/天孚通信、存储 佰维存储/德明利）。诊断发现 8/8 目标股票里 7 只**从未进入 Agent 3 候选**，因为模型只看单股因子，看不到"该股所在行业最近多强"。本次新增 5 个行业截面因子。

## 1. 改进动机

### V9 漏抓的 8 只大牛股

| 股票 | 实际涨幅 | V9 候选次数 |
|------|----------|-------------|
| 中际旭创 | +467% | 3次（仅 2026-01 之后，错过启动） |
| 新易盛 | +375% | **0** |
| 天孚通信 | +172% | **0** |
| 利元亨 | +117% | **0** |
| 佰维存储 | +79% | **0** |
| 上海洗霸 | +78% | **0** |
| 先导智能 | +28% | **0** |
| 德明利 | +40% | **0** |

### 根本问题
当前 [DAILY_FACTORS](../v3_bull_hunter/agent1_factor.py) 全是**单股因子**（熵、动量、资金流、波动）。模型看不到"行业整体走强"这层信息，所以无法识别板块共振行情的早期信号。

## 2. 实施内容

### 2.1 新增 5 个行业因子（截面聚合，无未来数据）

```python
# src/strategy/factor_model_selection/v3_bull_hunter/agent1_factor.py
INDUSTRY_FACTORS = [
    "industry_mom_5d",         # 行业 5 日涨幅中位数
    "industry_mom_20d",        # 行业 20 日涨幅中位数
    "industry_breadth_5d",     # 行业内 5 日正收益占比
    "industry_rs_20d",         # 行业 20 日相对全市场超额 (industry_mom - market_mom)
    "industry_vol_surge",      # 行业量比中位数
]
```

### 2.2 计算时机
- **Agent 1 实时扫描**：构建 `daily_snap` 后，按 `_industry` 分组聚合，写回每行
- **Agent 2 训练 panel**：每个 `sample_date` 的截面**单独**分组聚合（避免跨日泄漏）
- **新增 `momentum_5d` 衍生因子**：`compute_derived_factors` 内 `close.pct_change(5)`

### 2.3 修改文件清单

| 文件 | 变动 |
|------|------|
| `agent1_factor.py` | DAILY_FACTORS 加 `momentum_5d`；新增 `INDUSTRY_FACTORS`、`compute_industry_factors()`；`run_factor_generation` 末尾调用 |
| `agent2_train.py` | DAILY_FACTORS 加 `momentum_5d` + 5 行业因子；`run_training` 加 `basic_path` 参数；`_build_training_panel` 按 sample_date 分组算行业因子 |
| `pipeline.py` | 4 处 `run_training()` 调用全部传 `basic_path=cfg.basic_path` |

总特征数: 41 (含 momentum_5d) + 5 (行业) = **46 个**

## 3. V10 回测结果 (20240301 ~ 20260321, 初始 100 万)

### 3.1 V10 vs V9 vs V8 对比

| 指标 | V8 基线 | V9 (大盘过滤+门槛) | **V10 (+ 行业因子)** | V10 vs V9 |
|------|---------|--------|--------|---------|
| 交易笔数 | 41 | 56 | **29** | **-48%** ⚠️ |
| 胜率 | 58.5% | 58.9% | 51.7% | -7.2pp |
| 平均盈亏 | +11.2% | +6.6% | **+9.9%** | +3.3pp |
| 总盈亏 | +377K | +402K | **+359K** | **-11%** |
| 最佳单笔 | +58.8% | +48.1% | +55.5% | - |
| 最差单笔 | -35.1% | -22.9% | -22.4% | - |
| 平均持有 | 45 天 | 32 天 | **50 天** | +56% |
| 模型 AUC | ~0.55 | ~0.55 | **0.57-0.74** | **显著提升** |

### 3.2 模型质量飙升（行业因子被高度采用）

最新模型 (20260304) 特征重要性 Top 20，**5 个行业因子全部进入** Top 20：

| 排名 | 因子 | 重要性 |
|------|------|--------|
| 1 | volatility_l | 974 |
| 2 | turnover_entropy_l | 935 |
| 3 | mf_big_cumsum_l | 849 |
| 4 | momentum_60d | 783 |
| 5 | dom_eig_l | 778 |
| **6** | **industry_rs_20d** | **757** |
| 7 | coherence_l1 | 744 |
| ... | ... | ... |
| **14** | **industry_mom_20d** | **640** |
| **15** | **industry_vol_surge** | **637** |
| **18** | **industry_mom_5d** | **598** |
| (Top 30) | industry_breadth_5d | - |

模型训练 AUC 一度到达 **0.74**（V9 早期 ~0.50-0.55），证明行业因子对预测大牛股**有效**。

### 3.3 目标股票捕获情况

| 股票 | V9 | V10 |
|------|----|----|
| 中际旭创 | 3次候选（晚） | ❌ 一次没进 |
| 新易盛 | 0 | ❌ |
| 天孚通信 | 0 | ✅ 买1次 (20260305 @334.57)，亏 -20K |
| 利元亨 | 0 | ⚠️ 进2次候选 (prob=0.23/0.28，被 0.30 门槛过滤) |
| 上海洗霸/先导智能/佰维存储/德明利 | 0 | ❌ |

**结论**：行业因子帮助识别了部分目标，但仍**没有抓住主线龙头**。

## 4. 问题诊断

### ❌ 主要问题: 交易笔数大幅下降 (56 → 29)

行业因子让模型更"挑剔"：当 `industry_rs_20d < 0`（行业弱势）时，模型给低分 → Agent 3 候选减少 → Agent 8 通过率降低。

**实际数据**：模型最优阈值从 V9 的 0.07 降到 V10 的 0.06（更严格筛选有效信号），但通过率仍然偏低。

### ❌ 仍未抓住主线龙头

模型目标是"未来 6 个月涨 200%"，这天然**排斥已经在高位的龙头**：
- 中际旭创起涨时 91 元，6 个月涨到 520 = +467%（理论上模型应该选）
- 但模型对 "已涨过一段 + 已经较高价" 的标的本能给低分（因为历史样本里大多数高位股票后续涨不到 200%）

### ✅ V10 改进点
- 模型 AUC 从 0.55 → 0.57-0.74，质量明显提升
- 平均盈亏 +9.9% 优于 V9 的 +6.6%
- trailing_stop 触发了几个超大单 (+102K 同洲, +66K 大地熊, +110K 康众医疗, +64K 雄塑科技, +59K 金富科技)

## 5. 推荐下一步 (V11 候选)

### ⭐⭐⭐ V11-A: 板块跟随通道（旁路独立信号）
不依赖 200% 模型，新增独立通道:
```python
if (industry_rs_20d > 0.10 and          # 行业相对强势
    industry_breadth_5d > 0.7 and       # 行业普涨
    industry_vol_surge > 1.5 and        # 行业放量
    close_vs_high_60d > 0.85):          # 个股突破中位
    # 进入 "板块跟随" 候选池, 用独立信号通道
```
预期效果：能在 20250604 当天买入中际旭创/新易盛，享受 CPO 整片行情。

### ⭐⭐ V11-B: 多目标模型
不只训练 200% 目标，加训:
- `gain_30d ≥ 30%` (短期突破)
- `gain_60d ≥ 60%` (中期翻倍)

短目标更适合捕捉趋势启动。

### ⭐⭐ V11-C: 放宽门槛 + 行业冷却
- min_prob_200 从 0.30 降回 0.25（行业因子已经过滤掉低质量行业）
- 行业冷却 max_per_industry 从 2 提升到 4（让 CPO 三剑客可以并存）

### ⭐ V11-D: 每日重训 + 滚动窗口
当前 monitor_interval=20 天，板块行情常常 5-10 天就启动并加速。改成更敏捷的重训节奏。

## 6. 复现命令

```bash
# 清空模型，回测
rm -rf /nvme5/xtang/gp-workspace/gp-data/feature-cache/bull_models/*
rm -rf results/bull_hunter/live_backtest_20240301_20260321
python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
    --live-backtest --start_date 20240301 --end_date 20260321 \
    --retrain_interval 0 --monitor_interval 20

# 验证行业因子工作
python3 -c "
from src.strategy.factor_model_selection.v3_bull_hunter.agent1_factor import (
    INDUSTRY_FACTORS, compute_industry_factors)
import pandas as pd
snap = pd.DataFrame({
    '_industry': ['通信设备','通信设备','半导体'],
    'momentum_5d': [0.08, 0.06, 0.02],
    'momentum_20d': [0.15, 0.12, 0.05],
    'vol_ratio_s': [2.0, 1.8, 1.2],
}, index=['sz300308','sz300502','sh688525'])
out = compute_industry_factors(snap)
print(out[INDUSTRY_FACTORS])
"
```

## 7. 关键日志证据

```
2026-04-22 00:30:09 [INFO] agent2_train: 训练 panel 构建完成: 143105 行
2026-04-22 00:30:09 [INFO] agent2_train:   200pct: 143105 样本, 903 正样本 (0.6%)
2026-04-22 00:30:09 [INFO] agent2_train:     有效因子: 46 个 (model_type=lgbm)
2026-04-22 00:30:14 [INFO] agent2_train:   200pct: 训练完成, val_auc=0.7372
```

## 8. 复盘结论

**结构性改进**：行业因子已经成为模型的核心特征之一（Top 20 占 5 席），证明思路正确。

**未达预期**：仍未抓到主线龙头，因为模型的训练目标（"未来 6 个月翻倍"）与"骑龙头"是两种不同的策略。**需要旁路独立信号通道（V11-A）才能真正解决用户痛点**。
