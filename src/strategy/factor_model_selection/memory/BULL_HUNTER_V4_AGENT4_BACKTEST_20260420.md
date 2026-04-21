# Bull Hunter v6 — 3 层简化架构 + Agent 8 买入信号 (2026-04-21)

> v4 7-Agent → v6 3 层简化 + Agent 8 买入时机评估 + 因子自动调权
> 选股层(1+2+3+8) → 交易层(5+6) → 监控层(4+7 合并为统一复盘)
> Agent 8 与 Agent 6 对称: 买入时机因子(14个) ↔ 卖出时机因子(14个)
> 因子权重自动调整: Spearman 相关性 + 学习率平滑 (每 20 天复盘时执行)
> 公共逻辑: `_signal_common.py` 统一 Agent 6/8 的 IO / auto_adjust / 快照

---

## 零、Agent 完整交互图

### 总览: 3 层架构 + Agent 8 买入信号

> ℹ️ README 中有可渲染的 Mermaid 版本: [v3_bull_hunter/README.md](../v3_bull_hunter/README.md#核心思路)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   Bull Hunter v6 — 3 层架构 + Agent 8                            │
│                                                                                  │
│  ┌──────────────────────────── 数据层 ────────────────────────────┐              │
│  │  tushare-daily-full/  (日线 CSV)                                │              │
│  │  tushare_stock_basic.csv  (股票基本信息)                         │              │
│  │  feature-cache/daily/  (因子缓存)                               │              │
│  │  feature-cache/bull_models/  (训练模型)                          │              │
│  └─────────────────────────────────────────────────────────────────┘              │
│         │                                                                         │
│  ═══════╪═══════════════════════════════════════════════════════════════════       │
│  ║      ▼            选 股 层 (每日)                                    ║         │
│  ║                                                                      ║         │
│  ║  ╔═══════════════╗    全市场因子快照     ╔═══════════════╗           ║         │
│  ║  ║   Agent 1     ║ ───────────────────► ║   Agent 2     ║           ║         │
│  ║  ║  因子工程师    ║    (DataFrame)        ║  训练工程师    ║           ║         │
│  ║  ║               ║                       ║               ║           ║         │
│  ║  ║ • 日线因子x40 ║                       ║ • LightGBM    ║           ║         │
│  ║  ║ • 熵/动量/波动║                       ║ • 3目标分类器  ║           ║         │
│  ║  ║ • 资金流/量比 ║                       ║   (30/100/200)║           ║         │
│  ║  ╚═══════╤═══════╝                       ╚═══════╤═══════╝           ║         │
│  ║          │                                       │                   ║         │
│  ║          │ factor_snapshot                        │ model.pkl         ║         │
│  ║          ▼                                       ▼                   ║         │
│  ║  ╔═══════════════╗   候选 Top N           ╔═══════════════╗          ║         │
│  ║  ║   Agent 3     ║ ◄──── 加载模型 ─────── ║  bull_models/ ║          ║         │
│  ║  ║  预测选股      ║                        ║  (模型仓库)    ║          ║         │
│  ║  ║ • prob_200/100║                        ╚═══════════════╝          ║         │
│  ║  ║ • A/B/C 分级  ║                               ▲                  ║         │
│  ║  ╚═══════╤═══════╝                               │ retrain          ║         │
│  ║          │                                       │ (L1调参)         ║         │
│  ║          │ candidates                            │                   ║         │
│  ║          ▼                                       │                   ║         │
│  ║  ╔═══════════════════════════╗                   │                   ║         │
│  ║  ║   Agent 8 (NEW)          ║                    │                   ║         │
│  ║  ║  买入时机评估             ║                    │                   ║         │
│  ║  ║                           ║                    │                   ║         │
│  ║  ║ 14 个买入因子:            ║                    │                   ║         │
│  ║  ║  • 动量启动/加速度       ║                    │                   ║         │
│  ║  ║  • 波动率压缩/布林收窄   ║                    │                   ║         │
│  ║  ║  • 熵有序化/路径不可逆↑  ║                    │                   ║         │
│  ║  ║  • 资金流入加速/连续流入 ║                    │                   ║         │
│  ║  ║  • 站上均线/突破力度     ║                    │                   ║         │
│  ║  ║  • 量价协同/适度回调     ║                    │                   ║         │
│  ║  ║                           ║                    │                   ║         │
│  ║  ║ 规则引擎 → buy_quality   ║                    │                   ║         │
│  ║  ║ LightGBM 模型 (积累后)   ║                    │                   ║         │
│  ║  ║ 过滤: quality < 0.3 不买 ║                    │                   ║         │
│  ║  ╚═══════════╤═══════════════╝                    │                   ║         │
│  ║              │ filtered candidates                │                   ║         │
│  ║              │ (buy_quality >= min_threshold)      │                   ║         │
│  ═══════════════╪════════════════════════════════════╪═══════════════════          │
│                 │                                    │                             │
│  ═══════════════╪════════════════════════════════════╪═══════════════════          │
│  ║              ▼          交 易 层 (每日)           │                 ║          │
│  ║                                                   │                 ║          │
│  ║  ╔═══════════════════════════╗  ╔═══════════════╗ │                 ║          │
│  ║  ║   Agent 6                 ║  ║   Agent 5     ║ │                 ║          │
│  ║  ║  退出信号 (卖出时机)      ║  ║  组合管理      ║ │                 ║          │
│  ║  ║                           ║  ║               ║ │                 ║          │
│  ║  ║ 14 个卖出因子:            ║  ║ • 先卖后买    ║ │                 ║          │
│  ║  ║  • 动量衰减/减速度       ║  ║ • 止损/追踪   ║ │                 ║          │
│  ║  ║  • 波动率扩大/突刺       ║  ║ • 仓位控制    ║ │                 ║          │
│  ║  ║  • 熵无序化/不可逆崩塌   ║  ║ • min_quality ║ │                 ║          │
│  ║  ║  • 资金流出加速/连续流出 ║  ║   过滤        ║ │                 ║          │
│  ║  ║  • 跌破均线/ATR异常      ║  ╚═══════════════╝ │                 ║          │
│  ║  ║  • 盈利回撤/持有天数     ║        ▲            │                 ║          │
│  ║  ║                           ║        │sell_weight │                 ║          │
│  ║  ║ 规则引擎 → sell_weight   ║────────┘            │                 ║          │
│  ║  ║ LightGBM 模型 (积累后)   ║                     │                 ║          │
│  ║  ╚═══════════════════════════╝                     │                 ║          │
│  ║          ▲ 参数调整                    ▲ 参数调整   │                 ║          │
│  ║          │                             │            │                 ║          │
│  ═══════════╪═════════════════════════════╪════════════╪═════════════════          │
│             │                             │            │                            │
│  ═══════════╪═════════════════════════════╪════════════╪═════════════════          │
│  ║          │    监 控 层 (每 20 天统一复盘)           │                 ║          │
│  ║          │                             │            │                 ║          │
│  ║          │         ╔═══════════════════════════╗    │                 ║          │
│  ║          └─────────║     统一复盘              ║────┘                 ║          │
│  ║                    ║   run_unified_review()    ║                      ║          │
│  ║                    ╠═══════════════════════════╣                      ║          │
│  ║                    ║  选股评估 (原 Agent 4)    ║                      ║          │
│  ║                    ║   → selection_score       ║                      ║          │
│  ║                    ║  交易评估 (原 Agent 7)    ║                      ║          │
│  ║                    ║   → trading_score         ║                      ║          │
│  ║                    ║  unified = 0.5s + 0.5t    ║                      ║          │
│  ║                    ╚═══════════╤═══════════════╝                      ║          │
│  ║              ┌─────────────┬─────────────────┬─────────────────┐    ║          │
│  ║              ▼             ▼                  ▼                 ▼    ║          │
│  ║     选股 L1 调参   交易参数调整        Agent 6/8 重训    因子权重自调 ║          │
│  ║     → Agent 2      → Agent 5/6         → 买卖模型更新   → 6+8 权重  ║          │
│  ║                                                          Spearman相关║          │
│  ║                                                          lr=0.3平滑  ║          │
│  ═══════════════════════════════════════════════════════════════════════           │
│                                                                                    │
│  ┌──────────────── 辅助组件 ──────────────────────────────────────┐                │
│  │  Tracker:    tracking/{active,history}.csv (预测跟踪评估)       │                │
│  │  Portfolio:  portfolio/{positions,trades}.csv (持仓+交易记录)   │                │
│  │              portfolio/sell_weights/  (Agent 6 每日卖出权重)    │                │
│  │              portfolio/buy_quality/   (Agent 8 每日买入质量)    │                │
│  │              portfolio/buy_models/    (Agent 8 LightGBM 模型)   │                │
│  │  _signal_common.py:  Agent 6/8 共享的 IO + auto_adjust +       │                │
│  │              快照逻辑 (买/卖方向参数化, sign=+1/-1)             │                │
│  └────────────────────────────────────────────────────────────────┘                │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Agent 8 与 Agent 6 对称设计 (公共逻辑在 `_signal_common.py`)

```
                Agent 8 (买入时机)                    Agent 6 (卖出时机)
                ──────────────────                    ──────────────────
输入:           Agent 3 候选股                        Agent 5 已持仓
输出:           buy_quality (0~1)                     sell_weight (0~1)
因子数:         14 个买入因子                          14 个卖出因子
模型:           规则引擎 → LightGBM                   规则引擎 → LightGBM
训练标签:       买后20天涨≥5%=好买点                  卖后20天跌<5%=卖对了
持久化:       buy_quality/ + buy_models/            sell_weights/ + exit_models/
过滤:           quality < 0.3 不买                    weight > 0.6 触发卖
训练门槛:       ≥30 笔买入 (回测≥8)                  ≥50 笔卖出 (回测≥8)
权重调整:       auto_adjust_buy_weights()             auto_adjust_exit_weights()
                └─ 都委托给 _signal_common.auto_adjust_signal_weights(direction=...) ─┘
调整方法:       因子↑→买后涨=正相关=提权             因子↑→卖后跌=负相关=提权
                direction="buy", sign=+1              direction="sell", sign=-1
调整参数:       lr=0.3, clip=[0.01,0.25], ≥10样本    lr=0.3, clip=[0.01,0.25], ≥10样本

共享函数 (_signal_common.py):
  get_post_trade_return()      ← 买后/卖后 N 日收益计算
  load_recent_prices()         ← 加载最近 N 日价格
  load_recent_volumes()        ← 加载最近 N 日成交量
  load_rule_weights()          ← 加载权重 JSON
  save_rule_weights_to()       ← 保存权重 JSON
  apply_model_override()       ← LightGBM 覆盖规则引擎
  save_daily_snapshot()        ← 每日因子快照 CSV
  auto_adjust_signal_weights() ← Spearman 自适应调权 (核心)

因子对称:
  动量启动     ↔  动量衰减         (方向相反)
  动量加速     ↔  动量减速         (方向相反)
  波动率压缩   ↔  波动率扩大       (方向相反)
  布林带收窄   ↔  波动率突刺       (不同信号)
  熵有序化     ↔  熵无序化         (方向相反)
  路径不可逆↑  ↔  路径不可逆崩塌   (方向相反)
  资金流入加速 ↔  资金流出加速     (方向相反)
  连续净流入   ↔  连续净流出       (方向相反)
  站上MA20     ↔  跌破MA20         (方向相反)
  站上MA60     ↔  跌破MA60         (方向相反)
  突破力度     ↔  ATR异常          (不同角度)
  量价协同     ↔  盈利回撤         (不同角度)
  成交量放大   ↔  涨幅vs目标       (不同角度)
  适度回调     ↔  持有天数归一化   (不同角度)
```

### Agent 交互矩阵 (v6 + Agent 8)

```
发出方 →    Agent 1    Agent 2    Agent 3    Agent 8    Agent 5    Agent 6    统一复盘
接收方 ↓   (因子)     (训练)     (预测)    (买入信号)  (组合)     (退出)     (4+7合并)
───────────────────────────────────────────────────────────────────────────────────────
Agent 1      —
Agent 2    snapshot     —                                                  L1调参指令
Agent 3               model_dir    —
Agent 8    snapshot               candidates    —
Agent 5                                      filtered     —      sell_wt   参数调整
Agent 6    snapshot                                      trades    —       参数调整
统一复盘   snapshot              predictions   quality   trades   accuracy    —
```

### 每日 Live 循环时序 (v6 + Agent 8)

```
  时间轴 →

  Agent 1        Agent 3        Agent 8         Agent 6        Agent 5
    │              │              │               │              │
    │─ 因子快照 ─►│              │               │              │
    │  snapshot    │              │               │              │
    │              │              │               │              │
    │              │─ Top N ────►│               │              │
    │              │  candidates  │               │              │
    │              │              │               │              │
    │──── snapshot ────────────►│               │              │
    │              │              │               │              │
    │              │              │─ filtered ──►│              │
    │              │              │  (quality≥0.3)│              │
    │              │              │               │              │
    │              │              │               │─sell_weight─►│
    │              │              │               │  对已持仓    │
    │              │              │               │              │
    │              │              │               │              │─ 先卖后买
    │              │              │               │              │  完成交易
    ▼              ▼              ▼               ▼              ▼
```

### 回测调度流程 (v6 + Agent 8)

```
 ┌─────────── run_live_backtest() v6 ──────────────┐
 │  for day_i, date in enumerate(trading_days):      │
 │                                                    │
 │  ① 模型轮换                                       │
 │     get_model_for_date(scan_date)                  │
 │                                                    │
 │  ② Agent 2 周期重训 (按需)                         │
 │                                                    │
 │  ③ Agent 6 退出模型训练 (每 30 天, ≥8 笔卖出)     │
 │                                                    │
 │  ④ Agent 8 买入模型训练 (每 30 天, ≥8 笔买入)     │
 │                                                    │
 │  ⑤ run_live(date) — 每日完整循环                   │
 │     选股层: Agent 1 → Agent 3 → Agent 8 (过滤)    │
 │     交易层: Agent 6 → Agent 5                      │
 │                                                    │
 │  ⑥ 统一复盘 (每 20 天)                             │
 │     选股评估 + 交易评估 → unified_score            │
 │     → L1 调参 / 参数调整 / 模型重训                │
 │     → auto_adjust_exit_weights()  (Agent 6 调权)  │
 │     → auto_adjust_buy_weights()   (Agent 8 调权)  │
 └────────────────────────────────────────────────────┘
```

## 一、改动背景

此前 `run_live_backtest()` 只运行 Agent 1→3→6→5→7，Agent 4 (选股纠正监控) 未参与回测。
本次将 Agent 4 整合进回测循环，每 20 天执行一次，形成完整的双回路反馈闭环。

## 二、回测调度流程 (v6 + Agent 8)

```
 ┌─────────── run_live_backtest() v6 ──────────────┐
 │  for day_i, date in enumerate(trading_days):      │
 │                                                    │
 │  ① 模型轮换                                       │
 │     get_model_for_date(scan_date)                  │
 │     └→ 选 scan_date 前最新可加载 lgbm 模型        │
 │                                                    │
 │  ② Agent 2 周期重训 (retrain_interval 天)          │
 │     └→ 本次回测设为 0 (禁用, 仅靠统一复盘触发)    │
 │                                                    │
 │  ③ Agent 6 退出模型训练 (每 30 天, ≥8 笔卖出)     │
 │     └→ LightGBM 二分类 (该卖/不卖)                │
 │                                                    │
 │  ④ Agent 8 买入模型训练 (每 30 天, ≥8 笔买入) NEW │
 │     └→ LightGBM 二分类 (好买点/差买点)            │
 │                                                    │
 │  ⑤ run_live(date) — 每日完整循环                   │
 │     选股层: Agent 1 → Agent 3 → Agent 8 (过滤)    │
 │     交易层: Agent 6 → Agent 5                      │
 │                                                    │
 │  ⑥ 统一复盘 (每 monitor_interval=20 天)            │
 │     run_unified_review():                          │
 │       选股评估 (原 Agent 4) → selection_score      │
 │       交易评估 (原 Agent 7) → trading_score        │
 │       unified_score = 0.5*选股 + 0.5*交易          │
 │     └→ retrain_required? → Agent 2 带 L1 调参      │
 │     └→ trading_directives? → Agent 5/6 参数调整    │
 │     └→ signal_quality差? → Agent 6/8 模型重训      │
 │     └→ auto_adjust_exit_weights() → Agent 6 调权   │
 │     └→ auto_adjust_buy_weights()  → Agent 8 调权   │
 └────────────────────────────────────────────────────┘
```

## 三、统一复盘闭环

```
                   ┌─────────────────────┐
                   │     统一复盘         │ ← 每 20 天
                   │ run_unified_review() │
                   └──────────┬──────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                                         ▼
   选股质量评估                              交易质量评估
   (原 Agent 4)                              (原 Agent 7)
   ┌─────────────────┐                 ┌─────────────────┐
   │ 回路A: 漏选诊断  │                 │ 买入质量 (命中率)│
   │ 回路B: 跟踪反馈  │                 │ 卖出时机 (持仓)  │
   │                  │                 │ 信号相关性       │
   │ selection_score  │                 │ trading_score    │
   │ = (1-miss)*0.4   │                 │ = win_rate*0.6   │
   │ + track_win*0.6  │                 │ + buy_qual*0.4   │
   └────────┬─────────┘                 └────────┬─────────┘
            │                                     │
            └──────────┬──────────────────────────┘
                       │
              unified_score = 0.5 * 选股 + 0.5 * 交易
                       │
         ┌─────────────┼──────────────────┬────────────────┐
         ▼             ▼                  ▼                ▼
  选股 L1 调参   交易参数调整        Agent 6/8 重训   因子权重自调
  → Agent 2      → Agent 5/6        → 买卖模型更新   → Agent 6/8
    adjust_*       止损/阈值          (信号差时)       auto_adjust
                                                       Spearman相关
                                                       lr=0.3 平滑
```

### 因子权重自动调整机制 (实现: `_signal_common.auto_adjust_signal_weights()`)

```
auto_adjust_signal_weights(direction, factors, default_weights, ...)
────────────────────────────────────────────────────────
  ↑ 被 Agent 6 (direction="sell") 和 Agent 8 (direction="buy") 分别调用
  ↑ 两者的差异仅在方向参数 (sign=+1/-1) 和文件名/目录前缀────────

  ① 收集样本
     Agent 8: buy_quality/*.csv 中因子快照 + 买后 20 天收益
     Agent 6: sell_weights/*.csv 中因子快照 + 卖后 20 天收益
     最少 10 个有效样本才调整

  ② 计算因子有效性 (Spearman rank 相关性)
     Agent 8: corr(因子, 买后收益)   正相关 → 因子有效
     Agent 6: corr(因子, 卖后收益)   负相关 → 因子有效 (取 -corr)

  ③ 相关性 → 目标权重
     clip 到 [-0.5, 0.5] → 平移到 [0, 1] → 归一化
     clip 到 [0.01, 0.25] → 再归一化

  ④ 平滑混合 (防止剧烈变化)
     new_w = old_w × 0.7 + target_w × 0.3   (lr = 0.3)

  ⑤ 持久化
     Agent 8 → buy_models/buy_rule_weights.json
     Agent 6 → exit_models/rule_weights.json
     下次 run 自动加载

  示例 (Agent 8, 12 样本):
    momentum_ignition:  corr=+1.000  权重 0.100 → 0.109 (⬆ 有效)
    vol_compression:    corr=-1.000  权重 0.100 → 0.073 (⬇ 反指)

  示例 (Agent 6, 12 样本):
    momentum_decay:     corr=-1.000  权重 0.120 → 0.123 (⬆ 有效)
    vol_expansion:      corr=+1.000  权重 0.080 → 0.059 (⬇ 反指)
```

## 四、代码改动 (5 项)

### 4.1 Agent 4 整合 — `pipeline.py`

`run_live()` 新增返回 `factor_snapshot`:
```python
return {
    "predictions": predictions,
    "factor_snapshot": snapshot,  # ← 新增, 供 Agent 4 使用
    ...
}
```

`run_live_backtest()` 新增参数和逻辑:
```python
def run_live_backtest(
    ...,
    monitor_interval: int = 20,  # ← 新增: Agent 4 每 20 天触发
):
    # 每 monitor_interval 天:
    #   1. 调用 run_monitor() 获取诊断
    #   2. 检查 tuning_directives.retrain_required
    #   3. 若需要 → _apply_tuning_directives() + run_training()
    #   4. 保存健康报告到 agent4_reports/
```

### 4.2 Bug #1: `_retrain` 后缀目录不可见 — `agent2_train.py`

**问题**: `d.isdigit()` 对 `20250103_retrain` 返回 False，Agent 4 重训的模型被忽略
**修复**: 新增 `_extract_date()` 按 `_` 分割取首段
```python
def _extract_date(dirname: str) -> str | None:
    base = dirname.split("_")[0]
    if base.isdigit() and len(base) == 8:
        return base
    return None
```

### 4.3 Bug #2: xgboost 模型不可加载 — `agent2_train.py`

**问题**: 20250829+ 模型用 xgboost 训练，环境未安装 xgboost，`pickle.load()` 失败
**修复**: 新增 `_is_model_loadable()`
```python
def _is_model_loadable(model_dir: str) -> bool:
    # 读 meta.json → model_type
    # xgboost + ImportError → return False
```

`get_model_for_date()` 从最新往回遍历，跳过不兼容模型:
```python
for _, dirname in reversed(candidates):
    if _is_model_loadable(full_path):
        return full_path
    else:
        logger.warning(f"跳过不兼容模型: {dirname}")
```

### 4.4 Bug #3: 模型清理过度 — `agent2_train.py`

**问题**: `MAX_MODEL_VERSIONS=8`, Agent 4 频繁重训导致旧基础模型被删
**修复**: `MAX_MODEL_VERSIONS = 20`

### 4.5 CLI 新增 — `run_bull_hunter.py`

```python
parser.add_argument("--monitor_interval", type=int, default=20)
```

## 五、回测结果对比

### 回测命令
```bash
python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
    --live-backtest --start_date 20250101 --end_date 20251230 \
    --retrain_interval 0 --monitor_interval 20
```

### v2 vs v3 对比 (2025-01-01 ~ 2025-12-30)

| 指标 | v2 (有 xgboost bug) | v3 (修复后) | 变化 |
|------|---------------------|-------------|------|
| 交易笔数 | 10 | **13** | +3 |
| 胜率 | 30% | **46.2%** | +16pp |
| 平均盈亏 | +2.5% | **+15.6%** | +13pp |
| 总盈亏 | +57,924 | **+349,873** | **6x** |
| 最佳交易 | — | +122.5% | — |
| 最差交易 | — | -20.3% | — |
| 平均持有 | — | 75 天 | — |
| Agent 4 重训 | 8 次 | **12 次** | +4 |
| Agent 6 | AUC=0.857 | 已训练 | ✅ |
| xgboost 错误天数 | **82 天** (全部失败) | **0** | 修复 |

### 关键发现

1. **xgboost 跳过验证**: scan_date `20250901` 使用模型 `20250811` (lgbm)，
   正确跳过了 `20250829` (xgboost)，v2 就是在这个点开始失败的
2. **Agent 4 持续重训**: 12 次重训生成的 lgbm `_retrain` 模型覆盖了整个回测周期，
   xgboost 模型实际从未被选中 (Agent 4 的重训模型日期更近)
3. **Agent 7 最终状态 healthy**: 最后一天 `Supervisor=healthy`，
   说明系统在全年运行中自我修正到了稳定状态

### 模型轮换时间线

```
Day 1-2:     20250101 (基础 lgbm)
Day 3-22:    20250103_retrain (Agent 4 第1次)
Day 23-41:   20250210_retrain (Agent 4 第2次)
Day 42-62:   20250310_retrain (Agent 4 第3次)
Day 63-82:   20250409_retrain (Agent 4 第4次)
Day 83-102:  20250509_retrain (Agent 4 第5次)
Day 103-122: 20250609_retrain (Agent 4 第6次)
Day 123-142: 20250707_retrain (Agent 4 第7次)
Day 143-162: 20250804_retrain (Agent 4 第8次)
Day 163-202: 20250901_retrain (Agent 4 第9次)
Day 203-222: 20251104_retrain (Agent 4 第10次)
Day 223-241: 20251202_retrain (Agent 4 第11次)
Day 242:     20251230_retrain (Agent 4 第12次, 最后一天)
```

## 六、结果文件

```
results/bull_hunter/live_backtest_20250101_20251230/
├── portfolio/
│   ├── positions.csv
│   ├── trades.csv
│   └── exit_model/
├── tracking/
│   ├── active.csv
│   └── history.csv
├── review_reports/              ← v6: 统一复盘报告 (原 agent4_reports/)
│   ├── review_20250120.json
│   ├── review_20250217.json
│   └── ...
└── live_backtest_summary.json
```
