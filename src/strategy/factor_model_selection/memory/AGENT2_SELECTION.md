# Agent 2: 因子状态机选股 (v2)

> 代码: `agent_selection.py` → `run_selection()`, 配置 `StateConfig`
> **v2 重构**: 从 LightGBM 回归替换为三阶段因子状态机 (旧版备份: `agent_selection_lgb.py`)

---

## 职责

接收 Agent 1 的因子截面, 用**因子状态机**判断每只股票当前所处阶段 (蓄力/突破/空闲),
优先选取突破股, 不足时补充蓄力股。**所有 horizon 共用同一选股结果** (状态机不区分持有周期)。

## 输入 / 输出

| 项目 | 内容 |
|------|------|
| **输入** | `factor_snapshot` (Agent 1), `cache_dir`, `scan_date`, `top_n`, `horizons` |
| **输出** | `dict[horizon, DataFrame]` — columns: symbol, name, industry, phase, accum_quality, breakout_quality, composite_score, rank |

## 核心架构: 三阶段因子状态机

```
Phase 1: 蓄力吸筹 (Accumulation)
  6 个因子条件 → 满足 ≥3 个 → 连续 ≥3 天 = 蓄力确认
  ↓ (近 10 天有蓄力)
Phase 2: 量价突破 (Breakout)
  量能脉冲 + 价格突破 + N 个辅助条件 ≥3 = 突破确认
  + 周线确认 (PB百分位 + 换手率)
  ↓
Phase 3: 退出 (在 Agent 3 中执行)
```

**与旧版 LightGBM 的核心区别**:
- 旧版: 黑盒回归 → 排名 → Top N (不知道为什么选)
- 新版: 白盒多条件状态判断 → 有明确的物理含义 (散户吸筹→主力突破)
- 不再需要 `train_cutoff`, 不训练模型, 无前瞻偏差问题

## 处理流程

```
1. 预筛 (snapshot 单日值, 宽松阈值)
   ~5100 → ~3600 只 (71%)
   条件: mf_sm_proportion > 0.30 OR (breakout_range > 0.48 & vol_impulse > 1.04)

2. 逐股加载近 60 天因子时序 (daily cache CSV)
   对每只预筛股票做完整状态评估

3. Phase 1: 蓄力检测 → Phase 2: 突破检测

4. 排序: 突破优先 (composite_score 降序), 蓄力补充 (accum_quality 降序)

5. 取 Top N, 所有 horizon 共用
```

## Phase 1: 蓄力吸筹条件 (6 选 ≥3, 连续 ≥3 天)

| 条件 | 因子 | 阈值 | IC | 有效率 | 物理含义 |
|------|------|------|----|----|------|
| ① 散户占比高 | `mf_sm_proportion` | > 0.45 | +0.086 | 59.5% | 卖方以散户为主 |
| ② 缩量蓄力 | `vol_shrink` | < 0.70 | -0.047 | 40.6% | 交投冷清 |
| ③ 价格贴中轨 | `breakout_range` | < 0.50 | — | 58.3% | 未启动 |
| ④ 大单净额正 | `mf_big_cumsum_s` | > 0 | — | — | 主力在买入 |
| ⑤ 资金流不平衡 | `mf_flow_imbalance` | > 0 | — | 25.6% | 大单买散户卖 |
| ⑥ 路径不可逆 | `path_irrev_m` | > 0.02 | — | 17% | 有定向力量 |

**硬性要求**: 必须有 `mf_sm_proportion` 数据 (核心因子)。

## Phase 2: 量价突破条件 (≥3 个, 硬性 + 软性)

| 条件 | 因子 | 阈值 | 类型 |
|------|------|------|------|
| 近期蓄力 | is_accum rolling(10) | > 0 | 前提 |
| **量能放大** | `vol_impulse` | > 1.3 | **硬性** |
| **价格突破** | `breakout_range` | > 0.80 | **硬性** |
| 散户追涨 | `mf_sm_proportion` | > 0.50 | 软性 |
| 波动率可控 | `volatility_l` | < 0.05 | 软性 |
| 布林扩张 | `bbw_pctl` | > 0.30 | 软性 |

**硬性条件**: 量能放大 AND 价格突破 必须同时满足 (无论总分多高)。

## 周线确认 (仅突破需要)

- `pb_pctl` ≥ 0.20 (PB 百分位, 趋势延续)
- `weekly_turnover_ma4` ≤ 30.0 (不能过热)
- 无数据时不否决

## 质量评分

| 评分 | 范围 | 用途 |
|------|------|------|
| `accum_quality` | [0, 1] | 蓄力质量 (散户占比25% + 缩量20% + 贴中轨20% + 资金流15% + 路径10% + 熵10%) |
| `breakout_quality` | [0, 1] | 突破质量 (量能25% + 突破幅度25% + 散户追涨20% + 布林15% + 大单15%) |
| `composite_score` | [0, 1] | 0.4 × accum + 0.6 × breakout |

## 性能

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 预筛 | ~0.01s | snapshot DataFrame 操作 |
| 逐股评估 | ~47s | 加载 3600+ 个 CSV, 逐股计算状态 |
| 总计 | ~48s | 比旧版 LightGBM (~110s) 快 ~57% |

## 典型输出

```
scan_date=20250327:
  预筛: 5105 → 3635 只
  状态: 205 只突破, 1178 只蓄力 (共 1383 只有状态)
  选出: 5 只 (5 突破 + 0 蓄力)
```
