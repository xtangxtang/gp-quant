# Agent 3: 条件退出回测 (v2)

> 代码: `agent_validation.py` → `run_validation()`, 配置 `ExitConfig`
> **v2 重构**: 从固定持有期替换为条件退出 (旧版备份: `agent_validation_fixed.py`)

---

## 职责

对 Agent 2 选出的股票模拟交易, 使用**因子条件退出**替代固定持有期。
退出信号基于结构崩塌检测, 最大持有天数作为安全网。

## 输入 / 输出

| 项目 | 内容 |
|------|------|
| **输入** | `selections` (Agent 2 输出), `data_dir`, `scan_date`, `calendar`, `cache_dir` (退出条件因子) |
| **输出** | `dict[horizon, {trades, metrics, entry_date, exit_date}]` |

## 交易规则

```
买入: scan_date 次日开盘价 (同旧版)
退出: 条件触发 (≥2 个结构崩塌信号) 或 最大持有天数 (安全网)
退场价: 退出日收盘价
仓位: 等权分配
```

## 退出条件 (4 选 ≥2)

| 条件 | 因子 | 阈值 | 物理含义 |
|------|------|------|---------|
| ① 缩量飙升 | `vol_shrink` | > 1.5 | 先放量后急缩, 动能衰竭 |
| ② 散户骤降 | `mf_sm_proportion` | < 0.25 | 散户已跑光, 无人接盘 |
| ③ 突破回落 | `breakout_range` | < 0.10 | 价格跌回布林中轨 |
| ④ 极度无序 | `perm_entropy_m` | > 0.98 | 完全随机, 无定向力量 |

**触发规则**: ≥2 个条件同日满足 → 当日收盘退出。

### 与旧版的核心区别
- 旧版: 固定持有 hold_days 天, 不管中间发生什么
- 新版: 逐日检查退出条件, 可提前止损/止盈
- 新版: 最大持有天数 = hold_map[horizon], 作为安全网

## 最大持有天数 (安全网)

| Horizon | max_hold_days |
|---------|---------------|
| `3d` | 3 |
| `5d` | 5 |
| `1w` | 5 |
| `3w` | 15 |
| `5w` | 25 |

## 处理流程

```
1. 构建交易日历
2. 对每个 horizon:
   a. 确定 entry_date (scan_date + 1) 和 max_exit_date (entry + max_hold)
   b. 对推荐列表中每只股票:
      - 加载日线数据 (计算 PnL)
      - 加载因子时序 (条件退出检测, 仅 4 列)
      - 从 min_hold_days 到 max_hold_days 逐日检查退出条件
      - 首个满足 ≥2 个退出信号的日期 = 实际退出日
      - 无条件触发 → 按最大持有天数退出
   c. 计算绩效指标 (含 condition_exit_rate, avg_hold_days)
```

## 新增指标

| 指标 | 说明 |
|------|------|
| `avg_hold_days` | 实际平均持有天数 |
| `condition_exit_rate` | 条件退出比例 (0~1) |
| `exit_reason` | 每笔交易: "condition" 或 "max_hold" |

## 单笔交易记录 (trade dict)

| 字段 | 说明 |
|------|------|
| `signal_date` | 信号日 (scan_date) |
| `entry_date` | 买入日 |
| `exit_date` | 实际退出日 |
| `symbol` | 股票代码 |
| `name` | 股票名称 |
| `composite_score` | Agent 2 的状态机评分 |
| `phase` | 入选时的阶段 (breakout/accumulation) |
| `entry_price` | 买入价 |
| `exit_price` | 退出价 |
| `pnl_pct` | 收益率 (%) |
| `hold_days` | 实际持有天数 |
| `max_hold_days` | 最大持有天数 |
| `exit_reason` | "condition" 或 "max_hold" |

## 性能

~0.1s (仅加载少量 CSV, 逐日检查简单条件)

## 典型输出

```
scan_date=20250327, 5d:
  5 笔交易, 胜率 1/5
  条件退出 3 笔, 均持有 4.6 天
  均收益 -4.29%
```
