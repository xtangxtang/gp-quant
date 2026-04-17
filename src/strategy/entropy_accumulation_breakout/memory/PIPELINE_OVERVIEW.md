# 三步流水线：从原始行情到选股信号

> 记录 daily/weekly 缓存、factor_profile、选股信号三步之间的关系。

---

## 总览

```
原始行情 CSV (tushare-daily-full / moneyflow / weekly-5d)
  → [第一步] feature_engine (数学变换) → daily/weekly 缓存 (因子数值)
    → [第二步] factor_profiling (统计分析) → factor_profile (因子预测力评估)
      → [第三步] signal_detector (状态机) → 选股信号 (买入/退出)
```

---

## 第一步：daily/weekly 缓存 — 纯数学计算，无阈值

**代码**: `feature_engine.py` → `feature_cache.py`

做的事：对每只股票的原始行情用**固定滚动窗口**做数学变换。

| 计算项 | 方法 | 输出列 |
|--------|------|--------|
| 置换熵 | 10/20/60 日滚动窗口，计算时间序列的有序度 | `perm_entropy_s/m/l` |
| 路径不可逆 | 20/60 日 KL 散度 (正向 vs 反向回报) | `path_irrev_m/l` |
| 临界减速 | 20/60 日主特征值 (自相关矩阵) | `dom_eig_m/l` |
| 波动率 | 收益标准差、布林带宽 | `volatility_m/l`, `bbw_pctl` |
| 量能 | 量比、缩量度 | `vol_ratio_s`, `vol_shrink` |
| 资金流 | 大单/散户净额累加 | `mf_big_net`, `mf_sm_proportion` |
| 量子相干 | 密度矩阵相干性指标 | `coherence_l1`, `purity_norm` |

**输出**: 每只股票每天一行因子数值（如 `perm_entropy_m=0.72`），**本身不含任何买卖判断**。

**Weekly 额外**: 同样的指标用周窗口重算，另加 `pe_ttm_pctl`, `pb_pctl`, `weekly_turnover_ma4` 等 6 列周线特有。

详见 → [FEATURE_CACHE.md](FEATURE_CACHE.md)

---

## 第二步：factor_profile — 评估因子预测力

**代码**: `factor_profiling.py`

**触发方式**:
```bash
python -m src.strategy.entropy_accumulation_breakout.factor_profiling \
  --cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --workers 16
```

**做的事**:

```
对每只股票 (5169只):
  1. 读取 daily cache CSV (83列 × ~500行)
  2. 计算前瞻收益: close[t+N] / close[t] - 1  (N=1,3,5天)
  3. 对每个因子列:
     - 加权 Rank IC: 因子值排名 vs 未来收益排名的相关系数 (近期权重大, λ=0.007)
     - 滚动 IC → IC_IR: IC 均值 / 标准差 = 稳定性
  4. 分类: |IC| > 0.05 且 |IC_IR| > 0.5 → 有效 (正向/负向)
  5. 写入 factor_profile/{date}/{symbol}.json

  同理读取 weekly cache → 前瞻 1w/3w/5w 收益 → 同样分析
```

**输出**:
- 个股画像: `feature-cache/factor_profile/20260417/{symbol}.json`
- 日线汇总: `reports/20260417/factor_summary.csv`
- 周线汇总: `reports/20260417/weekly_factor_summary.csv`
- 总结报告: `reports/20260417/summary_report.md`

**关键结论 (20260417)**:
- 日线 Top: `mf_sm_proportion` (59.5%), `breakout_range` (58.3%), `vol_shrink` (40.6%)
- 周线 Top: `pb_pctl` (85.5%), `breakout_range` (78.9%), `pe_ttm_pctl` (75.4%)
- 熵因子在周线上全线衰退，日线有效率仅 14-17%

详见 → [FACTOR_PROFILING.md](FACTOR_PROFILING.md)

---

## 第三步：signal_detector — 三阶段状态机选股

**代码**: `signal_detector.py` → `scan_service.py`

**触发方式**:
```bash
python -m src.strategy.entropy_accumulation_breakout.run_entropy_accumulation_breakout \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --feature_cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --scan_date 20260417 --top_n 30
```

**做的事**: 用因子数值 + 固定阈值判断每只股票处于哪个阶段：

```
阶段1: ACCUMULATION (惜售吸筹)
  条件: perm_entropy_m < 0.65 & path_irrev_m > 0.05 & mf_flow_imbalance > 0.3 ...
  → 筹码集中、主力定向买入

阶段2: BIFURCATION (分岔突破)
  条件: 从 ACCUMULATION 转入、dom_eig_m > 0.85 & vol_impulse > 1.8× ...
  → 临界减速 + 放量突破

阶段3: COLLAPSE (结构崩塌)
  条件: 从 BIFURCATION 转入、熵扩散 + 量能衰竭
  → 退出信号
```

**输出**: `results/entropy_accumulation_breakout/scan_20260417.csv`

---

## 三步的关系

| 步骤 | 输入 | 输出 | 涉及阈值? |
|------|------|------|-----------|
| 第一步 feature_engine | 原始行情 CSV | daily/weekly 缓存 (因子数值) | ❌ 无阈值，纯数学 |
| 第二步 factor_profiling | daily/weekly 缓存 | factor_profile (因子好不好用) | ⚙️ IC/IC_IR 门槛 (评估用) |
| 第三步 signal_detector | daily/weekly 缓存 | 选股信号 | ✅ 策略阈值 (选股用) |

- 第一步和第二步是**独立的**：factor_profiling 读缓存但不修改缓存
- 第二步的结果 (哪些因子有效) **目前未自动反馈**到第三步的阈值
- 第三步的阈值仍是人工设定的全局固定值

### 潜在优化方向

factor_profiling 发现策略当前使用的熵因子 (`perm_entropy_m` 14.5%) 有效率远低于未使用的 `mf_sm_proportion` (59.5%) 和 `breakout_range` (58.3%)。未来可以：

1. **用 factor_profile 的个股画像动态调整阈值** — 每只股票用它自己的有效因子
2. **替换弱因子** — 用 `mf_sm_proportion` / `breakout_range` 替代或补充熵条件
3. **周线过滤器** — 引入 `pb_pctl` / `pe_ttm_pctl` 做周级别确认
