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

## 第三步：signal_detector_v2 — 数据驱动选股 (2026-04-17 重构)

**代码**: `signal_detector_v2.py` → `scan_service.py`

> 基于 factor_profiling 全市场统计结果重构, 替代旧版 `signal_detector.py`。
> 旧版以熵低位为核心条件 (perm_entropy < 0.65 → 全市场 0.2% 通过率), 已废弃。

**触发方式**:
```bash
python -m src.strategy.entropy_accumulation_breakout.run_entropy_accumulation_breakout \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --feature_cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --scan_date 20260417 --top_n 30
```

### 新旧对比

| 维度 | 旧版 signal_detector | 新版 signal_detector_v2 |
|------|---------------------|------------------------|
| 核心因子 | perm_entropy_m (14.5%) | mf_sm_proportion (59.5%) |
| 突破条件 | dom_eig > 0.85 (12.5%) | vol_impulse > 1.3 (P75) + breakout_range > 0.8 |
| 吸筹通过率 | 0.2% (2/949) | ~10% (92/949) |
| 突破通过率 | 0% (0/949) | ~4% (35/949) |
| 熵因子角色 | 核心过滤 | 辅助加分 (权重 10%) |
| 资金流 | 辅助 | 核心 (必须有数据才能进入吸筹) |

### 三阶段设计

```
阶段1: ACCUMULATION (蓄力吸筹)
  硬性: 必须有资金流数据 (mf_sm_proportion)
  条件 (6选3+连续3天):
    - mf_sm_proportion > 0.45 (散户占比高→主力吸筹)
    - vol_shrink < 0.70 (缩量→交投冷清)
    - breakout_range < 0.50 (价格贴中轨→尚未启动)
    - mf_big_cumsum_s > 0 (大单净额为正)
    - mf_flow_imbalance > 0 (大单买散户卖)
    - path_irrev_m > 0.02 (有定向力量)

阶段2: BREAKOUT (量价突破)
  硬性: vol_impulse > 1.3 且 breakout_range > 0.80
  条件 (6选3+硬性条件):
    - 近10天经历过蓄力阶段
    - vol_impulse > 1.3 (量能放大)
    - breakout_range > 0.80 (价格突破布林带)
    - mf_sm_proportion > 0.50 (散户追涨)
    - volatility_l < 0.05 (排除暴涨暴跌)
    - bbw_pctl > 0.30 (波动率扩张)
  周线确认: pb_pctl > 0.20 & weekly_turnover_ma4 < 30

阶段3: EXIT (退出)
  4信号中满足2个即退出:
    - vol_shrink > 1.5 (先放量后急缩)
    - mf_sm_proportion < 0.25 (散户跑光)
    - breakout_range < 0.10 (动能耗尽)
    - perm_entropy_m > 0.98 (极度无序)
  安全网: 最大持有 15 天
```

### 评分权重

**蓄力质量 AQ**: mf_sm_proportion 25% + vol_shrink 20% + breakout_range 20% + mf_flow_imbalance 15% + path_irrev_m 10% + perm_entropy_m 10%

**突破质量 BQ**: vol_impulse 25% + breakout_range 25% + mf_sm_proportion 20% + bbw_pctl 15% + mf_big_net_ratio 15%

**综合评分**: 40% AQ + 60% BQ + 周线 PB 加分

**输出**:
- `results/entropy_accumulation_breakout/market_snapshot_20260417.csv` (全市场)
- `results/entropy_accumulation_breakout/breakout_candidates_20260417.csv` (Top 30)

### 首次运行结果 (20260417)

全市场 5508 只 → 949 只通过基础过滤 → 92 只蓄力 → **35 只突破**

Top 5: 汽轮科技(0.495) / 四川长虹(0.474) / 翱捷科技(0.466) / 浙江荣泰(0.461) / 合合信息(0.451)

行业分布: 元器件(6)、电气设备(3)、小金属(3)、半导体(2)、汽车配件(2)

---

## 三步的关系

| 步骤 | 输入 | 输出 | 涉及阈值? |
|------|------|------|-----------|
| 第一步 feature_engine | 原始行情 CSV | daily/weekly 缓存 (因子数值) | ❌ 无阈值，纯数学 |
| 第二步 factor_profiling | daily/weekly 缓存 | factor_profile (因子好不好用) | ⚙️ IC/IC_IR 门槛 (评估用) |
| 第三步 signal_detector_v2 | daily/weekly 缓存 | 选股信号 | ✅ 数据驱动阈值 (选股用) |

- 第一步和第二步是**独立的**：factor_profiling 读缓存但不修改缓存
- **第二步的结果已用于指导第三步的阈值设计** (v2 重构)
- 第三步的阈值基于全市场分布百分位 + factor_profiling 有效率排名
- 旧版 `signal_detector.py` 保留用于回测对比, 不再被 scan_service 调用
