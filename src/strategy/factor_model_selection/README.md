# 因子模型选股策略 (Factor Model Selection)

> 数据驱动的截面选股：计算全市场因子 IC → 加权评分排序 → 选 Top N

---

## 核心思路

```
全市场特征计算 (feature_engine + feature_cache)
  → 因子有效性回测 (factor_profiling: IC 分析)
    → 加权评分排序 (ic_scoring / factor_model)
      → 选出 Top N 候选
```

与三阶段状态机 (`entropy_accumulation_breakout`) 的区别：

| 维度 | 状态机 | 因子模型 |
|------|--------|---------|
| 选股逻辑 | 规则过滤：必须经历吸筹→突破序列 | 截面排序：评分最高的 Top N |
| 退出逻辑 | 崩塌检测（熵扩散 + 量能衰竭） | 固定持有期（1d/3d/5d/1w/3w/5w） |
| 因子用法 | 硬阈值门槛 | 连续权重（IC 越大权重越大） |
| 可解释性 | 高（每步有物理含义） | IC 加权中等 / LightGBM 低 |
| 信号密度 | 极低 | 高（每天都有 Top N） |

## 文件结构

| 文件 / 目录 | 职责 |
|-------------|------|
| `ic_scoring.py` | IC 加权截面评分选股（线性模型，6 个 horizon 各取 Top N） |
| `factor_model.py` | LightGBM walk-forward 因子模型（非线性，时间衰减加权） |
| `factor_profiling.py` | 全市场个股因子有效性画像（计算 IC、有效率、方向） |
| `run_factor_model.py` | 统一 CLI 入口 |
| `v3_bull_hunter/` | **Bull Hunter v3**: 四 Agent 大牛股预测流水线（详见 [v3_bull_hunter/README.md](v3_bull_hunter/README.md)） |
| `memory/` | 设计文档 |

## 数据依赖

共享 `entropy_accumulation_breakout` 的特征引擎和缓存：

```
/gp-data/feature-cache/
├── daily/{symbol}.csv      ← feature_engine 计算的日线特征
├── weekly/{symbol}.csv     ← feature_engine 计算的周线特征
├── factor_profile/{date}/  ← factor_profiling 输出的 IC 权重
└── lgb_models/             ← factor_model 训练的 LightGBM 模型
```

## 使用方法

### 1. 因子画像 (先运行一次)

```bash
python -m src.strategy.factor_model_selection.run_factor_model \
  --mode profiling \
  --feature_cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --out_dir results/factor_model_selection
```

### 2. IC 加权选股

```bash
python -m src.strategy.factor_model_selection.run_factor_model \
  --mode ic_scoring \
  --feature_cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --scan_date 20260419
```

### 3. LightGBM 训练 + 选股

```bash
# 训练
python -m src.strategy.factor_model_selection.run_factor_model \
  --mode lgb_train \
  --feature_cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache

# 选股
python -m src.strategy.factor_model_selection.run_factor_model \
  --mode lgb_scoring \
  --feature_cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --scan_date 20260419

# Walk-forward 回测
python -m src.strategy.factor_model_selection.run_factor_model \
  --mode lgb_backtest_wf \
  --feature_cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --backtest_start_date 20250101 --backtest_end_date 20250630 \
  --horizons 5d --top_n 5
```

### 4. 直接调用 factor_model CLI

```bash
python -m src.strategy.factor_model_selection.factor_model \
  --cache_dir /nvme5/xtang/gp-workspace/gp-data/feature-cache \
  --mode backtest_wf \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/tushare-daily-full \
  --backtest_start_date 20250101 --backtest_end_date 20250630
```

### 5. Bull Hunter v3 — 大牛股预测

```bash
# 单日扫描
python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
  --scan_date 20260419

# 滚动回测 (带实际收益验证)
python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
  --backtest --start_date 20250301 --end_date 20251230 --interval_days 20 --top_n 10
```

## 因子体系

### 日线因子 (32 个)

- **熵因子**: perm_entropy_{s,m,l}, entropy_slope, entropy_accel
- **不可逆性**: path_irrev_{m,l}
- **临界减速**: dom_eig_{m,l}
- **换手率熵**: turnover_entropy_{m,l}
- **波动率**: volatility_{m,l}, vol_compression, bbw_pctl
- **量价**: vol_ratio_s, vol_impulse, vol_shrink, breakout_range
- **资金流**: mf_big_net, mf_big_net_ratio, mf_big_cumsum_{s,m,l}, mf_sm_proportion, mf_flow_imbalance, mf_big_momentum, mf_big_streak
- **量子相干**: coherence_l1, purity_norm, von_neumann_entropy, coherence_decay_rate

### 周线因子 (30 个)

日线因子子集 + 周线特有：pe_ttm_pctl, pb_pctl, weekly_big_net, weekly_big_net_cumsum, weekly_turnover_ma4, weekly_turnover_shrink
