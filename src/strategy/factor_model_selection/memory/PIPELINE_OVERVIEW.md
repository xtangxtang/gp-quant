# 因子模型选股 — Pipeline 全景

> 记录三条流水线:
> - (A) 特征缓存三步流水线 (数据预处理)
> - (B) 4-Agent 因子状态机选股流水线 (v2)
> - (C) **Bull Hunter v3 大牛股预测流水线** (v3, LightGBM 三分类器)

---

## 4-Agent Pipeline (v2 因子状态机)

> 代码: `pipeline.py` → `run_pipeline_single()` / `run_pipeline_rolling()`
> **v2 重构**: Agent 2/3 从 LightGBM + 固定持有期 → 因子状态机 + 条件退出

```
Agent 1 (因子快照) → Agent 2 (状态机选股) → Agent 3 (条件退出回测) → Agent 4 (策略分析)
```

| Agent | 文件 | 职责 | 耗时 |
|-------|------|------|------|
| **Agent 1** | `agent_factor.py` | 从缓存提取 scan_date 的全市场因子截面 | ~55-63s |
| **Agent 2** | `agent_selection.py` | 三阶段状态机 (蓄力→突破), 预筛→逐股评估→Top N | ~48s |
| **Agent 3** | `agent_validation.py` | 条件退出回测 (4选≥2 结构崩塌信号, 最大持有天数安全网) | ~0.1s |
| **Agent 4** | `agent_analysis.py` | 有效性判定 + 全市场大牛股捕获分析 + 收益贡献 | ~38s |

**详细文档**:
- [AGENT1_FACTOR.md](AGENT1_FACTOR.md) — 因子加载、过滤、合并
- [AGENT2_SELECTION.md](AGENT2_SELECTION.md) — 因子状态机: 蓄力/突破条件体系
- [AGENT3_VALIDATION.md](AGENT3_VALIDATION.md) — 条件退出: 结构崩塌检测
- [AGENT4_ANALYSIS.md](AGENT4_ANALYSIS.md) — 有效性阈值、大牛股诊断、贡献分析

### v2 vs v1 对比

| 方面 | v1 (LightGBM) | v2 (状态机) |
|------|--------------|------------|
| Agent 2 | 黑盒回归 → 排名 Top N | 白盒因子状态机: 蓄力→突破 |
| Agent 3 | 固定持有期 | 条件退出 + 安全网 |
| 依赖 | 需 train_cutoff, LightGBM | 仅需因子缓存 CSV |
| Agent 2 耗时 | ~110s | ~48s (快 57%) |
| 可解释性 | 低 (模型打分) | 高 (每个条件有物理含义) |

### 滚动模式

```
scan_interval = max(hold_days, user_scan_interval)

3d: 每 3 天扫描 → 12 个扫描日/季度
5d: 每 5 天扫描 → 12 个扫描日/季度
1w: 每 5 天扫描 → 12 个扫描日/季度
3w: 每 15 天扫描 → 4 个扫描日/季度
5w: 每 25 天扫描 → 3 个扫描日/季度
```

---

## 特征缓存三步流水线 (数据预处理)

```
原始行情 CSV → [第一步] feature_engine → daily/weekly 缓存
  → [第二步] factor_profiling → 因子预测力评估
    → [第三步] signal_detector → 选股信号
```

## 旧版备份

| 文件 | 说明 |
|------|------|
| `agent_selection_lgb.py` | v1 LightGBM 选股 (备份) |
| `agent_validation_fixed.py` | v1 固定持有期 (备份) |

---

## Bull Hunter v3 大牛股预测流水线

> 代码: `v3_bull_hunter/` 子目录
> 独立于 v2 状态机，目标不同: 找 30%/100%/200% 大牛股

```
Agent 1 (因子快照) → Agent 2 (LightGBM 训练) → Agent 3 (概率评级) → Agent 4 (监控诊断)
   ~60s                    ~40s                     ~30s                   <1s
```

| Agent | 文件 | 职责 | 耗时 |
|-------|------|------|------|
| **Agent 1** | `v3_bull_hunter/agent1_factor.py` | 从缓存提取 scan_date 的全市场因子截面 (32 因子) | ~60s |
| **Agent 2** | `v3_bull_hunter/agent2_train.py` | 构建训练 panel + 训练 3 个 LightGBM 二分类器 | ~40s |
| **Agent 3** | `v3_bull_hunter/agent3_predict.py` | 全市场概率预测 + A/B/C 分层评级 | ~30s |
| **Agent 4** | `v3_bull_hunter/agent4_monitor.py` | 模型健康检查 + SOP 自动诊断建议 | <1s |

### v3 三个预测目标

| 目标 | 标签定义 | 前瞻窗口 | 分级 |
|------|---------|---------|------|
| `30pct` | 涨幅 ≥ 30% | 10 个交易日 (2 周) | C 级 |
| `100pct` | 涨幅 ≥ 100% | 40 个交易日 (2 月) | B 级 |
| `200pct` | 涨幅 ≥ 200% | 120 个交易日 (6 月) | A 级 |

### v3 Agent 2 训练关键设计

- **训练数据**: 回看 12 个月, 每 5 天采样, 构建 (stock × date) panel (~10 万~24 万行)
- **标签**: `gain_Nd >= threshold` → 1, 正样本极稀少 (< 1%)
- **LightGBM**: 800 树, lr=0.03, max_depth=5, scale_pos_weight ≤ 5.0
- **不用 Early Stopping**: 极端不平衡下 val loss 从第 1 棵树单调递增
- **F1 最优阈值搜索**: [0.01, 0.50], 正样本稀少时固定 0.5 不合理
- **向量化 panel 构建**: pd.merge 替代三重循环, 10+ 分钟→1 秒

### v3 Agent 3 评级逻辑

```
A 级 (大牛股): prob_200 > best_threshold_200
B 级 (翻倍股): prob_100 > best_threshold_100 (且不是 A 级)
C 级 (短线强势): prob_30 > best_threshold_30 (且不是 A/B 级)
```

### v3 Agent 4 健康阈值

| 目标 | min_auc | min_precision | min_recall |
|------|---------|---------------|------------|
| 30pct | 0.58 | 0.10 | 0.15 |
| 100pct | 0.55 | 0.05 | 0.10 |
| 200pct | 0.55 | 0.03 | 0.05 |

### v3 回测结果 (2025-03 ~ 2025-12, 11 次扫描, 301 条预测)

| 等级 | 10d 均值 | 10d 胜率 | 40d 均值 | 40d 胜率 | 120d 均值 | 120d 胜率 |
|------|----------|----------|----------|----------|-----------|-----------|
| A | +3.8% | 57% | +6.0% | 58% | +32.7% | 76% |
| B | +2.1% | 60% | +9.2% | 63% | +31.7% | 74% |
| C | +2.0% | 52% | +7.7% | 57% | +21.2% | 72% |
| ALL | +2.5% | 56% | +7.8% | 60% | +28.0% | 74% |

### v2 vs v3 对比

| 方面 | v2 (状态机) | v3 (Bull Hunter) |
|------|------------|-----------------|
| 方法 | 白盒规则: 蓄力→突破 | LightGBM 二分类 |
| 目标 | 单一 (突破信号) | 三个独立目标 (30%/100%/200%) |
| 退出 | 条件退出 (结构崩塌) | 固定持有期 |
| 评级 | 无分级 | A/B/C 分层推荐 |
| 输出 | 每日少量 Top N | 每日 Top N per grade |
| 监控 | Agent 4 大牛分析 | Agent 4 SOP 自动诊断 |
| 回测 | 固定窗口 | 滚动扫描 + 实际收益验证 |

### v3 使用方法

```bash
# 单日扫描
python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
    --scan_date 20260419

# 滚动回测 (带实际收益验证)
python -m src.strategy.factor_model_selection.v3_bull_hunter.run_bull_hunter \
    --backtest --start_date 20250301 --end_date 20251230 \
    --interval_days 20 --top_n 10
```

### v3 数据路径

```
/gp-data/feature-cache/
├── daily/{symbol}.csv       ← 共享日线特征 (复用 Agent 1)
├── weekly/{symbol}.csv      ← 共享周线特征
└── bull_models/{scan_date}/ ← v3 训练的模型
    ├── model_30pct.pkl
    ├── model_100pct.pkl
    ├── model_200pct.pkl
    └── meta.json

results/bull_hunter/
├── {scan_date}/predictions.csv
├── {scan_date}/health_report.json
└── backtest_{start}_{end}/backtest_detail.csv
```

**详细文档**: [v3_bull_hunter/README.md](../v3_bull_hunter/README.md)
