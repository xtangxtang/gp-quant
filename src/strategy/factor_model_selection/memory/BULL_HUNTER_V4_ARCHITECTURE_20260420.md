# Bull Hunter v4 — 架构重构记录 (2026-04-20)

> Commit: [`c1995e5`](https://github.com/xtangxtang/gp-quant/commit/c1995e5d2d0c1e27858e998fc76d32188e031bee) — "bull hunter v4: A级200%专注架构, 周训+daily预测+tracker跟踪+Agent4双回路反馈"
>
> 本次重构: 从 3-classifier (A/B/C 三级) 简化为 A 级 200% 专注架构，新增周训/每日预测/tracker跟踪/双回路反馈。

---

## 一、重构动机

### V3 回测结论 (参见 BULL_HUNTER_V3_ITERATION_20260420.md)

- V1 Baseline (LightGBM + 32 因子) 全面优于 V2/V3 的因子扩展和 XGBoost 切换
- **A 级 120d 是核心竞争力**: +28.2% 均值, 78% 胜率
- B/C 级信号噪音大, 对整体组合收益贡献有限
- 3-classifier 系统 (30pct/100pct/200pct) 过于复杂, 维护成本高

### 设计目标

1. **聚焦 A 级 200%**: 只做最擅长的事 — 筛选 6 个月可能涨 200% 的大牛股
2. **分离训练与预测**: 训练重 (周频), 预测轻 (日频复用 latest 模型)
3. **持续跟踪反馈**: 记录每次预测 → 30 天后评估 → 反哺模型调优

---

## 二、架构变更

### 运行模式: 从 scan/feedback/rolling → daily/train/review/backtest

| 模式 | 命令 | 说明 |
|------|------|------|
| `--daily` | `--daily --scan_date 20260420` | 每日轻量预测: Agent 1 + Agent 3 (复用 latest 模型) + tracker 记录 |
| `--train` | `--train --scan_date 20260420 [--force]` | 周训: Agent 1 + Agent 2, 更新 latest 模型 (默认 5 交易日间隔) |
| `--review` | `--review --scan_date 20260420` | 复盘: Agent 4 双回路诊断, 可能触发 Agent 2 重训 |
| `--backtest` | `--backtest --start_date ... --end_date ...` | 滚动回测 + 实际收益验证 (兼容旧模式) |
| `--scan_date` (无模式flag) | `--scan_date 20260420` | 兼容旧接口: train + predict 一步完成 |

### 目标简化: 3 targets → 2 targets

| 变更前 | 变更后 |
|--------|--------|
| 30pct (10d), 100pct (40d), 200pct (120d) | **200pct (120d)** 主模型 + **100pct (40d)** 辅助模型 |
| A/B/C 三级评分 | **仅 A 级** (prob_200 > 阈值), 每日最多 Top 5 |

### 新增组件: tracker.py

```
results/bull_hunter/tracking/
├── active.csv    # 当前活跃跟踪 (30天窗口内)
└── history.csv   # 已到期评估记录
```

- `record_predictions()`: 记录每日预测, 含入场价
- `update_prices()`: 每日更新活跃项的最新价格和最大涨幅
- `evaluate_expired()`: 30 天到期后评估 (success: max_gain≥30%, fail: max_gain<10%, loss: current_gain<-10%)
- `get_recent_predictions()`: 供 Agent 3 去重 (3 天内不重复推荐同一股票)

### Agent 4 双回路

| 回路 | 内容 | 触发条件 |
|------|------|---------|
| **Loop A** (已有) | 漏网之鱼分析: 市场上实际涨 200%+ 但模型没选中的股票 | 每次 review |
| **Loop B** (新增) | 跟踪反馈: 分析已到期预测的胜率/亏损率/排名表现/因子模式 | 有到期评估时 |

触发重训条件: 胜率 < 30% 或 亏损率 > 30%, 安全限制: MAX_ACTIONS_PER_DIRECTIVE=3, MIN_RETRAIN_INTERVAL=2 天。

### 模型版本管理

```
feature-cache/bull_models/
├── 20250101/          # 日期版本目录
│   ├── model_200pct.pkl
│   ├── model_100pct.pkl
│   └── meta.json
├── 20250207/
├── ...
├── latest -> 20250101  # 符号链接, 指向最新可用模型
```

- `needs_training()`: 检查距上次训练是否 ≥ 5 个交易日
- `get_latest_model_dir()`: 读取 latest 符号链接
- `_cleanup_old_models()`: 保留最近 8 个版本, 保护 latest 指向的目录不被删除

---

## 三、文件变更清单

| 文件 | 行数变化 | 变更内容 |
|------|---------|---------|
| `agent2_train.py` | +339/-部分 | TARGETS 3→2 (移除 30pct); TrainConfig: model_type→lgbm, drop_factors→[], 新增 trigger/trigger_reason/force_retrain; 新增 `needs_training()`, `get_latest_model_dir()`, `_update_latest_link()`, `_cleanup_old_models()`; `_build_training_panel` 只算 gain_40d/gain_120d |
| `agent3_predict.py` | +152/-部分 | 全面重写: 仅 A 级输出, TOP_N=5, 新增 dedup (3天窗口); PredictConfig 移除 threshold_30pct/top_n_per_grade, 新增 top_n/dedup_days |
| `tracker.py` | +333 (新文件) | PredictionTracker 类: 持久化预测跟踪, 30 天评估, 去重支持 |
| `agent4_monitor.py` | +1170 | 新增 `_analyze_tracking_feedback()` (Loop B ~120行); 更新 `_generate_tuning_directives()`: 合并 tracking 信号, MAX_ACTIONS_PER_DIRECTIVE=3 限制; HEALTH_THRESHOLDS 移除 30pct |
| `pipeline.py` | +507/-部分 | 新增 `run_daily()`, `run_train()`, `run_review()`; 保留 `run_scan()` 兼容; 移除 `run_scan_with_feedback()` |
| `run_bull_hunter.py` | +292/-部分 | CLI 新增 --daily/--train/--review/--force; 移除 --feedback/--rolling; 新增 `_print_daily_result()`, `_print_review_result()`, `_print_scan_result()` |

### 关键默认值变更

| 参数 | V3 值 | V4 值 |
|------|-------|-------|
| `model_type` | `"xgboost"` | `"lgbm"` |
| `drop_factors` | `["mf_sm_proportion", "coherence_l1", "von_neumann_entropy"]` | `[]` (空, 用全量因子) |
| `TARGETS` | `["30pct", "100pct", "200pct"]` | `["200pct", "100pct"]` |
| `top_n` | 20 (per grade) | 5 (仅 A 级) |
| `TRAIN_INTERVAL_DAYS` | 无 (每次扫描都训练) | 5 (每周训练一次) |
| `TRACKING_DAYS` | 无 | 30 (跟踪评估窗口) |

---

## 四、V4 回测结果

**回测条件**: 2025-01-01 ~ 2025-12-30, 每 20 个交易日扫描, top_n=10, 仅 A 级

### 总体表现

| 指标 | V4 A 级 | V1 A 级 (基线) |
|------|---------|---------------|
| 预测数 | 95 条 (11 期) | 112 条 (12 期) |
| 10d 均值/胜率 | +0.9% / 52% | +1.4% / 54% |
| 40d 均值/胜率 | +5.9% / 54% | +6.0% / 50% |
| **120d 均值/胜率** | **+24.5% / 77%** | **+28.2% / 78%** |
| 40d 最大涨幅均值 | +19.6% | +18.4% |
| 40d ≥30% 比例 | 20% | 20% |

### 等权组合净值 (40 天持有)

| 扫描日 | 数量 | 等权收益 | 累计净值 |
|--------|------|---------|---------|
| 20250207 | 9 | -16.43% | 0.836 |
| 20250307 | 10 | -4.97% | 0.794 |
| 20250407 | 10 | +22.51% | 0.973 |
| 20250508 | 10 | +16.99% | 1.138 |
| 20250606 | 10 | +5.39% | 1.200 |
| 20250704 | 3 | +16.94% | 1.403 |
| 20250829 | 7 | +0.48% | 1.409 |
| 20250926 | 10 | -1.96% | 1.382 |
| 20251103 | 6 | +0.65% | 1.391 |
| 20251201 | 9 | +20.55% | 1.677 |
| 20251229 | 9 | +8.47% | 1.819 |

**最终净值: 1.819, 累计收益: +81.87%**

### 不同持有期

| 策略 | 最终净值 | 累计收益 |
|------|---------|---------|
| A 级等权 10 天 | 1.157 | +15.68% |
| **A 级等权 40 天** | **1.819** | **+81.87%** |
| A 级等权 120 天 | 5.368 | +436.80% (8 期有效) |

### V4 vs V1 差异分析

V4 净值 1.819 vs V1 净值 2.089, 差距来源:

1. **缺少首期 (20250102)**: V1 该期 A 级贡献 +16.3%, V4 由于训练与预测同日导致首期缺失
2. **前两期拖累大**: 20250207 (-16.4%) + 20250307 (-5.0%) 导致净值一度跌至 0.794
3. **筛选更严格**: 95 条 vs 112 条, V4 阈值更高, 部分有效信号被过滤

---

## 五、Bug 修复

### `_cleanup_old_models()` 误删 latest 指向的目录

**问题**: 当训练日期 (如 20250101) 在排序中位于旧模型 (如 20250704) 之前时, cleanup 会将新训练的模型当作"最旧的"删除, 导致 latest 符号链接指向不存在的目录。

**修复**: cleanup 前读取 latest 符号链接的目标目录名, 跳过该目录不删除。

```python
# 修复后
latest_link = os.path.join(root, "latest")
protected = None
if os.path.islink(latest_link):
    protected = os.path.basename(os.path.realpath(latest_link))
# ... 删除时跳过 protected
```

---

## 六、回测日志

| 版本 | 日志路径 |
|------|---------|
| V4 重构 | `/tmp/bull_backtest_v4.log` |
| V1 基线 (对照) | `/tmp/bull_backtest_2025_v2.log` |

---

## 七、后续方向

1. **修复首期缺失**: 回测模式中首日应先训练再预测, 避免空模型
2. **跟踪评估验证**: 在实际 daily 模式下积累 30 天跟踪数据, 验证 Loop B 反馈机制
3. **事件驱动重训**: 当 Agent 4 检测到胜率骤降时自动触发 Agent 2 重训
4. **因子精简**: 用 SHAP 分析 40 个因子中哪些对 200pct 目标真正有贡献, 剔除噪音因子
5. **考虑加入市场状态过滤**: 下跌市 (如 20250207 期) 应减仓或暂停预测
