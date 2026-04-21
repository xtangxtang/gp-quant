# Bull Hunter v3 — 迭代记录 (2026-04-20)

> 本次迭代: LLM file-mode 集成 → 因子扩展 → XGBoost 切换 → 分层因子剔除 → 三版回测对比

---

## 一、代码修改总览

### 1. LLM File-Mode 集成 (`llm_advisor.py`)

- 新增 `_file_mode_call()` 方法, 通过预生成的 JSON 文件模拟 LLM 响应
- 配置: `.env` 中 `LLM_MODE=file`, `LLM_FILE_DIR=results/bull_hunter/_llm_exchange`
- 角色前缀匹配 (factor_advisor / risk_monitor / strategy_analyst)
- 目的: 回测时避免 LLM API 调用, 保证可复现性

### 2. 因子扩展: 32 → 40 个 (`agent1_factor.py`, `agent2_train.py`)

在 `DAILY_FACTORS` 列表中新增 8 个衍生因子 (来自 factor_advisor 建议):

| 因子 | 计算方式 | 理由 |
|------|---------|------|
| `momentum_20d` | `close.pct_change(20)` | 短期动量, 30% 短线目标核心指标 |
| `momentum_60d` | `close.pct_change(60)` | 中期动量, 100% 翻倍目标参考 |
| `price_vs_ma20` | `close / MA20 - 1` | 价格偏离均线程度 |
| `vol_price_synergy` | `量比(5/20) × 5日正收益` | 量价协同度 |
| `volatility_ratio` | `vol_10d / vol_60d` | 短长期波动率之比 |
| `mf_reversal_zscore` | `rolling_z(net_mf_amount, 20)` | 资金流反转 z-score |
| `atr_20d` | `ATR(20) / close` | 标准化 ATR |
| `close_vs_high_60d` | `close / high_60d - 1` | 距60日新高距离 |

`compute_derived_factors()` 函数在 `agent1_factor.py` 中实现, 用于快照提取和训练 panel 构建。

### 3. 模型切换: LightGBM → XGBoost (`agent2_train.py`)

```python
model_type: str = "xgboost"   # lgbm | xgboost | random_forest
```

- XGBoost 支持在 `_train_single_target()` 中完整实现
- 含 fallback: 若 `import xgboost` 失败, 自动回退到 LightGBM

### 4. 统一因子剔除 → 分层因子剔除 (`agent2_train.py`)

**V2 (统一剔除)**: 所有目标共用 `drop_factors`:
```python
drop_factors: list[str] = ["mf_sm_proportion", "coherence_l1", "von_neumann_entropy"]
```

**V3 (分层剔除)**: 新增 `target_drop_factors` 字段, 按目标差异化:

| 目标 | 剔除因子 | 设计思路 |
|------|---------|---------|
| **30pct** (10d) | mf_sm_proportion, coherence_l1, von_neumann_entropy, perm_entropy_l, dom_eig_l | 短线不需要长周期熵因子, 保留动量因子 |
| **100pct** (40d) | mf_sm_proportion, coherence_l1, von_neumann_entropy | 中期平衡, 保留动量+熵 |
| **200pct** (120d) | mf_sm_proportion, momentum_20d, price_vs_ma20, vol_price_synergy | 长线剔除短期动量噪音, 保留熵结构因子 |

逻辑在 `_train_single_target()` 中:
```python
if target_name in cfg.target_drop_factors:
    target_drops = cfg.target_drop_factors[target_name]
else:
    target_drops = cfg.drop_factors  # fallback 统一列表
```

### 5. 回测报告增强 (`pipeline.py`)

- `_print_backtest_summary()` 新增详细格式: 各等级 × 各周期的均值/中位数/胜率/命中率/最大涨亏
- `skip_monitor` 参数: 回测模式跳过 Agent 4 监控

---

## 二、三版回测结果对比

**回测条件**: 2025-01-01 ~ 2025-12-30, 每 20 个交易日扫描一次 (共 13 次), top_n=10

### 总体对比 (ALL 等级)

| 版本 | 模型 | 因子数 | 预测数 | 10d均值 | 10d胜率 | 40d均值 | 40d胜率 | 120d均值 | 120d胜率 |
|------|------|-------|--------|---------|---------|---------|---------|----------|----------|
| **V1 Baseline** | LightGBM | 32 | 347 | +2.5% | 54% | +6.9% | 54% | **+21.1%** | **70%** |
| **V2 因子扩展** | XGBoost | 40 (统一剔除3) | 337 | **+3.3%** | **55%** | +6.0% | 49% | +15.8% | 62% |
| **V3 分层剔除** | XGBoost | 40 (分层剔除) | 356 | +1.7% | 49% | +3.1% | 51% | +14.3% | 62% |

### A 级对比 (大牛股 200% 目标)

| 版本 | A级预测数 | 10d均值 | 40d均值 | 40d胜率 | 120d均值 | 120d胜率 |
|------|----------|---------|---------|---------|----------|----------|
| **V1 Baseline** | 112 | +1.4% | +6.0% | 50% | **+28.2%** | **78%** |
| **V2 因子扩展** | 87 | +1.0% | +5.8% | 44% | +20.5% | 72% |
| **V3 分层剔除** | 106 | +1.0% | **+6.8%** | **57%** | +20.2% | 70% |

### B 级对比 (翻倍股 100% 目标)

| 版本 | B级预测数 | 10d均值 | 40d均值 | 120d均值 | 120d胜率 |
|------|----------|---------|---------|----------|----------|
| **V1 Baseline** | 116 | **+4.4%** | **+5.8%** | **+21.9%** | **71%** |
| **V2 因子扩展** | 120 | +3.0% | +3.7% | +10.6% | 60% |
| **V3 分层剔除** | 120 | +3.1% | +0.9% | +11.1% | 66% |

### C 级对比 (短线 30% 目标)

| 版本 | C级预测数 | 10d均值 | 40d均值 | 120d均值 |
|------|----------|---------|---------|----------|
| **V1 Baseline** | 119 | +1.6% | **+8.7%** | +13.8% |
| **V2 因子扩展** | 130 | **+5.1%** | +8.4% | **+16.8%** |
| **V3 分层剔除** | 130 | +1.0% | +2.1% | +12.5% |

---

## 三、结论与教训

### 关键发现

1. **V1 Baseline (LightGBM + 32因子) 总体表现最优**
   - 120d 均值 +21.1% 胜率 70%, A级 +28.2% 胜率 78%, 各项指标全面领先
   - LightGBM 在此数据规模和特征空间下可能比 XGBoost 更稳定

2. **新增 8 个因子没有带来整体提升**
   - V2 10d 短线略有改善 (+3.3% vs +2.5%), 但 40d/120d 显著下降
   - C级短线受益最大 (10d +5.1%), 说明动量因子对短线有帮助但引入长线噪音

3. **分层因子剔除未能挽救 V3**
   - V3 相对 V2 在 A级40d 有所改善 (+6.8% vs +5.8%), 但其他维度进一步下降
   - 差异化剔除策略方向正确, 但剔除组合可能不够优化

### 教训

| 教训 | 详情 |
|------|------|
| **因子不是越多越好** | 新增因子增加了过拟合风险, 尤其在样本量有限时 |
| **模型切换需控制变量** | V2 同时改了因子+模型, 无法归因是哪个因素导致退步 |
| **LightGBM 是更好的默认选择** | 对类别型特征、缺失值处理更鲁棒, 建议保持为默认 |
| **A级长线能力是核心竞争力** | V1 的 A级120d +28.2% 胜率78% 是最有价值的信号 |

### 下一步建议

1. **回退模型到 LightGBM**, 仅用 V1 的 32 因子作为 production baseline
2. 若要新增因子, 采用 **单因子增量测试** — 每次只加一个因子对比
3. 考虑 **特征选择** (SHAP / permutation importance) 替代手动剔除
4. 分层剔除方向可保留, 但需更精细的 **交叉验证** 确定每个目标的最优因子子集

---

## 四、文件变更清单

| 文件 | 变更内容 |
|------|---------|
| `agent1_factor.py` | 新增 `compute_derived_factors()`, DAILY_FACTORS 32→40 |
| `agent2_train.py` | 新增 `target_drop_factors`, model_type→xgboost, DAILY_FACTORS 32→40 |
| `pipeline.py` | 新增 `_print_backtest_summary()` 详细格式, `skip_monitor` 参数 |
| `llm_advisor.py` | 新增 `_file_mode_call()`, LLM file-mode 支持 |
| `.env` | LLM_MODE=file, LLM_FILE_DIR 配置 |

## 五、回测日志

| 版本 | 日志路径 |
|------|---------|
| V1 Baseline | `/tmp/bull_backtest_2025_v2.log` |
| V2 因子扩展 | `/tmp/bull_backtest_v2.log` |
| V3 分层剔除 | `/tmp/bull_backtest_v3.log` |
| V4 Live v2 (Agent 4, 有 xgboost bug) | `/tmp/bt_agent4_v2.log` |
| V4 Live v3 (Agent 4, 修复后) | `/tmp/bt_agent4_v3.log` |

---

## 六、V4 Live Backtest — Agent 4 整合 + Bug 修复 (2026-04-20 ~ 04-21)

### 6.1 背景

上述 V1-V3 是静态 interval backtest (每 20 天快照打分)。V4 转向 **Live Backtest**：
模拟真实交易循环 (每日买卖 + 持仓管理 + Agent 6 退出模型 + Agent 4 周期监控)。

### 6.2 代码改动 (5 项)

| 文件 | 改动 |
|------|------|
| `pipeline.py` | `run_live()` 返回 `factor_snapshot`; `run_live_backtest()` 新增 `monitor_interval` 参数, 每 N 天调 Agent 4 `run_monitor()`, 检查 `retrain_required` → 触发 Agent 2 重训 |
| `agent2_train.py` | 新增 `_extract_date()` 支持 `_retrain` 后缀目录; 新增 `_is_model_loadable()` 跳过 xgboost 模型; `MAX_MODEL_VERSIONS` 8→20 |
| `run_bull_hunter.py` | 新增 `--monitor_interval` CLI 参数 |

### 6.3 发现并修复 3 个 Bug

| Bug | 症状 | 根因 | 修复 |
|-----|------|------|------|
| **`_retrain` 目录不可见** | Agent 4 重训的模型被忽略 | `d.isdigit()` 对 `20250103_retrain` 返回 False | `_extract_date()` 按 `_` 分割取首段 |
| **xgboost 模型不可加载** | 82 天扫描失败 | 环境无 xgboost, `pickle.load()` 崩溃 | `_is_model_loadable()` 检查 `meta.json` model_type |
| **模型清理过度** | Agent 4 频繁重训导致旧基础模型被删 | `MAX_MODEL_VERSIONS=8` 太小 | 增至 20 |

### 6.4 Live Backtest 结果对比

**命令**: `--live-backtest --start_date 20250101 --end_date 20251230 --retrain_interval 0 --monitor_interval 20`

| 指标 | Live v2 (有 xgboost bug) | Live v3 (修复后) |
|------|--------------------------|-------------------|
| 交易笔数 | 10 | **13** |
| 胜率 | 30% | **46.2%** |
| 平均盈亏 | +2.5% | **+15.6%** |
| 总盈亏 | +57,924 | **+349,873 (6x)** |
| 最佳交易 | — | +122.5% (华丰股份) |
| 最差交易 | — | -20.3% (星徽股份) |
| Agent 4 重训次数 | 8 | **12** |
| xgboost 错误天数 | **82** | **0** |

### 6.5 关键发现: 分水岭 = 4月7日

- 第一批 (初始模型 20250101, prob_200=0.14~0.39): **7 笔全部止损**, 合计 -174,332
- 第二批 (Agent 4 重训模型, prob_200=0.44~0.65): **6 笔全部盈利**, 合计 +524,205
- 卖出全由 **trailing stop** 触发 (最高点回撤 20-31%)
- 华丰股份单笔 +122.5% (+243,761), 贡献总盈亏的 70%

### 6.6 Bug 发现: 回路 A 漏选复盘未生效 (2026-04-21 修复)

**症状**: 13 次 Agent 4 报告中 `missed_bull_replay = null`, `diagnosis = ""`, `actions = []`

**根因**: `_replay_missed_bulls()` 硬拼路径 `bull_models/{scan_date}/`, 但实际模型目录都带 `_retrain` 后缀 → 路径永远不存在 → 直接 `return {}`

**连锁后果**:
1. 漏选复盘返回空 → `diagnosis = ""` → 所有调参分支跳过
2. `actions = []` → 12 次重训全是"原样重训"(同因子、同参数、同模型)
3. 重训全由回路 B (跟踪反馈胜率低) 触发, 回路 A 的 miss_rate=1.0 / 40 个 factor_blind_spots / 行业盲区分析全部白算

**修复**: `agent4_monitor.py` `_replay_missed_bulls()` 改用 `get_model_for_date(cache_dir, scan_date)` 替代 `os.path.join(cache_dir, "bull_models", scan_date)`

**影响**: 下次回测时回路 A 将正常产出 `threshold_issue`/`label_gap`/`factor_blind` 诊断, 驱动 Agent 2 做带调参重训 (降阈值/调权重/剔因子/换模型), 而非空 actions 的原样重训

### 6.7 待验证

- [ ] 修复后重跑 live backtest, 验证回路 A 诊断 + 带调参重训是否进一步提升收益
- [ ] 观察 `diagnosis` 分布: threshold_issue vs label_gap vs factor_blind 哪个最常见
- [ ] 带调参重训 vs 原样重训的模型 AUC 对比
