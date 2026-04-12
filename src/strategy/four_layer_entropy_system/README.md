# 四层熵交易系统

基于 GP-QUANT 12 篇复杂系统论文实现的完整交易架构。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: 市场门控层 (Market Gate)                      │
│  论文：communication_induced_bifurcation_power_packet   │
│  核心：当噪声超过临界阈值时，采用"战略性放弃"策略        │
└─────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2: 个股状态层 (Stock State)                      │
│  论文：Seifert 2025 + period_doubling_dominant_eigenvalue│
│  核心：低熵压缩 → 临界减速 → 分叉启动                    │
└─────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 3: 执行成本层 (Execution Cost)                   │
│  论文：conservative_driving_discrete                    │
│  核心：保守驱动近优性 + 战略放弃机制                     │
└─────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 4: 实验模型层 (Experimental)                     │
│  论文：TDA + Reservoir + PINN                           │
│  核心：辅助观察，权重 0%                                 │
└─────────────────────────────────────────────────────────┘
```

## 四层详细说明

### Layer 1: 市场门控层

**论论文依据**: `communication_induced_bifurcation_power_packet.pdf`

**核心发现**: 当环境噪声超过临界阈值 Dc 时，最优策略是"战略性放弃"精细控制。

**输出状态**:
| 状态 | 含义 | 操作 |
|------|------|------|
| compression | 低熵压缩态 | 最佳开仓窗口 |
| transition | 转换态 | 谨慎开仓 |
| expansion | 扩张态 | 减仓 |
| distorted | 失真态 | 观望 |
| neutral | 中性态 | 正常 |
| abandon | 放弃态 | 清仓 |

**核心指标**:
- `coupling_entropy`: 行业耦合熵（衡量市场联动程度）
- `noise_cost`: 噪声成本（波动率聚集 + 流动性压力 + 极端收益频率）
- `gate_score`: 门控得分（0-1，越高越适合交易）

### Layer 2: 个股状态层

**论文依据**:
- Seifert (2025): 粗粒化熵产生理论
- Ma et al. (2026): 主导特征值与倍周期分岔预测

**状态流转**:
```
低熵压缩 → 临界减速 → 分叉启动 → 扩散/衰竭
  ↓          ↓          ↓          ↓
path_irrev  dominant_eig  突破确认  退出信号
< 0.05      → 0.9        → 放量     → 熵增
```

**核心特征**:
- `path_irreversibility`: 路径不可逆性熵（Seifert 2025）
- `dominant_eigenvalue`: 主导特征值（临界减速预警）
- `permutation_entropy`: 排列熵（有序/无序度量）
- `phase_adjusted_ar1`: 相位校正 AR(1)（周期去偏）

**质量得分**:
- `entropy_quality`: 熵质量（低熵=高质量）
- `bifurcation_quality`: 分叉质量（临界减速强度）
- `trigger_quality`: 触发质量（突破确认）

### Layer 3: 执行成本层

**论文依据**: `conservative_driving_discrete.pdf`

**核心发现**: 保守驱动方案的耗散最多是最优值的 2 倍，无需追求复杂执行。

**建仓模式**:
| 模式 | 仓位 | 适用场景 |
|------|------|---------|
| skip | 0% | 信号不足/战略放弃 |
| probe | 25% | 弱信号，试探性建仓 |
| staged | 50% | 中等信号，分 3 天建仓 |
| full | 80% | 强信号，分 2 天建仓 |

**退出模式**:
- `abandon`: 放弃（立即清仓，高噪声环境）
- `reduce`: 减仓（状态恶化）
- `trail`: 移动止盈（正常状态）

### Layer 4: 实验模型层

**论文依据**:
- `hopf_bifurcation_persistent_homology.pdf`: TDA 拓扑检测
- `tipping_points_reservoir_computing.pdf`: Reservoir 临界点预测
- `pinn_vs_neural_ode.pdf`: 结构信息潜变量模型

**注意**: 本层权重为 0%，仅作为辅助观察和研究。

## 运行方法

### 基本用法

```bash
python -m src.strategy.four_layer_entropy_system.run_scan \
  --data_dir /nvme5/xtang/gp-workspace/gp-data/trade \
  --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv \
  --out_dir results/four_layer_system \
  --max_stocks 50
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --data_dir | - | 分钟数据目录 |
| --basic_path | - | 股票基本信息 CSV |
| --out_dir | results/four_layer_system | 输出目录 |
| --scan_date | None | 扫描日期（YYYY-MM-DD） |
| --max_stocks | 50 | 最大股票数量 |
| --initial_capital | 1,000,000 | 初始资金 |

### 输出文件

```
results/four_layer_system/
├── market_gate.csv         # 市场门控结果
├── stock_decisions.csv     # 个股决策详情
├── summary.json            # 汇总统计
└── buy_signals.csv         # 买入信号列表
```

## 配置参数

在 `config.py` 中可调整各层参数：

```python
@dataclass
class Config:
    # 市场门控层
    market_gate: MarketGateConfig
    
    # 个股状态层
    stock_state: StockStateConfig
    
    # 执行成本层
    execution: ExecutionCostConfig
    
    # 实验模型层
    experimental: ExperimentalConfig
    
    # 各层权重
    layer: LayerConfig
```

### 默认权重

```python
# 最终决策权重
stock_state_weight = 0.70      # 个股状态 70%
market_gate_weight = 0.20      # 市场门控 20%
execution_weight = 0.10        # 执行成本 10%
experimental_weight = 0.00     # 实验层 0%
```

## 决策流程

```
1. 加载数据 → 计算收益率、成交量等基础数据
       │
2. Layer 1 → 评估市场门控（耦合熵 + 噪声成本）
       │
       ├─ 如果 abandonment_flag=True → 全部 wait
       │
3. Layer 2 → 逐股评估状态（熵 + 分叉 + 触发）
       │
4. Layer 3 → 评估执行成本（建仓模式 + 仓位）
       │
5. Layer 4 → 实验模型评估（仅观察）
       │
6. 综合决策 → 基于权重计算最终动作
       │
7. 输出结果 → 保存 CSV/JSON
```

## 与之前回测的区别

| 维度 | 之前回测 | 四层系统 |
|------|---------|---------|
| 市场 filter | 无 | 有（门控层） |
| 相位校正 | 无 | 有（周期去偏） |
| 执行策略 | 一次性建仓 | 分段建仓 |
| 战略放弃 | 无 | 有（高噪声环境） |
| 特征工程 | 单一熵 | 多特征融合 |

## 待完善功能

1. **真正的 TDA 计算**: 当前使用简化代理，需要 gudhi/ripser 库
2. **可训练 Reservoir**: 当前使用固定随机矩阵
3. **真正的结构信息模型**: 当前使用启发式潜变量
4. **实时数据接入**: 当前仅支持离线 CSV

## 参考论文

1. Seifert, U. (2025). Universal bounds on entropy production from fluctuating coarse-grained trajectories. arXiv:2512.07772
2. Ma, Z., et al. (2026). Predicting the onset of period-doubling bifurcations via dominant eigenvalue extracted from autocorrelation. arXiv:2603.05523
3. Hikihara, T. (2026). Communication-Induced Bifurcation and Collective Dynamics in Power Packet Networks.
4. van der Meer, J., & Dechant, A. (2026). Near-optimality of Conservative Driving in Discrete Systems.
