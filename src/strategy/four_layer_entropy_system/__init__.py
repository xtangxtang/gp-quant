"""
四层熵交易系统

基于 GP-QUANT 12 篇复杂系统论文实现的完整交易架构。

四层结构：
1. 市场门控层 (Market Gate) - 判断市场是否适合交易
2. 个股状态层 (Stock State) - 低熵压缩→临界减速→分叉启动
3. 执行成本层 (Execution Cost) - 保守驱动 + 战略放弃
4. 实验模型层 (Experimental) - TDA/Reservoir/Gray-box（辅助）

论文依据：
- Seifert (2025): 粗粒化熵产生理论
- Ma et al. (2026): 主导特征值与倍周期分岔预测
- Hikihara (2026): 通信诱导分岔与战略放弃
- van der Meer & Dechant (2026): 保守驱动近优性
"""

from .config import Config, LayerConfig
from .market_gate import MarketGate, MarketState
from .stock_state import StockState, StateFlow
from .execution_cost import ExecutionCost, ExecutionMode
from .experimental_model import ExperimentalModel
from .core_system import FourLayerEntropySystem

__all__ = [
    'Config',
    'LayerConfig',
    'MarketGate',
    'MarketState',
    'StockState',
    'StateFlow',
    'ExecutionCost',
    'ExecutionMode',
    'ExperimentalModel',
    'FourLayerEntropySystem',
]
