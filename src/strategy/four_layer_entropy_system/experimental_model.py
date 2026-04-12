"""
Layer 4: 实验模型层 (Experimental Model)

基于以下论文实现：
1. hopf_bifurcation_persistent_homology.pdf - TDA 拓扑检测
2. tipping_points_reservoir_computing.pdf - Reservoir 临界点预测
3. pinn_vs_neural_ode.pdf - 结构信息潜变量模型

注意：本层权重为 0%，仅作为辅助观察和研究的实验层。
所有计算结果仅写入日志，不参与决策，用于离线 OOS 验证。
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

from .config import ExperimentalConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentalOutput:
    """实验模型层输出"""

    # TDA 拓扑得分
    tda_score: float              # 持久同调得分
    tda_persistence: float        # 最大持久性
    tda_birth_death: tuple        # (birth, death) 对

    # Reservoir 临界点得分
    reservoir_tipping_score: float  # 临界点预测得分
    reservoir_state: np.ndarray     # Reservoir 状态

    # 结构潜变量得分
    structure_latent_score: float   # 结构化潜变量得分
    latent_state: Dict              # 潜变量状态

    # 综合实验得分（不用于主决策）
    experimental_score: float

    def to_dict(self) -> Dict:
        return {
            'tda_score': self.tda_score,
            'tda_persistence': self.tda_persistence,
            'reservoir_tipping_score': self.reservoir_tipping_score,
            'structure_latent_score': self.structure_latent_score,
            'experimental_score': self.experimental_score,
            'latent_state_compression': self.latent_state.get('compression', 0),
            'latent_state_instability': self.latent_state.get('instability', 0),
            'latent_state_launch': self.latent_state.get('launch', 0),
            'latent_state_diffusion': self.latent_state.get('diffusion', 0),
        }


class ExperimentalModel:
    """
    实验模型评估器

    实现三种实验性方法作为研究和辅助观察。
    """

    def __init__(self, config: ExperimentalConfig):
        self.config = config

    def compute_tda_score(
        self,
        returns: pd.Series,
        embedding_dim: int = 3,
        delay: int = 5,
    ) -> tuple:
        """
        计算 TDA 拓扑得分

        基于 hopf_bifurcation_persistent_homology 论文的简化版本。
        使用 delay embedding 后计算简单的持久性代理。

        返回
        ----
        (score, persistence, birth_death_pairs)
        """
        if len(returns) < embedding_dim * delay + 10:
            return 0.5, 0.0, []

        # 简化版 TDA：使用 delay embedding 后的几何特征
        # 完整 TDA 需要 gudhi 或 ripser 库

        # 构造 delay embedding
        n = len(returns) - (embedding_dim - 1) * delay
        if n < 10:
            return 0.5, 0.0, []

        embedded = np.zeros((n, embedding_dim))
        for i in range(embedding_dim):
            embedded[:, i] = returns.iloc[i * delay:i * delay + n].values

        # 简化持久性代理：计算点云的"环状"程度
        # 使用相关矩阵的特征值分布

        corr = np.corrcoef(embedded.T)
        eigenvalues = np.linalg.eigvalsh(corr)

        # 最大特征值占比表示"主导方向"
        max_eig_ratio = eigenvalues[-1] / eigenvalues.sum()

        # 特征值熵表示"复杂度"
        prob = eigenvalues / eigenvalues.sum()
        prob = prob[prob > 0]
        eig_entropy = -np.sum(prob * np.log(prob))
        max_entropy = np.log(len(eigenvalues))

        # TDA 得分：低熵 + 高主导特征值 = 有序结构
        tda_score = (1 - eig_entropy / max_entropy) * 0.5 + max_eig_ratio * 0.5

        # 持久性代理
        persistence = max_eig_ratio

        # birth-death 对（简化）
        birth_death = [(0.0, ev) for ev in eigenvalues]

        return tda_score, persistence, birth_death

    def compute_reservoir_tipping_score(
        self,
        returns: pd.Series,
        reservoir_size: int = 100,
        spectral_radius: float = 1.2,
    ) -> tuple:
        """
        计算 Reservoir 临界点得分

        基于 tipping_points_reservoir_computing 论文的简化版本。
        使用固定随机 reservoir 作为动力学敏感度代理。
        """
        if len(returns) < reservoir_size // 2:
            return 0.5, np.zeros(reservoir_size)

        # 简化 Reservoir Computing
        # 完整的 RC 需要训练 readout 层

        # 1. 构造随机 reservoir
        np.random.seed(42)  # 固定随机种子

        # 稀疏连接矩阵
        sparsity = self.config.reservoir_sparsity
        n = reservoir_size

        # 生成稀疏矩阵
        W = np.zeros((n, n))
        mask = np.random.random((n, n)) < sparsity
        W[mask] = np.random.randn(mask.sum()) * 0.5

        # 调整谱半径
        eig_max = np.max(np.abs(np.linalg.eigvals(W)))
        if eig_max > 0:
            W = W * (spectral_radius / eig_max)

        # 2. 输入映射
        u = returns.values
        T = len(u)

        # 归一化输入
        u = (u - np.nanmean(u)) / (np.nanstd(u) + 1e-10)
        u = np.tanh(u)  # 激活函数

        # 3. Reservoir 状态更新
        x = np.zeros(n)
        states = []

        for t in range(T):
            x_new = np.tanh(W @ x + np.random.randn(n) * u[t] * 0.1)
            x = x_new
            states.append(x.copy())

        states = np.array(states)

        # 4. 临界点代理：状态变化率
        state_diff = np.diff(states, axis=0)
        state_var = np.mean(np.std(state_diff, axis=0))

        # 5. 敏感性代理：对输入的响应方差
        sensitivity = np.mean(np.std(states, axis=0))

        # Tipping 得分：高敏感性 + 高变化率 = 接近临界
        tipping_score = min(1.0, (state_var + sensitivity) * 5)

        return tipping_score, states[-1]

    def compute_structure_latent_score(
        self,
        returns: pd.Series,
        volumes: pd.Series,
        latent_dim: int = 4,
    ) -> tuple:
        """
        计算结构信息潜变量得分

        基于 pinn_vs_neural_ode 论文的启示。
        使用启发式方法估计四个潜变量状态。
        """
        if len(returns) < 20:
            return 0.5, {
                'compression': 0.0,
                'instability': 0.0,
                'launch': 0.0,
                'diffusion': 0.0,
            }

        # 四个潜变量的启发式估计

        # 1. Compression（压缩态）：低波动 + 低熵
        vol = returns.std()
        vol_rank = min(1.0, vol / 0.05)  # 波动率排名
        compression = 1.0 - vol_rank

        # 2. Instability（不稳定）：高 AR(1) + 方差抬升
        ar1 = returns.autocorr() if len(returns) > 10 else 0
        var_recent = returns.tail(10).var()
        var_past = returns.tail(30).var() if len(returns) >= 30 else var_recent
        var_lift = var_recent / (var_past + 1e-10)
        instability = min(1.0, (abs(ar1) + min(1, var_lift - 1)) / 2)

        # 3. Launch（启动）：动量 + 成交量放大
        momentum = returns.tail(5).mean() * 10 if len(returns) >= 5 else 0
        vol_ratio = volumes.tail(5).mean() / volumes.tail(20).mean() if len(volumes) >= 20 else 1
        launch = min(1.0, (max(0, momentum) + min(1, vol_ratio - 1)) / 2)

        # 4. Diffusion（扩散）：高波动 + 无序
        vol_expansion = returns.tail(10).std() / returns.tail(30).std() if len(returns) >= 30 else 1
        diffusion = min(1.0, vol_expansion * 0.5 + (1 - compression) * 0.5)

        latent_state = {
            'compression': max(0, min(1, compression)),
            'instability': max(0, min(1, instability)),
            'launch': max(0, min(1, launch)),
            'diffusion': max(0, min(1, diffusion)),
        }

        # 综合得分：launch 高 + 其他适中 = 好信号
        structure_score = (
            latent_state['launch'] * 0.4 +
            latent_state['compression'] * 0.2 +
            (1 - latent_state['diffusion']) * 0.2 +
            (1 - latent_state['instability']) * 0.2
        )

        return structure_score, latent_state

    def evaluate(
        self,
        returns: pd.Series,
        volumes: pd.Series,
    ) -> ExperimentalOutput:
        """
        执行实验模型评估

        参数
        ----
        returns : pd.Series
            收益率序列
        volumes : pd.Series
            成交量序列

        返回
        ----
        ExperimentalOutput
            实验模型评估结果
        """
        # 1. TDA 得分
        tda_score, tda_persistence, birth_death = self.compute_tda_score(
            returns,
            self.config.tda_embedding_dim,
            self.config.tda_delay,
        )

        # 2. Reservoir 临界点得分
        reservoir_score, reservoir_state = self.compute_reservoir_tipping_score(
            returns,
            self.config.reservoir_size,
            self.config.reservoir_spectral_radius,
        )

        # 3. 结构潜变量得分
        structure_score, latent_state = self.compute_structure_latent_score(
            returns,
            volumes,
            self.config.latent_dim,
        )

        # 综合实验得分（权重均为 0，仅供观察）
        experimental_score = (
            tda_score * self.config.weight_tda +
            reservoir_score * self.config.weight_reservoir +
            structure_score * self.config.weight_structure
        )

        # 如果没有权重，则简单平均供观察
        if self.config.weight_tda + self.config.weight_reservoir + self.config.weight_structure == 0:
            experimental_score = (tda_score + reservoir_score + structure_score) / 3

        # 仅写入日志用于离线 OOS 分析，不参与决策
        logger.debug(
            "experimental_layer: tda=%.3f reservoir=%.3f structure=%.3f combined=%.3f latent=%s",
            tda_score, reservoir_score, structure_score, experimental_score,
            {k: f"{v:.3f}" for k, v in latent_state.items()},
        )

        return ExperimentalOutput(
            tda_score=tda_score,
            tda_persistence=tda_persistence,
            tda_birth_death=birth_death,
            reservoir_tipping_score=reservoir_score,
            reservoir_state=reservoir_state,
            structure_latent_score=structure_score,
            latent_state=latent_state,
            experimental_score=experimental_score,
        )
