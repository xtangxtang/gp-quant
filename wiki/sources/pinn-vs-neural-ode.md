---
title: "PINN vs Neural ODE (2026)"
tags: [gray-box, neural-network, physics-informed]
confidence: high
status: active
source_file: docs/papers/pinn_vs_neural_ode_morris_lecar.pdf
created: 2026-04-12
updated: 2026-04-12
---

# PINN vs Neural ODE in Critical Regimes (2026)

## 核心结论

在临界态（系统接近分岔点）下：

- **PINN（Physics-Informed Neural Network）** 显著优于纯 Neural ODE
- 嵌入物理约束（守恒律、对称性）使模型在数据稀疏的关键区域更稳定
- 纯黑箱模型在临界区域容易发散

## 对本项目的意义

这是我们选择 [[gray-box-over-black-box]] 路线的理论支撑：

| 方法 | 优势 | 劣势 |
|------|------|------|
| 纯黑箱 (NN/LSTM) | 灵活，无需领域知识 | 临界区域不稳定，不可解释 |
| 灰箱 (特征工程 + 结构约束) | 临界态更稳定，可解释 | 需要领域理论 |

## 项目影响

我们的整个方法论都是灰箱路线：
- 从物理/信息论推导出明确的特征（熵、特征值）
- 不使用端到端黑箱预测
- 模型可解释：每个信号有明确的物理含义

## 共识映射

→ 共识 #8: 灰箱 + 结构约束 > 纯黑箱时间序列模型  
→ 见 [[12-papers-synthesis]]

## 概念链接

- [[entropy]], [[bifurcation]], [[dominant-eigenvalue]]
