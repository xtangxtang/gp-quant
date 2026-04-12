---
title: "储层计算预测临界点 (2026)"
tags: [reservoir-computing, tipping-point, early-warning]
confidence: medium
status: active
source_file: docs/papers/tipping_points_reservoir_computing.pdf
created: 2026-04-12
updated: 2026-04-12
---

# Reservoir Computing for Ultra-Early Tipping Point Prediction (2026)

## 核心结论

储层计算（一种简化 RNN）可以在**极早期**检测到系统临界点逼近——比传统 CSD 指标更早。

## 方法

- 使用随机权重的储层网络处理时间序列
-  储层隐态中的特征值变化能反映系统动力学变化
- 无需训练分岔的标记数据

## 对本项目的态度

> **适合做门控/过滤器，不适合做主信号。**

这是 [[four-layer-system]] 第四层（实验层）的候选技术之一，当前权重 = 0%。

理由（共识 #5）：
- TDA 和储层计算的信号可靠性尚未在金融数据上充分验证
- 计算成本较高
- 适合作为额外的确认/否决信号

## 共识映射

→ 共识 #5: TDA + 储层计算适合作为门控/过滤器，不适合主信号  
→ 见 [[12-papers-synthesis]]

## 概念链接

- [[bifurcation]], [[dominant-eigenvalue]], [[four-layer-system]]
