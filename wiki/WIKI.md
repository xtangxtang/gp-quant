# GP-Quant Wiki — Schema Document

> 本文档定义 Wiki 的结构约定、工作流程和维护规则。
> LLM 和人类共同演进此 schema。

## 三层架构

```
docs/papers/          ← Raw Sources (不可变，LLM 只读)
wiki/                 ← Wiki (LLM 维护的 markdown 知识库)
wiki/WIKI.md          ← Schema (本文件，约定与工作流)
```

### Raw Sources (`docs/papers/`)
- 不可变的原始资料：论文 PDF、分析报告、外部文章
- LLM 从此读取但永不修改
- 新增来源放入此目录

### Wiki (`wiki/`)
- LLM 生成并维护的 markdown 文件集合
- 包含摘要、实体页、概念页、对比分析、综述
- LLM 负责创建页面、更新交叉引用、保持一致性
- 使用 `[[wikilink]]` 语法进行页面间链接

### Schema (`wiki/WIKI.md`)
- 本文件，定义 Wiki 的结构和约定
- 随使用经验共同演进

## 目录结构

```
wiki/
├── WIKI.md                    # Schema 文档（本文件）
├── index.md                   # 内容索引（按类别）
├── log.md                     # 时间线日志（append-only）
├── overview.md                # 项目全局综述
│
├── concepts/                  # 概念页：理论与方法论
│   ├── entropy.md             # 熵与不可逆性
│   ├── bifurcation.md         # 分岔与临界转变
│   ├── fractal.md             # 分形与多尺度结构
│   ├── dissipative-structure.md  # 耗散结构与相变
│   ├── permutation-entropy.md # 置换熵
│   ├── dominant-eigenvalue.md # 主特征值与临界减速
│   ├── path-irreversibility.md # 路径不可逆性
│   ├── strategic-abandonment.md # 战略放弃
│   └── multitimeframe-resonance.md # 多时间框架共振
│
├── entities/                  # 实体页：系统组件
│   ├── tick-entropy-module.md # 核心熵计算模块
│   ├── four-layer-system.md   # 四层选股系统
│   ├── multitimeframe-scanner.md # 多时间框架扫描器
│   ├── data-pipeline.md       # 数据管道
│   ├── continuous-decline-recovery.md # 连续下跌恢复策略
│   ├── hold-exit-system.md    # 持有/退出决策系统
│   └── web-dashboard.md       # Web 可视化面板
│
├── sources/                   # 来源摘要页
│   ├── seifert-2025-entropy-bounds.md
│   ├── period-doubling-eigenvalue.md
│   ├── reservoir-computing-tipping.md
│   ├── pinn-vs-neural-ode.md
│   ├── communication-induced-bifurcation.md
│   └── 12-papers-synthesis.md # 12 篇论文综合分析
│
├── experiments/               # 实验与回测记录
│   ├── entropy-backtest-minute.md    # 分钟级熵回测（失败）
│   ├── four-layer-backtest-2025.md   # 四层系统回测
│   └── multitimeframe-backtest.md    # 多时间框架回测
│
└── decisions/                 # 架构决策记录
    ├── why-daily-not-minute.md       # 为什么选日线不选分钟线
    └── gray-box-over-black-box.md    # 灰箱模型优于黑箱
```

## 页面 Frontmatter 约定

每个 Wiki 页面以 YAML frontmatter 开头：

```yaml
---
title: 页面标题
tags: [entropy, bifurcation, strategy]
confidence: high|medium|speculative
status: active|superseded|archived
sources: [seifert-2025, period-doubling-paper]
created: 2026-04-12
updated: 2026-04-12
open-questions:
  - 未解答的问题
contradictions:
  - 来源 A 说 X，来源 B 说 Y — 待人工判断
---
```

## 操作流程

### 1. Ingest（吸收新来源）
1. 将原始文件放入 `docs/papers/`
2. LLM 阅读来源，提取关键信息
3. 在 `wiki/sources/` 创建摘要页
4. 更新相关的 `concepts/` 和 `entities/` 页面
5. 更新 `wiki/index.md`
6. 追加条目到 `wiki/log.md`

### 2. Query（查询）
1. LLM 先读 `wiki/index.md` 定位相关页面
2. 深入阅读相关页面
3. 综合回答，引用出处
4. 有价值的回答可归档为新 Wiki 页面

### 3. Lint（健康检查）
定期检查：
- 页面间矛盾
- 新来源推翻的旧结论
- 无入站链接的孤儿页面
- 被提及但无独立页面的重要概念
- 缺失的交叉引用
- 可通过搜索填补的数据空白

## 交叉引用约定

- 概念 → 概念：`[entropy](concepts/entropy.md)` 链接到相关概念
- 实体 → 概念：引用其理论基础
- 来源 → 概念/实体：标注支撑或挑战的关系
- 实验 → 实体+概念：记录验证了什么假设
- 决策 → 实验+概念：解释为什么做这个选择
