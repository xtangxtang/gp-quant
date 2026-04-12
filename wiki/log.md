# GP-Quant Wiki — 时间线日志

> Append-only 记录。每条以 `## [日期] 操作类型 | 标题` 开头。  
> 用法: `grep "^## \[" wiki/log.md | tail -10` 查看最近 10 条。

---

## [2026-04-12] init | Wiki 初始化

- 基于 Karpathy LLM Wiki 模式创建三层架构
- Raw Sources: `docs/papers/`（12 篇论文 PDF + 分析报告）
- Wiki: `wiki/`（本目录）
- Schema: `wiki/WIKI.md`
- 创建初始页面：overview, 9 concepts, 7 entities, 6 sources, 3 experiments, 2 decisions
- 从现有文档 `docs/complexity_theory_notes.md`、`docs/complexity_research_12_papers_conclusion.md`、`memory/entropy_backtest_results.md` 编译核心知识

## [2026-04-12] ingest | 12 篇复杂性理论论文

- 来源: `docs/papers/*.pdf` + `docs/papers/12_papers_deep_analysis.md`
- 创建 6 个来源摘要页 (`wiki/sources/`)
- 提取 8 个核心共识写入概念页
- 标注 6 个「不可直接套用」的物理类比
- 关键洞见: 市场熵 = 不可逆性代理而非真实热力学量

## [2026-04-12] ingest | 熵回测结果（分钟级失败）

- 来源: `memory/entropy_backtest_results.md`
- 创建决策页 `why-daily-not-minute.md`
- 关键发现: 熵因子在分钟级无预测力（35% 胜率），日线/周线级别有效
- 更新实体页 `tick-entropy-module.md` 添加适用时间尺度警告
