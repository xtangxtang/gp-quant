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

## [2026-04-13] create | 熵惜售分岔突破策略

- 新建策略 `src/strategy/entropy_accumulation_breakout/`（5 个源文件 + README）
- 三阶段状态机: 惜售吸筹 → 分岔突破 → 结构崩塌退出
- 搜索并引入 3 篇新论文作为理论支撑:
  - Fan et al. (2025) — KLD 不可逆性检测金融不稳定
  - Dmitriev et al. (2025) — 股票市场自组织到相变边缘
  - Yan et al. (2023) — 热力学预测分岔与非平衡相变
- 创建实体页 `wiki/entities/entropy-accumulation-breakout.md`
- 创建来源页 `wiki/sources/fan-2025-irreversibility.md`
- 创建来源页 `wiki/sources/dmitriev-2025-self-organization.md`
- 创建来源页 `wiki/sources/yan-2023-thermodynamic-bifurcation.md`
- 更新 `index.md`、`overview.md`、`WIKI.md` 目录结构
- 烟雾测试通过: 50 只 SH 主板股票扫描 + 前瞻回测均无错误

## [2026-04-13] create | 大盘趋势判断系统（从小见大）

- 新建策略 `src/strategy/market_trend/`（6 个源文件）
  - config.py — 7维权重配置与趋势判定门槛
  - data_loader.py — 批量加载个股日线/指数/涨跌停/两融/SHIBOR/行业分类
  - micro_indicators.py — 5维微观指标引擎 (广度/资金流/波动/熵/动量)
  - macro_indicators.py — 2维宏观指标引擎 (杠杆资金/流动性)
  - trend_engine.py — 主引擎: 7维评分 → 综合得分 → 趋势判定
  - run_market_trend.py — CLI 入口
- 新建运行脚本 `scripts/run_market_trend.sh`
- 核心思想: 不看指数K线，从5000+只个股微观行为聚合出大盘真实状态
- 数据源: tushare-daily-full + tushare-index-daily + tushare-stk_limit + tushare-margin + tushare-shibor + tushare-index_member_all
- 冒烟测试通过: 5182只A股 × 7天扫描无错误

## [2026-04-13] experiment | 大盘趋势回测 2023H2-2026Q2

- 回测区间: 20230703~20260410, 672个交易日, 5182只A股
- 结果位置: `results/market_trend_2024/`
- 详细分析报告: `src/strategy/market_trend/report/market_trend_analysis_2023H2_2026Q2.md`
- 核心发现:
  - DOWN占44%, NEUTRAL占43%, UP仅7%——A股多数时间不在上涨
  - STRONG_DOWN后20日平均涨+3.25%, 胜率58%——恐慌即机会
  - 杠杆是核心驱动: 两融14000→26000亿支撑3年慢牛
  - 资金流持续为负是常态——机构长期派发，散户+杠杆买入
  - 熵得分始终为负——A股缺乏持续有序趋势
- 关键时间节点:
  - 2024.02.05 杠杆得分-1.000 = 绝对底部
  - 2024.10.08 杠杆得分+1.000 = 924行情高潮
  - 2025Q3 最健康上涨段 (广度+0.50, 杠杆+0.55, 无STRONG_DOWN)
- 创建实体页 `wiki/entities/market-trend-system.md`
- 创建实验页 `wiki/experiments/market-trend-backtest-2024.md`
- 更新 `WIKI.md` 目录、`index.md`、`overview.md` (新增策略+数据源+教训)
