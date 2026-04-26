# 更高维度审视：Transformer 判断状态是否可行 + 路线选择

> 日期: 2026-04-25
> 背景: Phase 1 (IC=0.058) 完成，Phase 2 (CrossSectionalEncoder) 全面失败后的方向性讨论

## 一、Transformer 判断 5 种状态本质上可行吗？

**结论：理论可行，但当前的"5 种状态"定义本身有问题。**

### 1. Transformer 在金融预测的天花板

学术界和工业界共识：
- 股票收益预测的 IC 上限约 0.05~0.10（日频，单因子甚至更低）
- 顶级机构（Renaissance, Two Sigma）多年的 IC 也就 0.05~0.08
- Transformer 没有打破这个天花板，只是提供了更好的非线性拟合能力
- **数据量 << 文本/图像** 是根本约束（A 股每天 5000 股 × 365 天 ≈ 180万样本，vs LLM 万亿 token）

Phase 1 的 Eval IC=0.058 **已经接近这个理论上限**，不是路径问题，是天花板问题。

### 2. "5 种状态"的根本悖论

最初目标是预测 `accumulation/breakout/hold/collapse/idle` 5 种状态。但这些状态的定义来自规则（state_evaluator.py 的阈值），**它们本身就是因子的函数**。让 Transformer 学这种状态等于：

```
Transformer 学习: f(因子序列) → g(因子)  其中 g 是规则函数
```

这是**自我循环的伪标签**（Vision 文档自己也指出了），最多学到规则的近似。要真正有价值，应该让 Transformer 直接预测**未来收益**，状态只是事后贴标签的可视化语言。

**Phase 1 已经做对了这件事**：删除 state_cls_head，改用收益分位数回归。所以 Phase 1 的 0.058 IC 是 Transformer 在这条路上能到的合理值。

### 3. 现在这条路是否正确？

**部分正确，但方向需要调整。**

| 决策 | 评价 |
|------|------|
| 删除 state_cls_head 用纯监督 | ✅ 正确 |
| Quantile + IC 多目标 | ✅ 正确 |
| Walk-Forward 评估 | ✅ 正确 |
| 加大模型容量到 575K 参数 | ⚠️ 边界 — 数据量不够，已过拟合 |
| CrossSectionalEncoder | ❌ 加错了维度 |
| 加截面位置编码到 stocks | ❌ A 股没有"股票排序"这个稳定结构 |

**真正应该走的方向**：不是把模型变大，而是**改进输入数据质量**。Vision 文档第二节（多尺度时间窗、因子分组嵌入、PatchTST 价格分支）才是真正的杠杆点 — 这些都还没做。

---

## 二、路线 A vs 路线 B 对比

### 路线 A：止损，回退到 Phase 1

**优点**：
- Eval IC=0.058 已达标，可直接进入实盘验证
- 节省时间，立刻产出回测/实盘信号
- Phase 1 模型已经训好、验证过、save 到磁盘
- 避免在已知天花板附近继续 marginal improvement

**缺点**：
- 不知道截面 batch + IC loss 的真实价值（一个未解之谜）
- 失去了改进的可能性 — 如果 Phase 2.2 真的能 IC=0.07，相当于 +20% 收益
- Vision 文档第二节（多尺度、PatchTST）的潜力完全没探索

**适用场景**：想立刻做实盘 / 把精力转向其他策略

---

### 路线 B：做 Phase 2.2 消融

**优点**：
- 改动极小（2 行代码：换 forward 路径）
- 1 次训练（约 2.5 小时）就能定论
- 如果有效，建立了截面训练的方法论，未来可复用
- 即使无效，也彻底排除了一个变量，未来不会再纠结

**缺点**：
- 大概率结果与 Phase 1 接近（IC=0.04~0.07），改进不明显
- 即使 Phase 2.2 IC=0.06，也只比 Phase 1 高 0.002，统计显著性存疑
- 真正的杠杆在 Vision 文档第二节，不在数据组织方式
- 持续投入 Transformer 优化容易陷入"再调一次就好了"的陷阱

**适用场景**：想完整收尾 Phase 2 这条线，避免遗留疑问

---

## 三、最终判断

**推荐路线 A + 探索新方向。**

理由：
1. Phase 1 已经接近 Transformer 在这个任务上的天花板
2. 路线 B 的预期收益太小（最多 +0.01 IC），不值得 2.5 小时训练 + 大量分析
3. 真正的差异化机会在别处：
   - **多策略融合**：Phase 1 Transformer 信号 + 现有的 entropy-accumulation-breakout / multitimeframe-resonance 做集成
   - **Phase 1 模型直接进回测**：用 `--backtest` 跑 2025 年实际收益，看 Sharpe / max drawdown，这比再调 IC 重要得多
   - **特征质量**：Vision 文档第二节的因子分组嵌入、价格 patch 是真正的 SOTA 路线，但代价大（重写数据管道）

---

## 四、推荐行动

**短期（今天）**：路线 A，把 Phase 1 模型接入回测，看实盘表现。

```bash
python -m src.strategy.adaptive_state_machine.run_adaptive_state_machine \
  --backtest --start_date 20250101 --end_date 20260331 \
  --interval_days 20 --top_n 10
```

**中期（本周）**：如果 Phase 1 回测的 Sharpe > 1.0，把它纳入策略池；否则考虑 Vision 第二节的真正升级。

**Phase 2.2 不做了**（除非执着于完整性）。把那 2.5 小时留给真正的回测验证。
