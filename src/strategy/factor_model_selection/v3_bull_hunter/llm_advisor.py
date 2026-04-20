"""
Bull Hunter v3 — LLM 顾问模块

三个角色:
  A. 因子顾问 (factor_advisor)   — Agent 4 反馈环节: 读诊断报告, 输出因子增删/调参建议
  B. 终筛顾问 (final_filter)     — Agent 3 之后: 对候选做定性筛选
  C. 因子工程师 (factor_engineer) — 初始阶段: 生成新衍生因子公式

使用 DashScope (OpenAI 兼容 API), 环境变量 DASHSCOPE_API_KEY。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import requests

logger = logging.getLogger(__name__)

# ── LLM 配置 ──
DEFAULT_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
DEFAULT_MODEL = "qwen-plus"

# LLM 模式: "api" (调用远程 API) / "file" (从文件读取预生成的响应)
# 通过环境变量 LLM_MODE 或 .env 设置
# file 模式: 将 prompt 写入 {LLM_FILE_DIR}/prompt_{role}.txt,
#            从 {LLM_FILE_DIR}/response_{role}.json 读取响应
LLM_FILE_DIR_DEFAULT = "results/bull_hunter/_llm_exchange"

# ── 因子语义字典 (供 LLM 理解每个因子的含义) ──
FACTOR_DESCRIPTIONS = {
    "perm_entropy_s": "短期置换熵 (5日): 价格序列有序度, 越低越有序 (主力控盘)",
    "perm_entropy_m": "中期置换熵 (10日): 中期价格有序度",
    "perm_entropy_l": "长期置换熵 (20日): 长期价格有序度",
    "entropy_slope": "熵斜率: 置换熵的变化速率, 负值=趋向有序",
    "entropy_accel": "熵加速度: 熵斜率的变化率, 检测拐点",
    "path_irrev_m": "中期路径不可逆性 (10日): 时间序列的不可逆程度, 高=主力活跃",
    "path_irrev_l": "长期路径不可逆性 (20日): 更长时间的不可逆性",
    "dom_eig_m": "中期主特征值 (10日): 临界减速指标, |λ|→1 = 系统临界态",
    "dom_eig_l": "长期主特征值 (20日): 长窗口临界减速",
    "turnover_entropy_m": "中期换手率熵: 换手集中度, 低=筹码集中",
    "turnover_entropy_l": "长期换手率熵: 更长时间的换手集中度",
    "volatility_m": "中期波动率: 10日收益率标准差",
    "volatility_l": "长期波动率: 20日收益率标准差",
    "vol_compression": "波动率压缩: 短/长波动率之比, 低=横盘蓄力",
    "bbw_pctl": "布林带宽度百分位: 当前BBW在历史中的位置, 低=即将突破",
    "vol_ratio_s": "短期量比: 5日均量/20日均量, 高=放量",
    "vol_impulse": "量能脉冲: 成交量突变程度",
    "vol_shrink": "缩量程度: 成交量萎缩幅度",
    "breakout_range": "突破幅度: 价格相对近期区间的突破距离",
    "mf_big_net": "大单净流入: 当日大单买-卖金额",
    "mf_big_net_ratio": "大单净流入比: 大单净流入/总成交额",
    "mf_big_cumsum_s": "短期大单累计: 5日大单净流入累计",
    "mf_big_cumsum_m": "中期大单累计: 10日大单净流入累计",
    "mf_big_cumsum_l": "长期大单累计: 20日大单净流入累计",
    "mf_sm_proportion": "散户资金占比: 小单成交占总成交比例",
    "mf_flow_imbalance": "资金流失衡: (大单买-大单卖)/(大单买+大单卖)",
    "mf_big_momentum": "大单动量: 大单净流入的趋势方向",
    "mf_big_streak": "大单连续天数: 大单净流入的连续方向天数",
    "coherence_l1": "L1相干性: 密度矩阵非对角元素绝对值之和",
    "purity_norm": "纯度: 量子态的纯度指标, 高=确定性强",
    "von_neumann_entropy": "冯诺依曼熵: 量子信息熵, 描述系统混合度",
    "coherence_decay_rate": "相干衰减率: 相干性随时间衰减的速率",
}


# ══════════════════════════════════════════════════════════
#  通用 LLM 调用
# ══════════════════════════════════════════════════════════

def _load_env_file() -> None:
    """从项目根 .env 文件加载环境变量 (仅补充未设置的)。"""
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".env")
    env_path = os.path.normpath(env_path)
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if key and key not in os.environ:
                os.environ[key] = val


# 模块加载时自动读取 .env
_load_env_file()


def _file_mode_call(system_prompt: str, user_msg: str, role: str) -> str:
    """
    文件模式: 从预生成文件读取 LLM 响应。
    1. 保存 prompt 到 {dir}/prompt_{role}.md
    2. 读取 {dir}/response_{role}.json
    3. 如果精确文件不存在, 尝试前缀匹配 (如 final_filter_batch2 → final_filter)
    """
    file_dir = os.environ.get("LLM_FILE_DIR", LLM_FILE_DIR_DEFAULT)
    os.makedirs(file_dir, exist_ok=True)

    if not role:
        role = "unknown"

    # 保存 prompt (供调试/Copilot 阅读)
    prompt_path = os.path.join(file_dir, f"prompt_{role}.md")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(f"# System Prompt\n\n{system_prompt}\n\n# User Message\n\n{user_msg}\n")
    logger.info(f"  📝 [file 模式] prompt 已保存 → {prompt_path}")

    # 读取响应: 精确匹配 → 前缀匹配 (final_filter_batch2 → final_filter)
    resp_path = os.path.join(file_dir, f"response_{role}.json")
    if not os.path.exists(resp_path):
        import re
        base_role = re.sub(r"_batch\d+$", "", role)
        resp_path = os.path.join(file_dir, f"response_{base_role}.json")

    if not os.path.exists(resp_path):
        logger.warning(f"  📝 [file 模式] 响应文件不存在: response_{role}.json")
        return ""

    with open(resp_path, "r", encoding="utf-8") as f:
        content = f.read()

    logger.info(f"  📝 [file 模式] 已读取响应 ← {resp_path}")
    return content


def _llm_call(
    system_prompt: str,
    user_msg: str,
    temperature: float = 0.3,
    base_url: str = "",
    model: str = "",
    api_key: str = "",
    timeout: int = 60,
    retries: int = 2,
    role: str = "",
) -> str:
    """
    LLM 调用 — 支持两种模式:
      - api: 调用远程 OpenAI 兼容 API
      - file: 从文件读取预生成响应 (由 Copilot/人工提供)

    file 模式:
      1. 将 prompt 写入 {LLM_FILE_DIR}/prompt_{role}.md
      2. 从 {LLM_FILE_DIR}/response_{role}.json 读取响应
      3. 如果响应文件不存在, 返回空 (触发降级)
    """
    llm_mode = os.environ.get("LLM_MODE", "api")

    if llm_mode == "file":
        return _file_mode_call(system_prompt, user_msg, role)

    # ── API 模式 ──
    base_url = base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")

    if not api_key:
        logger.warning("未设置 DASHSCOPE_API_KEY, 跳过 LLM 调用")
        return ""

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = 5 * (attempt + 1)
                logger.warning(f"LLM 调用失败 (第{attempt+1}次): {e}, {wait}s 后重试")
                time.sleep(wait)

    logger.error(f"LLM 调用最终失败: {last_err}")
    return ""


def _parse_json_from_llm(text: str) -> dict:
    """从 LLM 回复中提取 JSON 对象。"""
    if not text:
        return {}
    # 找 ```json ... ``` 块
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        # 找第一个 { ... } 块
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"LLM 返回无法解析的 JSON: {text[:200]}")
        return {}


# ══════════════════════════════════════════════════════════
#  方案 A: 因子顾问 — Agent 4 反馈环节
# ══════════════════════════════════════════════════════════

FACTOR_ADVISOR_SYSTEM = """你是 A 股量化策略的因子分析专家。
你的任务是分析模型诊断报告，给出因子增删和调参建议。

当前系统使用 32 个日线因子 (熵/波动率/资金流/量子态) 训练 LightGBM/XGBoost 分类器，
预测 30%/100%/200% 涨幅的股票。

你需要返回严格的 JSON 格式:
```json
{
  "drop_factors": ["factor_name1", "factor_name2"],
  "drop_reasons": {"factor_name1": "原因", "factor_name2": "原因"},
  "boost_factors": ["factor_name"],
  "boost_reasons": {"factor_name": "为什么这个因子重要"},
  "new_factor_ideas": [
    {"name": "momentum_20d", "formula": "close / close.shift(20) - 1", "rationale": "..."},
    {"name": "industry_rs", "formula": "stock_return_20d - industry_mean_return_20d", "rationale": "..."}
  ],
  "model_suggestion": "keep|switch_xgboost|switch_rf",
  "analysis": "简短的定性分析 (2-3 句话)"
}
```

规则:
- drop_factors: 只从提供的因子列表中选择, 最多剔除 5 个
- new_factor_ideas: 公式必须是合法的 pandas 表达式, 只能用 close/open/high/low/volume/amount/turnover_rate 列
- 不要编造不存在的因子名
- 每个建议都要有明确的金融逻辑依据"""


def run_factor_advisor(
    training_meta: dict,
    miss_diagnosis: dict,
    missed_bull_replay: dict,
    factor_blind_spots: list[dict],
    blind_industries: list[dict],
    current_model: str = "lgbm",
    current_drop_factors: list[str] | None = None,
) -> dict:
    """
    方案 A: LLM 因子顾问 — 分析 Agent 4 诊断结果, 输出因子/模型建议。

    Returns:
        {
            "drop_factors": [...],
            "boost_factors": [...],
            "new_factor_ideas": [...],
            "model_suggestion": "keep|switch_xgboost|switch_rf",
            "analysis": "...",
            "raw_response": "...",
        }
    """
    current_drop_factors = current_drop_factors or []

    # ── 构建 prompt ──
    # 1. 因子列表 + 描述
    factor_desc_lines = []
    for f, desc in FACTOR_DESCRIPTIONS.items():
        marker = " [已剔除]" if f in current_drop_factors else ""
        factor_desc_lines.append(f"  - {f}: {desc}{marker}")
    factor_desc_text = "\n".join(factor_desc_lines)

    # 2. 模型 feature importance
    importance_text = ""
    for tname in ["200pct", "100pct", "30pct"]:
        tmeta = training_meta.get(tname, {})
        top_feats = tmeta.get("top_features", [])[:10]
        if top_feats:
            feat_lines = [f"    {i+1}. {f['name']} (importance={f['importance']:.0f})"
                          for i, f in enumerate(top_feats)]
            auc = tmeta.get("val_auc", 0)
            importance_text += f"\n  {tname} 模型 (AUC={auc:.3f}):\n" + "\n".join(feat_lines)

    # 3. 诊断结果
    miss_rate = miss_diagnosis.get("miss_rate", 0)
    diagnosis = missed_bull_replay.get("diagnosis", "unknown")
    n_replayed = missed_bull_replay.get("n_replayed", 0)
    avg_prob_200 = missed_bull_replay.get("avg_prob_200", 0)

    # 4. 因子盲区 (top 5)
    blind_spot_text = ""
    if factor_blind_spots:
        lines = []
        for spot in factor_blind_spots[:8]:
            f = spot["factor"]
            diff = spot["diff"]
            nd = spot["norm_diff"]
            direction = "高于" if diff > 0 else "低于"
            lines.append(f"    - {f}: 漏选大牛股{direction}入选股 (标准化差异={nd:.2f})")
        blind_spot_text = "\n".join(lines)

    # 5. 行业盲区
    industry_text = ""
    if blind_industries:
        lines = [f"    - {ind['industry']}: 漏选{ind['n_missed']}只, 捕获{ind['n_caught']}只"
                 for ind in blind_industries[:5]]
        industry_text = "\n".join(lines)

    # 6. 漏选大牛股样本
    scores = missed_bull_replay.get("scores", [])
    bull_samples = ""
    if scores:
        lines = []
        for s in scores[:5]:
            lines.append(f"    - {s['symbol']} {s.get('name','')} "
                        f"涨幅={s.get('gain_120d',0):.0%} → "
                        f"p30={s.get('prob_30',0):.4f} "
                        f"p100={s.get('prob_100',0):.4f} "
                        f"p200={s.get('prob_200',0):.4f}")
        bull_samples = "\n".join(lines)

    user_msg = f"""## 模型诊断报告

当前模型: {current_model}
Miss Rate: {miss_rate:.0%} (市场 Top30 大牛股漏选比例)
诊断结论: {diagnosis}
漏选复盘: {n_replayed} 只大牛股, 平均 prob_200={avg_prob_200:.4f}
已剔除因子: {', '.join(current_drop_factors) if current_drop_factors else '无'}

## 当前因子列表 (共 {len(FACTOR_DESCRIPTIONS)} 个)
{factor_desc_text}

## 模型特征重要性
{importance_text}

## 因子盲区分析 (漏选大牛股 vs 入选股的因子差异)
{blind_spot_text or '  无数据'}

## 行业盲区
{industry_text or '  无数据'}

## 漏选大牛股样本 (模型给分极低)
{bull_samples or '  无数据'}

请分析以上诊断报告，给出:
1. 应该剔除的因子 (误导模型的)
2. 应该重点提权的因子
3. 建议新增的衍生因子 (给出 pandas 计算公式)
4. 是否需要换模型
5. 简短的定性分析"""

    logger.info("  🧠 LLM 因子顾问: 分析诊断报告...")
    raw = _llm_call(FACTOR_ADVISOR_SYSTEM, user_msg, temperature=0.3, role="factor_advisor")

    if not raw:
        logger.warning("  LLM 因子顾问: 无响应, 回退到规则引擎")
        return {}

    result = _parse_json_from_llm(raw)
    result["raw_response"] = raw

    # 校验 drop_factors 只包含合法因子名
    valid_factors = set(FACTOR_DESCRIPTIONS.keys())
    if "drop_factors" in result:
        result["drop_factors"] = [f for f in result["drop_factors"] if f in valid_factors]

    if result.get("analysis"):
        logger.info(f"  🧠 LLM 分析: {result['analysis'][:100]}")
    if result.get("drop_factors"):
        logger.info(f"  🧠 建议剔除: {', '.join(result['drop_factors'])}")
    if result.get("new_factor_ideas"):
        for nf in result["new_factor_ideas"][:3]:
            logger.info(f"  🧠 建议新因子: {nf.get('name', '?')} = {nf.get('formula', '?')[:60]}")

    return result


# ══════════════════════════════════════════════════════════
#  方案 B: 终筛顾问 — Agent 3 之后定性筛选
# ══════════════════════════════════════════════════════════

FINAL_FILTER_SYSTEM = """你是 A 股投资经理，擅长从量化模型候选中做定性筛选。

你将收到一批股票候选（含因子值、行业、模型评分），需要做最终判断。

返回 JSON 格式:
```json
{
  "approved": ["symbol1", "symbol2"],
  "rejected": ["symbol3"],
  "reasons": {
    "symbol1": "看好理由",
    "symbol3": "拒绝理由"
  },
  "risk_warnings": ["整体风险提示1", "..."]
}
```

筛选准则:
- 同一行业不超过 3 只 (分散风险)
- ST/退市风险股直接剔除
- 行业基本面恶化的股票标注风险
- 因子矛盾的候选 (如高熵+高波动 = 混乱) 降级
- 保持原有评级顺序, 不要颠覆模型排名
- approved 数量应 >= 原候选的 60%"""


def run_final_filter(
    predictions: "pd.DataFrame",
    factor_snapshot: "pd.DataFrame",
    scan_date: str,
    batch_size: int = 15,
) -> dict:
    """
    方案 B: LLM 终筛 — 对 Agent 3 输出的候选做定性筛选。

    Returns:
        {
            "approved": [symbol, ...],
            "rejected": [symbol, ...],
            "reasons": {symbol: reason},
            "risk_warnings": [...],
            "raw_responses": [...],
        }
    """
    import pandas as pd

    if predictions.empty:
        return {"approved": [], "rejected": [], "reasons": {}, "risk_warnings": []}

    all_approved = []
    all_rejected = []
    all_reasons = {}
    all_warnings = []
    all_raw = []

    symbols = predictions["symbol"].tolist()

    # 分批处理 (每批 batch_size 只)
    for batch_start in range(0, len(symbols), batch_size):
        batch_syms = symbols[batch_start:batch_start + batch_size]
        batch_preds = predictions[predictions["symbol"].isin(batch_syms)]

        # 构建候选描述
        candidate_lines = []
        for _, row in batch_preds.iterrows():
            sym = row["symbol"]
            name = row.get("name", "")
            grade = row.get("grade", "?")
            industry = row.get("industry", "未知")
            p30 = row.get("prob_30", 0)
            p100 = row.get("prob_100", 0)
            p200 = row.get("prob_200", 0)

            # 关键因子值
            factor_vals = ""
            if factor_snapshot is not None and sym in factor_snapshot.index:
                key_factors = ["perm_entropy_l", "path_irrev_l", "volatility_l",
                               "mf_big_cumsum_l", "vol_compression", "bbw_pctl"]
                vals = []
                for f in key_factors:
                    if f in factor_snapshot.columns:
                        v = factor_snapshot.loc[sym, f]
                        if pd.notna(v):
                            short_name = f.replace("perm_entropy_l", "熵").replace(
                                "path_irrev_l", "不可逆").replace(
                                "volatility_l", "波动率").replace(
                                "mf_big_cumsum_l", "大单累计").replace(
                                "vol_compression", "波动压缩").replace(
                                "bbw_pctl", "BBW分位")
                            vals.append(f"{short_name}={v:.3f}")
                factor_vals = ", ".join(vals)

            candidate_lines.append(
                f"  [{grade}] {sym} {name} ({industry}) — "
                f"p30={p30:.2%} p100={p100:.2%} p200={p200:.2%}"
                f"\n      因子: {factor_vals}"
            )

        user_msg = f"""扫描日期: {scan_date}
本批候选 ({len(batch_syms)} 只):

{chr(10).join(candidate_lines)}

请对以上候选做定性筛选。"""

        logger.info(f"  🧠 LLM 终筛: 第 {batch_start//batch_size+1} 批 "
                    f"({len(batch_syms)} 只)...")
        raw = _llm_call(FINAL_FILTER_SYSTEM, user_msg, temperature=0.2,
                       role=f"final_filter_batch{batch_start//batch_size+1}")
        all_raw.append(raw)

        if not raw:
            # LLM 失败时全部通过
            all_approved.extend(batch_syms)
            continue

        parsed = _parse_json_from_llm(raw)
        all_approved.extend(parsed.get("approved", batch_syms))
        all_rejected.extend(parsed.get("rejected", []))
        all_reasons.update(parsed.get("reasons", {}))
        all_warnings.extend(parsed.get("risk_warnings", []))

    # 去重
    approved_set = set(all_approved) - set(all_rejected)
    # 保持原始顺序
    final_approved = [s for s in symbols if s in approved_set]
    final_rejected = [s for s in symbols if s not in approved_set]

    n_total = len(symbols)
    n_approved = len(final_approved)
    n_rejected = len(final_rejected)
    logger.info(f"  🧠 LLM 终筛完成: {n_approved}/{n_total} 通过, {n_rejected} 剔除")

    return {
        "approved": final_approved,
        "rejected": final_rejected,
        "reasons": all_reasons,
        "risk_warnings": list(set(all_warnings)),
        "raw_responses": all_raw,
    }


# ══════════════════════════════════════════════════════════
#  方案 C: 因子工程师 — 生成新衍生因子公式
# ══════════════════════════════════════════════════════════

FACTOR_ENGINEER_SYSTEM = """你是量化因子工程专家，精通 A 股市场微观结构。

你的任务是基于现有数据列，设计新的衍生因子来捕捉大牛股特征。

可用的原始数据列 (每只股票的日线 CSV):
  close, open, high, low, volume, amount, turnover_rate, pct_chg,
  net_mf_amount (资金流净额), buy_lg_amount (特大单买), sell_lg_amount (特大单卖),
  buy_md_amount (中单买), sell_md_amount (中单卖)

已有因子:
  {existing_factors}

返回 JSON 格式:
```json
{
  "new_factors": [
    {
      "name": "momentum_20d",
      "formula": "close / close.shift(20) - 1",
      "description": "20日动量: 当前价格相对20日前的涨跌幅",
      "rationale": "大牛股启动前通常有持续正动量",
      "category": "momentum"
    }
  ]
}
```

规则:
- formula 必须是合法的 pandas Series 表达式 (df["col"] 形式或 col.method() 形式)
- 只能用上述可用列, 不要引入不存在的数据
- 每个因子都要有明确的金融逻辑
- 生成 5-10 个因子, 覆盖不同类别 (动量/趋势/行业/波动/资金)
- 因子名用小写下划线格式
- category 取值: momentum, trend, volatility, money_flow, structure"""


def run_factor_engineer(
    existing_factors: list[str],
    miss_diagnosis: dict | None = None,
    factor_blind_spots: list[dict] | None = None,
    blind_industries: list[dict] | None = None,
) -> list[dict]:
    """
    方案 C: LLM 因子工程师 — 生成新衍生因子公式。

    Returns:
        [{name, formula, description, rationale, category}, ...]
    """
    # 构建上下文
    existing_text = ", ".join(existing_factors)

    context_lines = []
    if miss_diagnosis:
        miss_rate = miss_diagnosis.get("miss_rate", 0)
        context_lines.append(f"当前 Miss Rate: {miss_rate:.0%}")

    if factor_blind_spots:
        lines = []
        for spot in factor_blind_spots[:5]:
            f = spot["factor"]
            desc = FACTOR_DESCRIPTIONS.get(f, "")
            diff = spot["diff"]
            direction = "偏高" if diff > 0 else "偏低"
            lines.append(f"  - {f} ({desc}): 漏选大牛股{direction}")
        context_lines.append("因子盲区:\n" + "\n".join(lines))

    if blind_industries:
        inds = [f"{i['industry']}(漏{i['n_missed']}只)" for i in blind_industries[:5]]
        context_lines.append(f"行业盲区: {', '.join(inds)}")

    context_text = "\n".join(context_lines) if context_lines else "首次生成, 无诊断信息"

    system = FACTOR_ENGINEER_SYSTEM.replace("{existing_factors}", existing_text)

    user_msg = f"""## 任务
请设计 5-10 个新的衍生因子, 重点补充以下短板:
1. 动量/趋势因子 (当前完全缺失)
2. 行业相对强度
3. 主力洗盘 vs 真实出货的识别
4. 价格形态因子

## 当前系统诊断
{context_text}

## 设计要求
- 公式用 pandas 语法, 假设 df 是单只股票的日线 DataFrame
- 行业相对强度需要标注 "需要全市场数据" (后续由系统处理)"""

    logger.info("  🧠 LLM 因子工程师: 生成新因子公式...")
    raw = _llm_call(system, user_msg, temperature=0.5, role="factor_engineer")

    if not raw:
        logger.warning("  LLM 因子工程师: 无响应")
        return []

    parsed = _parse_json_from_llm(raw)
    factors = parsed.get("new_factors", [])

    # 校验
    valid_factors = []
    for f in factors:
        name = f.get("name", "")
        formula = f.get("formula", "")
        if not name or not formula:
            continue
        # 基本安全检查: 不允许 import, exec, eval, os, sys
        dangerous = ["import ", "exec(", "eval(", "os.", "sys.", "__", "subprocess"]
        if any(d in formula for d in dangerous):
            logger.warning(f"  因子 {name} 公式包含危险操作, 跳过: {formula[:50]}")
            continue
        valid_factors.append(f)

    logger.info(f"  🧠 LLM 因子工程师: 生成 {len(valid_factors)} 个新因子")
    for f in valid_factors:
        logger.info(f"    - {f['name']}: {f.get('description', '')[:50]}")

    return valid_factors


# ══════════════════════════════════════════════════════════
#  因子公式执行器 (安全沙箱)
# ══════════════════════════════════════════════════════════

# 白名单: 允许在因子公式中使用的函数/属性
_SAFE_NAMES = {
    "shift", "rolling", "mean", "std", "max", "min", "sum",
    "pct_change", "diff", "abs", "rank", "ewm", "expanding",
    "fillna", "clip", "apply", "cumsum", "cummax", "cummin",
    "median", "quantile", "corr", "cov",
}

# 允许的列名
_ALLOWED_COLS = {
    "close", "open", "high", "low", "volume", "amount",
    "turnover_rate", "pct_chg",
    "net_mf_amount", "buy_lg_amount", "sell_lg_amount",
    "buy_md_amount", "sell_md_amount",
}


def compute_derived_factor(df: "pd.DataFrame", formula: str, name: str) -> "pd.Series | None":
    """
    安全执行 LLM 生成的因子公式。

    Args:
        df: 单只股票日线 DataFrame
        formula: pandas 表达式字符串
        name: 因子名

    Returns:
        Series 或 None (执行失败时)
    """
    import pandas as pd
    import numpy as np

    # 安全检查
    dangerous = ["import ", "exec(", "eval(", "os.", "sys.", "__",
                 "subprocess", "open(", "write(", "read(", "compile("]
    if any(d in formula for d in dangerous):
        logger.warning(f"因子 {name} 公式包含危险操作, 跳过")
        return None

    # 构建安全执行环境
    safe_env = {"pd": pd, "np": np}
    for col in _ALLOWED_COLS:
        if col in df.columns:
            safe_env[col] = df[col]

    try:
        result = eval(formula, {"__builtins__": {}}, safe_env)
        if isinstance(result, pd.Series):
            return result
        elif isinstance(result, (int, float)):
            return pd.Series(result, index=df.index)
        else:
            logger.warning(f"因子 {name} 返回非 Series 类型: {type(result)}")
            return None
    except Exception as e:
        logger.warning(f"因子 {name} 公式执行失败: {e}")
        return None
