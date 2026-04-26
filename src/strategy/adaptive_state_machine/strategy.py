"""
Adaptive State Machine — 统一策略类

单一入口, 内部组件:
  feature.py         → 因子计算
  attention_learner  → Transformer 推理 (attention 权重 + 分位数回归)
  state_evaluator.py → 规则判定 (rules 模式 fallback)

流程: 因子计算 → Transformer 推理 → 截面排序

用法:
  strategy = AdaptiveStateMachine(daily_dir="...", data_root="...")
  result = strategy.scan("20250102")           # 单日扫描
  df = strategy.backtest("20250101", "20260331")  # 回测
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from .config import AdaptiveConfig, StockState
from .feature import FactorCalculator
from .state_evaluator import StateEvaluator, StateResult

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════

def load_trade_cal(data_root: str) -> list[str]:
    """加载交易日历, 返回 YYYYMMDD 列表"""
    cal_path = os.path.join(data_root, "tushare-trade_cal", "trade_cal.csv")
    if not os.path.exists(cal_path):
        return []
    df = pd.read_csv(cal_path)
    if "cal_date" in df.columns and "is_open" in df.columns:
        df = df[df["is_open"] == 1]
        return sorted(df["cal_date"].astype(str).tolist())
    return []


# ═════════════════════════════════════════════════════════
# 统一策略类
# ═════════════════════════════════════════════════════════

class AdaptiveStateMachine:
    """自适应状态机策略 — 单一日线扫描 + 回测入口。"""

    def __init__(
        self,
        daily_dir: str,
        data_root: str = "",
        cache_dir: str = "",
        max_workers: int = 28,
        attention_model_path: str = "",
        attention_alpha: float = 1.0,
        cls_mode: str = "rules",
    ):
        self.daily_dir = daily_dir
        self.data_root = data_root
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.attention_model_path = attention_model_path
        self.attention_alpha = attention_alpha
        self.cls_mode = cls_mode  # "rules" or "model"
        self._attention_learner = None
        # 因子截面历史缓存 (用于构建真实序列)
        self._factor_history: list[pd.DataFrame] = []
        self._seq_len = 60  # 默认序列长度 (与模型匹配)

    def _get_attention_learner(self):
        """Lazy load attention learner"""
        if self._attention_learner is None and self.attention_model_path:
            from .attention_learner import AttentionLearner
            self._attention_learner = AttentionLearner(
                model_path=self.attention_model_path,
                seq_len=self._seq_len,
            )
            if not self._attention_learner.load_model():
                logger.warning("Failed to load attention model, using IC weights only")
                self._attention_learner = None
            else:
                # 同步模型的实际序列长度
                self._seq_len = self._attention_learner.seq_len
        return self._attention_learner

    def _build_real_sequences(
        self,
        cross_section: pd.DataFrame,
        seq_len: Optional[int] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        从因子历史缓存构建真实时序数据。

        Args:
            cross_section: 当前截面数据
            seq_len: 序列长度 (默认用 self._seq_len)

        Returns:
            sequences: (n_stocks, seq_len, n_factors)
            symbols: 对应的股票列表
        """
        if seq_len is None:
            seq_len = self._seq_len

        # 取数值列
        numeric_cols = [c for c in cross_section.columns
                        if cross_section[c].dtype in (np.float32, np.float64, np.int32, np.int64)]

        # 取当前截面中的所有股票
        symbols = list(cross_section.index)
        factor_values = cross_section[numeric_cols].values.astype(np.float32)
        n_stocks, n_factors = factor_values.shape

        # 构建序列: 从历史缓存中取最近 seq_len-1 个截面 + 当前截面
        history = self._factor_history[-(seq_len - 1):]  # 最多取 seq_len-1 个历史
        n_history = len(history)
        actual_len = n_history + 1  # 历史 + 当前

        sequences = np.zeros((n_stocks, seq_len, n_factors), dtype=np.float32)

        # 历史部分: 从缓存中按 symbol 对齐
        for h_idx, hist_df in enumerate(history):
            seq_idx = h_idx  # 从 0 开始
            for i, sym in enumerate(symbols):
                if sym in hist_df.index:
                    row = hist_df.loc[sym]
                    vals = row[numeric_cols].values.astype(np.float32)
                    sequences[i, seq_idx, :] = vals
                else:
                    sequences[i, seq_idx, :] = np.nan

        # 当前截面
        sequences[:, actual_len - 1, :] = factor_values

        # 前向填充: 如果历史缺失则用最近的可用值
        for i in range(n_stocks):
            for j in range(1, seq_len):
                if np.any(np.isnan(sequences[i, j, :])):
                    # 用前一个时间步填充
                    sequences[i, j, :] = sequences[i, j - 1, :]
            # 如果第一个时间步仍为 NaN，用当前截面填充
            if np.any(np.isnan(sequences[i, 0, :])):
                sequences[i, 0, :] = sequences[i, actual_len - 1, :]

        # NaN 填充 (全列用均值)
        col_means = np.nanmean(sequences.reshape(-1, n_factors), axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        for j in range(n_factors):
            mask = np.isnan(sequences[:, :, j])
            sequences[mask, j] = col_means[j]

        return sequences, symbols

    # ── 单日扫描 ─────────────────────────────────────────

    def scan(
        self,
        scan_date: str,
        config: Optional[AdaptiveConfig] = None,
        config_path: str = "",
        output_dir: str = "",
        evaluator: Optional[StateEvaluator] = None,
    ) -> tuple[list, dict, AdaptiveConfig]:
        """
        执行单次扫描 (Transformer 化流程)。

        流程: 因子计算 → Transformer 推理 → 截面排序 / 规则判定

        Args:
            scan_date: YYYYMMDD
            config: 已有配置 (回测传入, 单日扫描留空)
            config_path: 配置持久化路径
            output_dir: 信号输出目录
            evaluator: 状态评估器 (rules 模式复用, 保留 hold 状态)

        Returns:
            (results, summary_dict, config)
        """
        if config is None:
            config = AdaptiveConfig()
            config.last_updated = scan_date

        if evaluator is None:
            evaluator = StateEvaluator(config=config)

        # 1. 因子计算
        calculator = FactorCalculator(
            daily_dir=self.daily_dir,
            data_root=self.data_root,
            cache_dir=self.cache_dir,
            max_workers=self.max_workers,
        )
        daily_results = calculator.compute_all(scan_date=scan_date)
        cross_section = calculator.build_cross_section(daily_results)

        if cross_section.empty:
            return [], {"scan_date": scan_date, "total_stocks": 0, "error": "no data"}, config

        # 缓存当前截面到历史 (用于后续扫描构建真实序列)
        self._factor_history.append(cross_section)

        # 2. 提取 Attention (始终执行)
        attn_weights = None
        model_predictions = None
        attn_learner = self._get_attention_learner()
        if attn_learner is not None:
            try:
                attn_weights, model_predictions = self._extract_attention_from_cross_section(
                    attn_learner, cross_section,
                )
                if attn_weights:
                    logger.info(
                        f"Attention weights extracted (alpha={self.attention_alpha}), "
                        f"top3={sorted(attn_weights.items(), key=lambda x: x[1], reverse=True)[:3]}"
                    )
            except Exception as e:
                logger.warning(f"Attention extraction failed: {e}")
                attn_weights = None
                model_predictions = None

        # 3. 状态判定
        if self.cls_mode == "model" and attn_learner is not None:
            # Transformer 分位数回归 → 截面排序
            results = self._predict_states_from_model(
                attn_learner, cross_section, daily_results, model_predictions,
            )
        else:
            # 现有阈值规则 (传入模型预测校准)
            evaluator.config = config
            results = evaluator.evaluate_all(
                cross_section, daily_results, config, model_predictions,
            )

        signal_counts = {s.value: 0 for s in StockState}
        for r in results:
            signal_counts[r.state.value] = signal_counts.get(r.state.value, 0) + 1

        logger.info(
            f"Signals: {signal_counts.get('accumulation', 0)} accumulation, "
            f"{signal_counts.get('breakout', 0)} breakout, "
            f"{signal_counts.get('hold', 0)} hold, "
            f"{signal_counts.get('collapse', 0)} collapse"
        )

        # 5. 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._save_signals(results, scan_date, output_dir)

        if config_path:
            config.last_updated = scan_date
            config.save(config_path)
            logger.info(f"Saved config v{config.version} to {config_path}")

        # 6. 打印摘要
        _print_scan_summary(results, scan_date)

        summary = {
            "scan_date": scan_date,
            "config_version": config.version,
            "total_stocks": len(results),
            "accumulation": signal_counts.get("accumulation", 0),
            "breakout": signal_counts.get("breakout", 0),
            "hold": signal_counts.get("hold", 0),
            "collapse": signal_counts.get("collapse", 0),
        }

        return results, summary, config

    def _extract_attention_from_cross_section(
        self,
        attn_learner,
        cross_section: pd.DataFrame,
    ) -> tuple[dict[str, float], Optional[dict[str, dict]]]:
        """
        从真实历史序列提取 attention 权重 + 模型预测。
        使用因子历史缓存构建 20 天序列, 而非伪序列。

        Returns:
            (attention_weights, model_predictions)
            attention_weights: {factor: weight}
            model_predictions: {symbol: {"pred_return": float, "pred_up_prob": float}}
        """
        if attn_learner.model is None:
            return {}, None

        n_model_factors = attn_learner.model.n_factors
        stock_symbols = list(cross_section.index)

        # 从历史缓存构建真实序列
        sequences, symbols = self._build_real_sequences(cross_section, seq_len=attn_learner.seq_len)

        # 截取模型需要的因子数
        sequences = sequences[:, :, :n_model_factors]

        # 采样 (避免过多股票)
        max_stocks = min(500, len(symbols))
        sequences = sequences[:max_stocks]

        # 提取权重
        weights = attn_learner.extract_weights(sequences)

        # 提取每只股票的回归/分类预测
        model_predictions = {}
        try:
            output = attn_learner.model.forward(sequences, training=False)
            pred_returns = output["regression"]
            pred_cls = output["classification"]

            exp_logits = np.exp(pred_cls - pred_cls.max(axis=1, keepdims=True))
            up_probs = exp_logits[:, 1] / exp_logits.sum(axis=1)

            for i in range(max_stocks):
                sym = symbols[i]
                model_predictions[sym] = {
                    "pred_return": float(pred_returns[i]),
                    "pred_up_prob": float(up_probs[i]),
                }
        except Exception as e:
            logger.warning(f"Model prediction extraction failed: {e}")

        return weights, model_predictions

    def _predict_states_from_model(
        self,
        attn_learner,
        cross_section: pd.DataFrame,
        daily_results: dict,
        model_predictions: Optional[dict[str, dict]],
    ) -> list[StateResult]:
        """
        用 Transformer 模型输出构建 StateResult (Phase 1: 分位数回归)。
        使用真实历史序列而非伪序列。

        用分位数预测计算预期收益和风险, 做截面排序。
        """
        if attn_learner.model is None:
            return []

        # 从历史缓存构建真实序列
        sequences, symbols = self._build_real_sequences(cross_section, seq_len=attn_learner.seq_len)

        # 截取模型需要的因子数
        n_model_factors = attn_learner.model.n_factors
        sequences = sequences[:, :, :n_model_factors]

        # 推理: 自动检测是否有截面编码器
        has_cs = hasattr(attn_learner.model, 'cross_sectional_transformer')
        try:
            if has_cs:
                output = attn_learner.model.forward_cross_sectional(sequences, training=False)
            else:
                output = attn_learner.model.forward(sequences, training=False)
            pred_returns = output["regression"]
            pred_cls = output["classification"]
            pred_quantiles = output["quantiles"]  # (n_stocks, 9)
            # Convert torch tensors to numpy
            if hasattr(pred_returns, "detach"):
                pred_returns = pred_returns.detach().cpu().numpy()
                pred_cls = pred_cls.detach().cpu().numpy()
                pred_quantiles = pred_quantiles.detach().cpu().numpy()
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")
            return []

        # 计算上涨概率（保留用于记录，但不用于状态判定）
        exp_logits = np.exp(pred_cls - pred_cls.max(axis=1, keepdims=True))
        up_probs = exp_logits[:, 1] / exp_logits.sum(axis=1)
        # 用分位数中位数 (q50) 判断方向，替代分类头
        # q50 占 40% loss 权重，比分类头 (15% CE) 可靠得多
        from src.strategy.adaptive_state_machine.attention_learner import QUANTILE_LEVELS
        _q_idx = {q: i for i, q in enumerate(QUANTILE_LEVELS)}
        q50 = pred_quantiles[:, _q_idx[0.5]]  # median prediction

        # 用分位数计算风险 (IQR)
        q_low = pred_quantiles[:, _q_idx[0.2]]   # 20th percentile
        q_high = pred_quantiles[:, _q_idx[0.8]]  # 80th percentile
        risk_iqr = q_high - q_low

        # 综合评分: 预期收益 / 风险 (Sharpe-like)
        sharpe_scores = np.where(risk_iqr > 1e-6, pred_returns / risk_iqr, 0.0)

        # 截面排名百分位
        ranks = np.argsort(sharpe_scores)
        n = len(ranks)
        percentiles = np.zeros(n)
        for i, rank_idx in enumerate(ranks):
            percentiles[rank_idx] = i / max(n - 1, 1)

        results = []
        for i, sym in enumerate(symbols):
            if sym not in daily_results:
                continue

            df_d, _ = daily_results[sym]
            last = df_d.iloc[-1]
            trade_date = str(last.get("trade_date", ""))

            pred_return = float(pred_returns[i])
            pred_up_prob = float(up_probs[i])
            q50_val = float(q50[i])
            pct = float(percentiles[i])

            # 用截面排名百分位 + 分位数中位数方向做状态判定
            # q50 占 40% loss 权重，比分类头 (15% CE) 可靠
            # 但 q50 可能整体偏负（熊市），用相对排名替代绝对 > 0 阈值
            q50_pctile = float(np.sum(q50 <= q50_val) / max(len(q50), 1))

            if pct > 0.9 and q50_pctile > 0.8 and pred_return > 0:
                state = StockState.BREAKOUT
                confidence = pct
            elif pct > 0.7 and q50_pctile > 0.5 and pred_return > 0:
                state = StockState.ACCUMULATION
                confidence = pct
            elif pct < 0.1 and q50_pctile < 0.2:
                state = StockState.COLLAPSE
                confidence = 1.0 - pct
            elif pct > 0.3:
                state = StockState.HOLD
                confidence = pct * 0.7
            else:
                state = StockState.IDLE
                confidence = 0.0

            aq = float(percentiles[i]) if state in (StockState.ACCUMULATION, StockState.HOLD) else 0.0
            bq = float(percentiles[i]) if state == StockState.BREAKOUT else 0.0

            results.append(StateResult(
                symbol=sym,
                trade_date=trade_date,
                state=state,
                confidence=round(min(1.0, max(0.0, confidence)), 4),
                aq_score=round(aq, 4),
                bq_score=round(bq, 4),
                composite_score=round(confidence, 4),
                pred_return=round(pred_return, 6),
                pred_up_prob=round(pred_up_prob, 4),
            ))

        state_counts = {}
        for r in results:
            state_counts[r.state] = state_counts.get(r.state, 0) + 1
        logger.info(
            f"Model prediction (q50 direction): {len(results)} stocks. "
            f"State distribution: {state_counts}"
        )
        return results

    def _save_signals(self, results: list, scan_date: str, output_dir: str):
        """保存信号到 CSV"""
        signal_rows = []
        for r in results:
            if r.state in (StockState.ACCUMULATION, StockState.BREAKOUT,
                           StockState.HOLD, StockState.COLLAPSE):
                signal_rows.append({
                    "symbol": r.symbol,
                    "trade_date": r.trade_date,
                    "state": r.state.value,
                    "confidence": r.confidence,
                    "aq_score": r.aq_score,
                    "bq_score": r.bq_score,
                    "composite_score": r.composite_score,
                    "details": json.dumps(r.details, ensure_ascii=False),
                })

        if signal_rows:
            signal_df = pd.DataFrame(signal_rows)
            output_path = os.path.join(output_dir, f"signals_{scan_date}.csv")
            signal_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(signal_df)} signals to {output_path}")

    # ── 历史回测 ─────────────────────────────────────────

    def backtest(
        self,
        start_date: str,
        end_date: str,
        interval_days: int = 5,
        config_path: str = "",
        output_dir: str = "",
    ) -> pd.DataFrame:
        """
        历史回测: 从 start_date 到 end_date, 每隔 interval_days 天执行一次扫描。

        Returns:
            所有日期的信号汇总 DataFrame
        """
        trade_dates = load_trade_cal(self.data_root)
        if not trade_dates:
            logger.error("No trade calendar found")
            return pd.DataFrame()

        trade_dates = [d for d in trade_dates if start_date <= d <= end_date]
        if not trade_dates:
            logger.error(f"No trading dates in range {start_date} to {end_date}")
            return pd.DataFrame()

        scan_dates = trade_dates[::interval_days]
        logger.info(f"Backtest: {len(scan_dates)} scan dates from {start_date} to {end_date}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 加载配置
        config = AdaptiveConfig.load(config_path) if config_path and os.path.exists(config_path) else AdaptiveConfig()

        # 持久化 evaluator (跨扫描日期复用, 保留 hold 状态)
        evaluator = StateEvaluator(config=config)

        all_signals = []
        summary_rows = []

        for i, scan_date in enumerate(scan_dates):
            logger.info(f"\n{'='*60}")
            logger.info(f"Backtest: Date {i+1}/{len(scan_dates)} — {scan_date}")

            results, summary, config = self.scan(
                scan_date=scan_date,
                config=config,
                evaluator=evaluator,
                output_dir=output_dir,
            )

            if summary.get("error"):
                continue

            # 收集信号
            for r in results:
                if r.state in (StockState.ACCUMULATION, StockState.BREAKOUT,
                               StockState.HOLD, StockState.COLLAPSE):
                    all_signals.append({
                        "scan_date": scan_date,
                        "symbol": r.symbol,
                        "state": r.state.value,
                        "confidence": r.confidence,
                        "aq_score": r.aq_score,
                        "bq_score": r.bq_score,
                        "composite_score": r.composite_score,
                    })

            summary_rows.append(summary)

        # 保存汇总
        if all_signals and output_dir:
            signal_df = pd.DataFrame(all_signals)
            output_path = os.path.join(output_dir, "backtest_signals.csv")
            signal_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(signal_df)} backtest signals to {output_path}")

        if summary_rows and output_dir:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(output_dir, "backtest_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Saved backtest summary to {summary_path}")

        return pd.DataFrame(all_signals) if all_signals else pd.DataFrame()


# ═════════════════════════════════════════════════════════
# 摘要打印
# ═════════════════════════════════════════════════════════

def _print_scan_summary(results: list, scan_date: str):
    """打印扫描摘要"""
    state_counts = {}
    for r in results:
        state_counts[r.state.value] = state_counts.get(r.state.value, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Adaptive State Machine Scan — {scan_date}")
    print(f"{'='*60}")
    print(f"  Total stocks evaluated: {len(results)}")
    for state in ["idle", "accumulation", "breakout", "hold", "collapse"]:
        count = state_counts.get(state, 0)
        if count > 0:
            print(f"  {state:>15}: {count}")
    print(f"{'='*60}")

    accumulation = [r for r in results if r.state == StockState.ACCUMULATION]
    breakout = [r for r in results if r.state == StockState.BREAKOUT]

    if accumulation:
        top_acc = sorted(accumulation, key=lambda x: x.confidence, reverse=True)[:5]
        print(f"\n  Top Accumulation Signals:")
        for r in top_acc:
            print(f"    {r.symbol:>10}  confidence={r.confidence:.3f}  aq={r.aq_score:.3f}")

    if breakout:
        top_bo = sorted(breakout, key=lambda x: x.confidence, reverse=True)[:5]
        print(f"\n  Top Breakout Signals:")
        for r in top_bo:
            print(f"    {r.symbol:>10}  confidence={r.confidence:.3f}  bq={r.bq_score:.3f}")
