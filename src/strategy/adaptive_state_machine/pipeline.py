"""
Adaptive State Machine — Pipeline 编排

向后兼容的 run_scan / run_backtest 函数, 内部委托给 AdaptiveStateMachine。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

from .config import AdaptiveConfig
from .strategy import AdaptiveStateMachine

logger = logging.getLogger(__name__)


def run_scan(
    daily_dir: str,
    data_root: str,
    scan_date: str,
    cache_dir: str = "",
    config_dir: str = "",
    output_dir: str = "",
    attention_model_path: str = "",
    attention_alpha: float = 1.0,
    cls_mode: str = "rules",
) -> dict:
    """执行单次扫描 (向后兼容接口)。"""
    config_path = os.path.join(config_dir, "adaptive_config.json") if config_dir else ""

    config = AdaptiveConfig.load(config_path) if config_path and os.path.exists(config_path) else None
    if config is None:
        config = AdaptiveConfig()
        config.last_updated = scan_date

    strategy = AdaptiveStateMachine(
        daily_dir=daily_dir,
        data_root=data_root,
        cache_dir=cache_dir,
        attention_model_path=attention_model_path,
        attention_alpha=attention_alpha,
        cls_mode=cls_mode,
    )

    _, summary, config = strategy.scan(
        scan_date=scan_date,
        config=config,
        config_path=config_path,
        output_dir=output_dir,
    )

    return summary


def run_backtest(
    daily_dir: str,
    data_root: str,
    start_date: str,
    end_date: str,
    interval_days: int = 5,
    cache_dir: str = "",
    config_dir: str = "",
    output_dir: str = "",
    attention_model_path: str = "",
    attention_alpha: float = 1.0,
    cls_mode: str = "rules",
) -> pd.DataFrame:
    """历史回测 (向后兼容接口)。"""
    config_path = os.path.join(config_dir, "adaptive_config.json") if config_dir else ""

    strategy = AdaptiveStateMachine(
        daily_dir=daily_dir,
        data_root=data_root,
        cache_dir=cache_dir,
        attention_model_path=attention_model_path,
        attention_alpha=attention_alpha,
        cls_mode=cls_mode,
    )

    return strategy.backtest(
        start_date=start_date,
        end_date=end_date,
        interval_days=interval_days,
        config_path=config_path,
        output_dir=output_dir,
    )
