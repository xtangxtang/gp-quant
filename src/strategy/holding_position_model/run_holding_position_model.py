#!/usr/bin/env python3
"""
Holding Position Model — CLI 入口

用法:
  # 训练
  python -m src.strategy.holding_position_model.run_holding_position_model \
      --train \
      --data_dir /path/to/tushare-daily-full \
      --data_root /path/to/gp-data \
      --output_model src/strategy/holding_position_model/models/holding_model.pt \
      --max_stocks 500

  # 单只持仓决策
  python -m src.strategy.holding_position_model.run_holding_position_model \
      --symbol sz000001 \
      --entry-price 15.20 \
      --entry-date 20260424 \
      --current-date 20260427 \
      --data_dir /path/to/tushare-daily-full \
      --data_root /path/to/gp-data \
      --model src/strategy/holding_position_model/models/holding_model.pt

  # 批量持仓扫描
  python -m src.strategy.holding_position_model.run_holding_position_model \
      --scan-positions positions.csv \
      --data_dir /path/to/tushare-daily-full \
      --data_root /path/to/gp-data \
      --model src/strategy/holding_position_model/models/holding_model.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = os.environ.get("GP_DATA_DIR", "/home/xtang/gp-workspace/gp-data")
DEFAULT_DAILY_DIR = os.path.join(DEFAULT_DATA_ROOT, "tushare-daily-full")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_existing_memmaps(data_cache: str) -> dict | None:
    """加载已有的 memmap 文件（正确大小，data_builder resize 后已截断文件）。"""
    required = [
        "X_factors_train", "X_holding_train", "y_stay_train",
        "y_collapse_train", "y_days_train", "sample_weights_train",
    ]
    paths = {}
    for name in required:
        p = os.path.join(data_cache, f"{name}.mmap")
        if not os.path.exists(p):
            return None
        paths[name] = p

    for name in ["X_factors_eval", "X_holding_eval", "y_stay_eval",
                  "y_collapse_eval", "y_days_eval", "sample_weights_eval"]:
        p = os.path.join(data_cache, f"{name}.mmap")
        if os.path.exists(p):
            paths[name] = p

    # 从日志获取实际样本数
    log_path = os.path.join(os.path.dirname(data_cache.rstrip("/")), "train.log")
    n_train = 0
    n_eval = 0
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                if "Actual written:" in line:
                    parts = line.split("Actual written:")[1].strip().split()
                    n_train = int(parts[0])
                    n_eval = int(parts[3])
                    break

    if n_train == 0:
        logger.warning("Could not determine sample count from log")
        return None

    # 推断 n_factors
    x_train_size = os.path.getsize(paths["X_factors_train"])
    n_factors = x_train_size // (n_train * 60 * 4)

    logger.info(f"Found existing memmap files: train={n_train:,}, eval={n_eval:,}, factors={n_factors}")

    data = {}
    shapes = {
        "X_factors_train": (n_train, 60, n_factors),
        "X_holding_train": (n_train, 5),
        "y_stay_train": (n_train,),
        "y_collapse_train": (n_train,),
        "y_days_train": (n_train,),
        "sample_weights_train": (n_train,),
    }
    if n_eval > 0:
        shapes["X_factors_eval"] = (n_eval, 60, n_factors)
        shapes["X_holding_eval"] = (n_eval, 5)
        shapes["y_stay_eval"] = (n_eval,)
        shapes["y_collapse_eval"] = (n_eval,)
        shapes["y_days_eval"] = (n_eval,)
        shapes["sample_weights_eval"] = (n_eval,)

    for name, shape in shapes.items():
        path = paths.get(name)
        if not path:
            continue
        data[name] = np.memmap(path, dtype=np.float32, mode="r", shape=shape)

    data["factor_names"] = []
    data["standardize_mean"] = np.array([])
    data["standardize_std"] = np.array([])
    data["memmap_dir"] = data_cache
    data["memmap_files"] = paths
    data["n_train"] = n_train
    data["n_eval"] = n_eval
    data["n_factors"] = n_factors

    logger.info("Loading data into RAM...")
    return data


def _load_to_ram(data: dict) -> dict:
    """将 memmap 数组加载到 RAM，消除磁盘 I/O 瓶颈。"""
    import gc
    ram_keys = [
        "X_factors_train", "X_holding_train", "y_stay_train",
        "y_collapse_train", "y_days_train", "sample_weights_train",
        "X_factors_eval", "X_holding_eval", "y_stay_eval",
        "y_collapse_eval", "y_days_eval", "sample_weights_eval",
    ]
    total_gb = 0.0
    for key in ram_keys:
        if key in data:
            old = data[key]
            if hasattr(old, "_mmap"):  # is memmap
                data[key] = np.array(old)  # load into RAM
                total_gb += data[key].nbytes / (1024 ** 3)
                logger.info(f"  Loaded {key} into RAM: {data[key].shape}, {data[key].nbytes / (1024**3):.1f} GB")
    logger.info(f"Total data in RAM: {total_gb:.1f} GB")
    gc.collect()
    return data


def cmd_train(args):
    """训练模式"""
    from .data_builder import build_training_data
    from .model import HoldingPositionModel
    from .train import HoldingTrainer
    from .config import (
        SEQ_LEN, D_MODEL, N_HEADS, N_LAYERS, HOLDING_DIM, HOLDING_HIDDEN,
        DROPOUT, WEIGHT_DECAY, N_EPOCHS, BATCH_SIZE, LEARNING_RATE,
        LOSS_STAY_WEIGHT, LOSS_COLLAPSE_WEIGHT, LOSS_DAYS_WEIGHT,
    )

    daily_dir = args.data_dir
    data_root = args.data_root
    output_model = args.output_model

    logger.info(f"Training mode: daily_dir={daily_dir}, max_stocks={args.max_stocks}")

    # 构建或加载训练数据
    data_cache = os.path.join(os.path.dirname(output_model), "data_cache")

    # 优先使用已有的 memmap 文件（跳过耗时的重建）
    data = _load_existing_memmaps(data_cache)
    if data is None:
        logger.info("No existing memmap found, building from scratch...")
        data = build_training_data(
            daily_dir=daily_dir,
            data_root=data_root,
            max_stocks=args.max_stocks,
            scan_date=args.scan_date or "",
            seq_len=args.seq_len or SEQ_LEN,
            max_hold_days=args.max_hold_days or 20,
            entry_sample_ratio=args.entry_sample_ratio or 0.3,
            train_ratio=args.train_ratio or 0.8,
            output_dir=data_cache,
        )

    # 加载到 RAM 消除 I/O 瓶颈
    logger.info("Loading data from memmap to RAM...")
    data = _load_to_ram(data)

    n_factors = len(data["factor_names"])
    seq_len = args.seq_len or SEQ_LEN

    # 创建模型
    model = HoldingPositionModel(
        n_factors=n_factors,
        seq_len=seq_len,
        d_model=args.d_model or D_MODEL,
        n_heads=args.n_heads or N_HEADS,
        n_layers=args.n_layers or N_LAYERS,
        holding_dim=HOLDING_DIM,
        holding_hidden=HOLDING_HIDDEN,
        dropout=args.dropout or DROPOUT,
        factor_names=data["factor_names"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created: {total_params:,} parameters")

    # 训练
    device = args.device or "cpu"
    model.to(device)
    trainer = HoldingTrainer(model, lr=args.lr or LEARNING_RATE, weight_decay=WEIGHT_DECAY, device=device)

    result = trainer.train(
        X_factors_train=data["X_factors_train"],
        X_holding_train=data["X_holding_train"],
        y_stay_train=data["y_stay_train"],
        y_collapse_train=data["y_collapse_train"],
        y_days_train=data["y_days_train"],
        w_train=data["sample_weights_train"],
        X_factors_eval=data["X_factors_eval"],
        X_holding_eval=data["X_holding_eval"],
        y_stay_eval=data["y_stay_eval"],
        y_collapse_eval=data["y_collapse_eval"],
        y_days_eval=data["y_days_eval"],
        n_epochs=args.epochs or N_EPOCHS,
        batch_size=args.batch_size or BATCH_SIZE,
        save_path=output_model,
        standardize_mean=data["standardize_mean"],
        standardize_std=data["standardize_std"],
        loss_stay_weight=LOSS_STAY_WEIGHT,
        loss_collapse_weight=LOSS_COLLAPSE_WEIGHT,
        loss_days_weight=LOSS_DAYS_WEIGHT,
    )

    logger.info(f"Training complete. Model saved to {output_model}")

    # 打印最后几个 epoch
    for h in result["history"][-3:]:
        eval_str = ""
        if "eval_loss" in h:
            eval_str = (
                f" eval_loss={h['eval_loss']:.4f} "
                f"stay_acc={h['stay_acc']:.3f} "
                f"collapse_acc={h['collapse_acc']:.3f} "
                f"days_mae={h['days_mae']:.1f}"
            )
        logger.info(f"  Epoch {h['epoch']}: loss={h['train_loss']:.4f}{eval_str}")


def cmd_predict(args):
    """单只持仓决策"""
    from .inference import HoldingInference

    inference = HoldingInference(
        model_path=args.model,
        daily_dir=args.data_dir,
        data_root=args.data_root,
    )

    if not inference.load():
        print("Failed to load model")
        return 1

    result = inference.predict(
        symbol=args.symbol,
        entry_price=args.entry_price,
        entry_date=args.entry_date,
        current_date=args.current_date or "",
    )

    # 打印结果
    print(f"\n{'='*60}")
    print(f"  Holding Position Decision — {result['symbol']}")
    print(f"{'='*60}")
    print(f"  Entry:       {result['entry_price']}")
    print(f"  Current:     {result['current_price']} ({result['unrealized_pnl']:+.1f}%)")
    print(f"  Peak:        {result['peak_price']} (drawdown {result['drawdown']:+.1f}%)")
    print(f"  Days held:   {result['days_held']}")
    print(f"{'='*60}")
    print(f"  stay_prob:     {result['stay_prob']:.3f}  {'→ 拿住' if result['stay_prob'] > 0.7 else ''}")
    print(f"  collapse_risk: {result['collapse_risk']:.3f}  {'→ 走人' if result['collapse_risk'] > 0.3 else ''}")
    print(f"  expected_days: {result['expected_days']:.1f}  {'→ 减仓' if result['expected_days'] < 3 else ''}")
    print(f"{'='*60}")
    print(f"  建议: {result['recommendation']}")
    print(f"{'='*60}\n")


def cmd_scan_positions(args):
    """批量持仓扫描"""
    import pandas as pd
    from .inference import HoldingInference

    positions_df = pd.read_csv(args.scan_positions)
    positions = positions_df.to_dict("records")

    inference = HoldingInference(
        model_path=args.model,
        daily_dir=args.data_dir,
        data_root=args.data_root,
    )

    if not inference.load():
        print("Failed to load model")
        return 1

    results = inference.predict_batch(positions)

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"  Position Scan — {len(results)} positions")
    print(f"{'='*60}")

    hold_count = 0
    sell_count = 0
    reduce_count = 0

    for r in results:
        if "error" in r:
            print(f"  {r['symbol']:>12}: ERROR — {r['error']}")
            continue

        rec = r["recommendation"]
        if rec == "持有":
            hold_count += 1
        elif rec == "走人":
            sell_count += 1
        elif rec == "减仓":
            reduce_count += 1

        print(
            f"  {r['symbol']:>12}: {r['unrealized_pnl']:+.1f}% | "
            f"stay={r['stay_prob']:.2f} risk={r['collapse_risk']:.2f} | "
            f"→ {rec}"
        )

    print(f"{'='*60}")
    print(f"  持有: {hold_count} | 减仓: {reduce_count} | 走人: {sell_count}")
    print(f"{'='*60}\n")

    # 输出结果
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        pd.DataFrame(results).to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Holding Position Model — 持仓决策模型",
    )

    # 模式
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="训练模式")
    group.add_argument("--symbol", type=str, help="单只持仓决策 (股票代码)")
    group.add_argument("--scan-positions", type=str, help="批量扫描 (CSV 路径)")

    # 数据路径
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DAILY_DIR, help="日线数据目录")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="数据根目录")
    parser.add_argument("--model", type=str, default="", help="模型路径")
    parser.add_argument("--output_model", type=str, default="", help="输出模型路径 (训练模式)")

    # 训练参数
    parser.add_argument("--max_stocks", type=int, default=500, help="最大股票数")
    parser.add_argument("--seq_len", type=int, default=60, help="序列长度")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--d_model", type=int, default=128, help="模型维度")
    parser.add_argument("--n_heads", type=int, default=8, help="Attention 头数")
    parser.add_argument("--n_layers", type=int, default=4, help="Transformer 层数")
    parser.add_argument("--device", default="cpu", help="设备 (cpu/cuda)")
    parser.add_argument("--scan_date", default="", help="限制数据到该日期")
    parser.add_argument("--max_hold_days", type=int, default=20, help="最大模拟持有天数")
    parser.add_argument("--entry_sample_ratio", type=float, default=0.3, help="Entry day 采样比例")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例 (时间 split)")

    # 预测参数
    parser.add_argument("--entry-price", type=float, default=0, help="入场价格")
    parser.add_argument("--entry-date", type=str, default="", help="入场日期 (YYYYMMDD)")
    parser.add_argument("--current-date", type=str, default="", help="当前日期 (YYYYMMDD)")
    parser.add_argument("--output", type=str, default="", help="输出文件路径 (批量扫描)")

    # 其他
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.train:
        if not args.output_model:
            args.output_model = os.path.join(DEFAULT_MODEL_DIR, "holding_model.pt")
        cmd_train(args)
    elif args.symbol:
        if not args.model:
            args.model = os.path.join(DEFAULT_MODEL_DIR, "holding_model.pt")
        if not args.entry_date:
            parser.error("--entry-date required for single position prediction")
        cmd_predict(args)
    elif args.scan_positions:
        if not args.model:
            args.model = os.path.join(DEFAULT_MODEL_DIR, "holding_model.pt")
        cmd_scan_positions(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
