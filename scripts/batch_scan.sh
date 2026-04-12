#!/bin/bash
# 并行扫描 A 股所有股票
# 使用批处理方式，每批处理一定数量的股票，避免内存溢出

DATA_DIR="/nvme5/xtang/gp-workspace/gp-data/trade"
BASIC_PATH="/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv"
OUT_DIR="/nvme5/xtang/gp-workspace/gp-quant/results/four_layer_system"
MAX_STOCKS=${1:-5498}
WORKERS=${2:-16}
BATCH_SIZE=$((MAX_STOCKS / WORKERS + 1))

echo "开始批量扫描..."
echo "  总股票数：$MAX_STOCKS"
echo "  并发数：$WORKERS"
echo "  每批数量：$BATCH_SIZE"

# 创建临时目录存储各批次的结果
TEMP_DIR=$(mktemp -d)
echo "  临时目录：$TEMP_DIR"

# 启动并行任务
pids=()
for ((i=0; i<WORKERS; i++)); do
    offset=$((i * BATCH_SIZE))
    batch_out="$TEMP_DIR/batch_${i}.csv"

    echo "启动批次 $i: offset=$offset, limit=$BATCH_SIZE"

    python3 -m src.strategy.four_layer_entropy_system.run_scan \
        --data_dir "$DATA_DIR" \
        --basic_path "$BASIC_PATH" \
        --out_dir "$TEMP_DIR/batch_${i}" \
        --max_stocks "$BATCH_SIZE" \
        --workers 1 \
        2>&1 | tee "$TEMP_DIR/batch_${i}.log" &

    pids+=($!)
done

# 等待所有任务完成
echo "等待所有批次完成..."
for pid in "${pids[@]}"; do
    wait $pid
done

echo "所有批次完成，汇总结果..."

# 汇总所有批次的 stock_decisions.csv
header=$(head -1 "$TEMP_DIR/batch_0/stock_decisions.csv")
echo "$header" > "$OUT_DIR/stock_decisions.csv"

for ((i=0; i<WORKERS; i++)); do
    batch_file="$TEMP_DIR/batch_${i}/stock_decisions.csv"
    if [ -f "$batch_file" ]; then
        tail -n +2 "$batch_file" >> "$OUT_DIR/stock_decisions.csv"
    fi
done

# 汇总买入信号
echo "汇总买入信号..."
cat "$TEMP_DIR"/batch_*/buy_signals.csv 2>/dev/null | head -1 > "$OUT_DIR/buy_signals.csv"
for ((i=0; i<WORKERS; i++)); do
    batch_file="$TEMP_DIR/batch_${i}/buy_signals.csv"
    if [ -f "$batch_file" ]; then
        tail -n +2 "$batch_file" >> "$OUT_DIR/buy_signals.csv"
    fi
done

# 清理临时目录
rm -rf "$TEMP_DIR"

echo "完成！结果保存在 $OUT_DIR"
echo "买入信号："
wc -l "$OUT_DIR/buy_signals.csv"
