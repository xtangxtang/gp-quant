"""
四层熵交易系统 - 扫描运行脚本

使用方法：
    python src/strategy/four_layer_entropy_system/run_scan.py \\
        --data_dir /nvme5/xtang/gp-workspace/gp-data/trade \\
        --basic_path /nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv \\
        --out_dir results/four_layer_system \\
        --max_stocks 50
"""

import argparse
import os
import json
import re
import pandas as pd
from datetime import datetime

from .core_system import FourLayerEntropySystem, SystemOutput
from .config import Config


def parse_args():
    parser = argparse.ArgumentParser(description='四层熵交易系统扫描')

    parser.add_argument(
        '--data_dir',
        type=str,
        default='/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full',
        help='日线数据目录（tushare-daily-full）'
    )
    parser.add_argument(
        '--basic_path',
        type=str,
        default='/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv',
        help='股票基本信息 CSV 路径'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='/nvme5/xtang/gp-workspace/gp-quant/results/four_layer_system',
        help='输出目录'
    )
    parser.add_argument(
        '--scan_date',
        type=str,
        default=None,
        help='扫描日期（YYYY-MM-DD）'
    )
    parser.add_argument(
        '--max_stocks',
        type=int,
        default=50,
        help='最大股票数量'
    )
    parser.add_argument(
        '--initial_capital',
        type=float,
        default=1_000_000.0,
        help='初始资金'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=-1,
        help='并发工作进程数（-1 表示使用 CPU 核心数，1 表示串行）'
    )
    parser.add_argument(
        '--no_parallel',
        action='store_true',
        help='禁用并行计算（串行模式）'
    )

    return parser.parse_args()


def save_results(output: SystemOutput, out_dir: str):
    """保存扫描结果"""
    os.makedirs(out_dir, exist_ok=True)

    # 1. 保存市场门控结果
    market_gate_df = pd.DataFrame([output.market_gate.to_dict()])
    market_gate_df.to_csv(os.path.join(out_dir, 'market_gate.csv'), index=False)

    # 2. 保存个股决策结果
    decisions_data = []
    for d in output.decisions:
        decisions_data.append(d.to_dict())

    decisions_df = pd.DataFrame(decisions_data)
    decisions_df.to_csv(os.path.join(out_dir, 'stock_decisions.csv'), index=False)

    # 3. 保存汇总统计
    summary = {
        'scan_date': output.scan_date,
        'market_state': output.market_gate.state,
        'gate_score': output.market_gate.gate_score,
        'total_buy': output.total_buy,
        'total_sell': output.total_sell,
        'total_hold': output.total_hold,
        'total_wait': output.total_wait,
        'recommended_total_position': output.recommended_total_position,
        'coupling_entropy': output.market_gate.coupling_entropy,
        'noise_cost': output.market_gate.noise_cost,
    }

    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 4. 保存买入信号列表
    buy_signals = decisions_df[decisions_df['action'] == 'buy']

    if len(buy_signals) > 0:
        # 用正则表达式提取嵌套的 stock_state 字段
        def extract_current_state(x):
            match = re.search(r"'current_state':\s*'(\w+)'", str(x))
            return match.group(1) if match else ''

        def extract_total_score(x):
            match = re.search(r"'total_score':\s*np\.float64\(([\d\.\-]+)\)", str(x))
            return float(match.group(1)) if match else 0.0

        buy_signals_df = buy_signals.copy()
        buy_signals_df['current_state'] = buy_signals_df['stock_state'].apply(extract_current_state)
        buy_signals_df['total_score'] = buy_signals_df['stock_state'].apply(extract_total_score)
        buy_signals_df['market_state'] = output.market_gate.state
        buy_signals_df['gate_score'] = output.market_gate.gate_score

        buy_signals_df = buy_signals_df[
            ['stock_code', 'stock_name', 'confidence', 'position_size',
             'market_state', 'gate_score', 'current_state', 'total_score']
        ].sort_values('confidence', ascending=False)
        buy_signals_df.to_csv(os.path.join(out_dir, 'buy_signals.csv'), index=False)
        print(f"  - buy_signals.csv: 买入信号 ({len(buy_signals_df)} 只)")
    else:
        # 无买入信号时创建空文件
        pd.DataFrame(columns=['stock_code', 'stock_name', 'confidence', 'position_size',
                              'market_state', 'gate_score', 'current_state', 'total_score']
        ).to_csv(os.path.join(out_dir, 'buy_signals.csv'), index=False)
        print(f"  - buy_signals.csv: 无买入信号")


def print_summary(output: SystemOutput):
    """打印摘要信息"""
    print("\n" + "=" * 80)
    print("四层熵交易系统扫描结果")
    print("=" * 80)

    print(f"\n扫描日期：{output.scan_date}")

    print(f"\n【市场门控层】")
    print(f"  市场状态：{output.market_gate.state}")
    print(f"  门控得分：{output.market_gate.gate_score:.2f}")
    print(f"  耦合熵：{output.market_gate.coupling_entropy:.3f}")
    print(f"  噪声成本：{output.market_gate.noise_cost:.3f}")
    print(f"  战略放弃：{output.market_gate.abandonment_flag}")

    print(f"\n【个股决策统计】")
    print(f"  买入：{output.total_buy} 只")
    print(f"  卖出：{output.total_sell} 只")
    print(f"  持有：{output.total_hold} 只")
    print(f"  观望：{output.total_wait} 只")
    print(f"  建议总仓位：{output.recommended_total_position:,.0f} 元")

    # 显示 top 买入信号
    buy_decisions = [d for d in output.decisions if d.action == 'buy']
    if buy_decisions:
        print(f"\n【Top 买入信号】")
        buy_decisions.sort(key=lambda x: x.confidence, reverse=True)
        for i, d in enumerate(buy_decisions[:10]):
            print(f"  {i+1}. {d.stock_code}: 置信度={d.confidence:.2f}, "
                  f"仓位={d.position_size:,.0f}元, "
                  f"状态={d.stock_state.current_state}")

    print("\n" + "=" * 80)


def main():
    args = parse_args()

    # 创建配置
    config = Config()
    config.data_dir = args.data_dir
    config.output_dir = args.out_dir
    config.max_stocks = args.max_stocks
    config.execution.initial_capital = args.initial_capital
    config.num_workers = args.workers
    config.use_parallel = not args.no_parallel

    # 创建系统
    system = FourLayerEntropySystem(config)

    # 执行扫描
    print("开始扫描...")
    output = system.scan(
        data_dir=args.data_dir,
        basic_path=args.basic_path,
        scan_date=args.scan_date,
        max_stocks=args.max_stocks,
    )

    # 打印摘要
    print_summary(output)

    # 保存结果
    save_results(output, args.out_dir)


if __name__ == "__main__":
    main()
