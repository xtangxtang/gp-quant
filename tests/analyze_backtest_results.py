"""
熵因子回测结果分析脚本

生成详细的分析报告和可视化图表
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BacktestAnalyzer:
    """回测结果分析器"""

    def __init__(self, results_dir: str = "/nvme5/xtang/gp-workspace/gp-quant/results/entropy_backtest"):
        self.results_dir = results_dir
        self.scheme_names = {
            'A': '固定持有 30 分钟',
            'B': '因子驱动退出',
            'C': '止盈止损',
            'D': '组合方案'
        }

    def load_all_results(self) -> dict:
        """加载所有回测结果"""
        results = {}

        for scheme in ['A', 'B', 'C', 'D']:
            trades_file = os.path.join(self.results_dir, f"trades_scheme_{scheme}.csv")
            equity_file = os.path.join(self.results_dir, f"equity_scheme_{scheme}.csv")

            scheme_data = {}

            if os.path.exists(trades_file):
                scheme_data['trades'] = pd.read_csv(trades_file)
                scheme_data['trades']['exit_time'] = pd.to_datetime(scheme_data['trades']['exit_time'])
            else:
                scheme_data['trades'] = pd.DataFrame()

            if os.path.exists(equity_file):
                scheme_data['equity'] = pd.read_csv(equity_file)
                scheme_data['equity']['trade_time'] = pd.to_datetime(scheme_data['equity']['trade_time'])
            else:
                scheme_data['equity'] = pd.DataFrame()

            results[scheme] = scheme_data

        return results

    def calculate_ic(self, data_dir: str, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 IC (Information Coefficient)

        IC = 因子值与未来收益的 Rank Correlation
        """
        # 这里简化处理，实际需要从原始数据计算
        # 假设 factors_df 已经包含因子和后续收益
        pass

    def plot_equity_curves(self, results: dict, save_path: str = None):
        """绘制 4 种方案的权益曲线对比"""
        fig, ax = plt.subplots(figsize=(14, 7))

        colors = {'A': 'blue', 'B': 'green', 'C': 'red', 'D': 'purple'}

        for scheme, data in results.items():
            if data['equity'].empty:
                continue

            equity = data['equity']
            # 归一化到 1 开始
            if len(equity) > 0 and equity['total_value'].iloc[0] > 0:
                normalized = equity['total_value'] / equity['total_value'].iloc[0]
            else:
                normalized = equity['total_value']

            ax.plot(equity.index, normalized,
                    label=self.scheme_names.get(scheme, f"方案{scheme}"),
                    color=colors.get(scheme, 'gray'),
                    linewidth=1.5)

        ax.set_xlabel('时间')
        ax.set_ylabel('累计收益 (归一化)')
        ax.set_title('4 种卖出方案权益曲线对比')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"权益曲线图已保存：{save_path}")

        plt.close()

    def plot_returns_comparison(self, results: dict, save_path: str = None):
        """绘制收益率对比柱状图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = [
            ('total_return', '总收益率 (%)', lambda x: x * 100),
            ('win_rate', '胜率 (%)', lambda x: x * 100),
            ('max_drawdown', '最大回撤 (%)', lambda x: abs(x) * 100)
        ]

        schemes = ['A', 'B', 'C', 'D']
        x_pos = np.arange(len(schemes))
        width = 0.2

        for idx, (metric, label, transform) in enumerate(metrics):
            values = []
            for scheme in schemes:
                stats = self._calculate_stats(results[scheme])
                values.append(transform(stats.get(metric, 0)))

            axes[idx].bar(x_pos, values, width, color=['blue', 'green', 'red', 'purple'])
            axes[idx].set_xticks(x_pos)
            axes[idx].set_xticklabels([self.scheme_names.get(s, s) for s in schemes], rotation=15)
            axes[idx].set_ylabel(label)
            axes[idx].set_title(label)
            axes[idx].grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.1, f'{v:.1f}%', ha='center', fontsize=9)

        plt.suptitle('4 种方案核心指标对比', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"收益对比图已保存：{save_path}")

        plt.close()

    def plot_exit_reason_distribution(self, results: dict, save_path: str = None):
        """绘制退出原因分布"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        colors = plt.cm.Set3(np.linspace(0, 1, 20))

        for idx, scheme in enumerate(['A', 'B', 'C', 'D']):
            ax = axes[idx]
            trades = results[scheme]['trades']

            if trades.empty or 'exit_reason' not in trades.columns:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"方案{scheme}: {self.scheme_names.get(scheme, '')}")
                continue

            # 统计退出原因
            exit_reasons = trades['exit_reason'].value_counts()

            # 绘制饼图
            wedges, texts, autotexts = ax.pie(
                exit_reasons.values,
                labels=exit_reasons.index,
                autopct='%1.1f%%',
                colors=colors[:len(exit_reasons)],
                startangle=90
            )

            # 设置字体大小
            for text in texts:
                text.set_fontsize(8)
            for autotext in autotexts:
                autotext.set_fontsize(8)

            ax.set_title(f"方案{scheme}: {self.scheme_names.get(scheme, '')}\n(总交易：{len(trades)})")

        plt.suptitle('退出原因分布对比', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"退出原因图已保存：{save_path}")

        plt.close()

    def plot_trade_distribution(self, results: dict, save_path: str = None):
        """绘制交易盈亏分布"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, scheme in enumerate(['A', 'B', 'C', 'D']):
            ax = axes[idx]
            trades = results[scheme]['trades']

            if trades.empty or 'returns' not in trades.columns:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"方案{scheme}")
                continue

            # 绘制盈亏分布直方图
            returns = trades['returns'] * 100  # 转为百分比

            ax.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(x=returns.mean(), color='red', linestyle='--',
                       label=f'均值：{returns.mean():.2f}%')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

            ax.set_xlabel('收益率 (%)')
            ax.set_ylabel('频数')
            ax.set_title(f"方案{scheme}: {self.scheme_names.get(scheme, '')}\n"
                        f"交易数：{len(trades)}, 胜率：{(returns > 0).mean()*100:.1f}%")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('交易盈亏分布对比', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"盈亏分布图已保存：{save_path}")

        plt.close()

    def _calculate_stats(self, scheme_data: dict) -> dict:
        """计算统计数据"""
        trades = scheme_data.get('trades', pd.DataFrame())
        equity = scheme_data.get('equity', pd.DataFrame())

        stats = {}

        if not trades.empty:
            stats['total_trades'] = len(trades)
            stats['winning_trades'] = len(trades[trades['pnl'] > 0])
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
            stats['total_return'] = (1 + trades['returns']).prod() - 1
            stats['avg_return'] = trades['returns'].mean()
            stats['max_drawdown'] = 0  # 需要从权益曲线计算

        if not equity.empty and len(equity) > 0:
            peak = equity['total_value'].cummax()
            drawdown = (equity['total_value'] - peak) / peak
            stats['max_drawdown'] = drawdown.min()
            stats['final_value'] = equity['total_value'].iloc[-1]

        return stats

    def generate_report(self, results: dict, save_path: str = None):
        """生成文本报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("熵因子分钟数据回测报告")
        report_lines.append("=" * 80)
        report_lines.append("")

        # 方案对比
        report_lines.append("一、4 种方案核心指标对比")
        report_lines.append("-" * 80)
        report_lines.append(f"{'方案':<15} {'交易次数':<12} {'胜率':<10} {'总收益':<12} {'最大回撤':<12}")
        report_lines.append("-" * 80)

        for scheme in ['A', 'B', 'C', 'D']:
            stats = self._calculate_stats(results[scheme])
            trades = stats.get('total_trades', 0)
            win_rate = f"{stats.get('win_rate', 0)*100:.1f}%"
            total_return = f"{stats.get('total_return', 0)*100:.1f}%"
            max_dd = f"{abs(stats.get('max_drawdown', 0))*100:.1f}%"

            report_lines.append(
                f"{self.scheme_names.get(scheme, scheme):<15} "
                f"{trades:<12} {win_rate:<10} {total_return:<12} {max_dd:<12}"
            )

        report_lines.append("")

        # 各方案详细统计
        for scheme in ['A', 'B', 'C', 'D']:
            trades = results[scheme]['trades']
            if trades.empty:
                continue

            report_lines.append(f"\n二、方案{scheme} ({self.scheme_names.get(scheme, '')}) 详细统计")
            report_lines.append("-" * 80)

            # 基础统计
            report_lines.append(f"总交易次数：{len(trades)}")
            report_lines.append(f"盈利交易：{len(trades[trades['pnl'] > 0])}")
            report_lines.append(f"亏损交易：{len(trades[trades['pnl'] <= 0])}")

            # 盈亏统计
            report_lines.append(f"平均盈亏：{trades['pnl'].mean():.2f}元")
            report_lines.append(f"最大单笔盈利：{trades['pnl'].max():.2f}元")
            report_lines.append(f"最大单笔亏损：{trades['pnl'].min():.2f}元")

            # 收益率统计
            report_lines.append(f"平均收益率：{trades['returns'].mean()*100:.2f}%")
            report_lines.append(f"收益率标准差：{trades['returns'].std()*100:.2f}%")

            # 退出原因统计
            if 'exit_reason' in trades.columns:
                report_lines.append("\n退出原因分布:")
                exit_counts = trades['exit_reason'].value_counts()
                for reason, count in exit_counts.items():
                    pct = count / len(trades) * 100
                    report_lines.append(f"  {reason}: {count} ({pct:.1f}%)")

            report_lines.append("")

        # 结论
        report_lines.append("\n三、结论")
        report_lines.append("-" * 80)

        # 找出最优方案
        best_scheme = None
        best_return = -float('inf')

        for scheme in ['A', 'B', 'C', 'D']:
            stats = self._calculate_stats(results[scheme])
            total_return = stats.get('total_return', 0)
            if total_return > best_return:
                best_return = total_return
                best_scheme = scheme

        if best_scheme:
            report_lines.append(f"最优方案：{self.scheme_names.get(best_scheme, best_scheme)}")
            report_lines.append(f"总收益率：{best_return*100:.1f}%")

        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("报告结束")
        report_lines.append("=" * 80)

        # 保存报告
        report_text = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"文本报告已保存：{save_path}")

        return report_text


def main():
    """主函数"""
    analyzer = BacktestAnalyzer()

    # 加载结果
    print("加载回测结果...")
    results = analyzer.load_all_results()

    # 检查是否有数据
    has_data = any(not r['trades'].empty for r in results.values())
    if not has_data:
        print("未找到回测结果，请先运行 backtest_entropy_factors.py")
        return

    # 创建输出目录
    output_dir = analyzer.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # 生成图表
    print("生成图表...")

    analyzer.plot_equity_curves(
        results,
        save_path=os.path.join(output_dir, "equity_comparison.png")
    )

    analyzer.plot_returns_comparison(
        results,
        save_path=os.path.join(output_dir, "returns_comparison.png")
    )

    analyzer.plot_exit_reason_distribution(
        results,
        save_path=os.path.join(output_dir, "exit_reason_distribution.png")
    )

    analyzer.plot_trade_distribution(
        results,
        save_path=os.path.join(output_dir, "trade_distribution.png")
    )

    # 生成文本报告
    print("生成文本报告...")
    report = analyzer.generate_report(
        results,
        save_path=os.path.join(output_dir, "backtest_report.txt")
    )

    print("\n" + report)

    print("\n分析完成!")


if __name__ == "__main__":
    main()
