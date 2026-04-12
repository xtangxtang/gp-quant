#!/usr/bin/env python3
"""
诊断脚本: 分析当前策略对中际旭创/天孚通信/新易盛的信号输出。

目标: 
1. 计算 2025-04 ~ 2025-06 每个交易日的日线熵指标
2. 检测分岔预警信号
3. 找到策略命中/遗漏的时间点和原因
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from src.strategy.dual_entropy_accumulation.config import (
    DailyEntropyConfig, IntradayEntropyConfig, FusionSignalConfig, SellSignalConfig,
)
from src.strategy.dual_entropy_accumulation.daily_entropy import DailyEntropy
from src.strategy.dual_entropy_accumulation.intraday_entropy import IntradayEntropyAnalyzer
from src.strategy.dual_entropy_accumulation.fusion_signal import FusionSignal
from src.strategy.dual_entropy_accumulation.bifurcation import (
    BifurcationDetector, BifurcationConfig,
)


DAILY_DIR = '/nvme5/xtang/gp-workspace/gp-data/tushare-daily-full'
MINUTE_DIR = '/nvme5/xtang/gp-workspace/gp-data/trade'

TARGETS = [
    ('sz300308', '中际旭创'),
    ('sz300394', '天孚通信'),
    ('sz300502', '新易盛'),
]

# 分析 2025-04-01 ~ 2025-06-30
DATE_START = '20250401'
DATE_END = '20250630'


def load_daily(code):
    path = os.path.join(DAILY_DIR, f'{code}.csv')
    df = pd.read_csv(path)
    df['trade_date'] = pd.to_numeric(df['trade_date'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df = df.dropna(subset=['trade_date', 'close', 'vol', 'open'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    return df


def diagnose_daily_entropy(code, name, df):
    """逐日计算日线熵指标，输出时序表。"""
    calculator = DailyEntropy(DailyEntropyConfig())
    
    start_int = int(DATE_START)
    end_int = int(DATE_END)
    
    mask = (df['trade_date'] >= start_int) & (df['trade_date'] <= end_int)
    target_dates = df.loc[mask, 'trade_date'].astype(int).tolist()
    
    rows = []
    for d in target_dates:
        idx = df.index[df['trade_date'] == d][0] + 1  # cutoff = up to and including this row
        if idx < 70:
            continue
        
        start = max(0, idx - 250)
        close_sub = df['close'].values[start:idx]
        vol_sub = df['vol'].values[start:idx]
        
        if len(close_sub) < 70:
            continue
        
        prices = pd.Series(close_sub, index=np.arange(len(close_sub)))
        volumes = pd.Series(vol_sub, index=np.arange(len(vol_sub)))
        
        result = calculator.compute(prices, volumes)
        if result is None:
            continue
        
        # 计算价格结构特征（分岔预筛需要）
        returns = np.diff(np.log(close_sub[-25:]))
        mom20 = float(np.sum(returns[-20:])) if len(returns) >= 20 else 0
        if len(returns) >= 10:
            ret5_recent = np.sum(returns[-5:])
            ret5_prev = np.sum(returns[-10:-5])
            price_accel = ret5_recent - ret5_prev
        else:
            price_accel = 0
        avg_vol_20 = np.mean(vol_sub[-20:])
        liquidity = close_sub[-1] * avg_vol_20
        
        d_str = f'{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}'
        rows.append({
            'date': d_str,
            'close': close_sub[-1],
            'PE20': result.perm_entropy_20,
            'PE60': result.perm_entropy_60,
            'gap': result.entropy_gap,
            'EP_pctile': result.entropy_percentile,
            'PI': result.path_irreversibility,
            'DE': result.dominant_eigenvalue,
            'var_lift': result.var_lift,
            'compressed': result.is_compressed,
            'mom20': mom20,
            'price_accel': price_accel,
            'liquidity': liquidity / 1e6,  # 百万
        })
    
    return pd.DataFrame(rows)


def diagnose_bifurcation(code, name, df):
    """逐日运行分岔预警检测。"""
    calculator = DailyEntropy(DailyEntropyConfig())
    detector = BifurcationDetector()
    
    start_int = int(DATE_START)
    end_int = int(DATE_END)
    
    mask = (df['trade_date'] >= start_int) & (df['trade_date'] <= end_int)
    target_dates = df.loc[mask, 'trade_date'].astype(int).tolist()
    
    results = []
    for d in target_dates:
        idx = df.index[df['trade_date'] == d][0] + 1
        if idx < 80:
            continue
        
        close_arr = df['close'].values[:idx]
        vol_arr = df['vol'].values[:idx]
        
        # 价格结构
        returns = np.diff(np.log(close_arr[-25:]))
        if len(returns) < 20:
            continue
        ret20 = returns[-20:]
        mom20 = float(np.sum(ret20))
        
        if len(ret20) >= 10:
            ret5_recent = np.sum(ret20[-5:])
            ret5_prev = np.sum(ret20[-10:-5])
            price_accel = ret5_recent - ret5_prev
        else:
            price_accel = 0
        
        avg_vol_20 = np.mean(vol_arr[-20:])
        liquidity = close_arr[-1] * avg_vol_20
        
        # 日线熵
        start = max(0, idx - 250)
        close_sub = close_arr[start:]
        vol_sub = vol_arr[start:]
        if len(close_sub) < 70:
            continue
        
        prices = pd.Series(close_sub, index=np.arange(len(close_sub)))
        volumes = pd.Series(vol_sub, index=np.arange(len(vol_sub)))
        result = calculator.compute(prices, volumes)
        if result is None:
            continue
        
        # 5天前的日线熵
        idx5 = max(70, idx - 5)
        start5 = max(0, idx5 - 250)
        close_sub5 = df['close'].values[start5:idx5]
        vol_sub5 = df['vol'].values[start5:idx5]
        result5 = result
        if len(close_sub5) >= 70:
            p5 = pd.Series(close_sub5, index=np.arange(len(close_sub5)))
            v5 = pd.Series(vol_sub5, index=np.arange(len(vol_sub5)))
            r5 = calculator.compute(p5, v5)
            if r5 is not None:
                result5 = r5
        
        # 预筛检查
        prescreen_pass = detector.passes_prescreen(
            dominant_eigenvalue=result.dominant_eigenvalue,
            price_accel=price_accel,
            liquidity=liquidity,
            mom20=mom20,
        )
        
        d_str = f'{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}'
        
        # 不管预筛是否通过都计算得分
        br = detector.evaluate(
            stock_code=code,
            trade_date=d_str,
            perm_entropy_20=result.perm_entropy_20,
            perm_entropy_60=result.perm_entropy_60,
            entropy_gap=result.entropy_gap,
            entropy_percentile=result.entropy_percentile,
            path_irreversibility=result.path_irreversibility,
            dominant_eigenvalue=result.dominant_eigenvalue,
            var_lift=result.var_lift,
            price_accel=price_accel,
            mom20=mom20,
            liquidity=liquidity,
            dominant_eigenvalue_prev5=result5.dominant_eigenvalue,
            path_irrev_prev5=result5.path_irreversibility,
        )
        
        results.append({
            'date': d_str,
            'close': close_arr[-1],
            'prescreen': prescreen_pass,
            'signal': br.signal,
            'total_score': br.total_score,
            'eig_score': br.eigenvalue_score,
            'ent_div_score': br.entropy_divergence_score,
            'dir_score': br.directionality_score,
            'mom_score': br.variance_state_score,
            '|DE|': abs(result.dominant_eigenvalue),
            'price_accel': price_accel,
            'liquidity_M': liquidity / 1e6,
            'mom20': mom20,
            # Gate values for debugging
            'gate_eig': abs(result.dominant_eigenvalue) >= 0.65,
            'gate_accel': price_accel <= -0.10,
            'gate_liq': liquidity >= 3_000_000,
            'gate_mom': mom20 >= -0.05,
        })
    
    return pd.DataFrame(results)


def main():
    for code, name in TARGETS:
        print(f'\n{"=" * 100}')
        print(f'  {name} ({code})')
        print(f'{"=" * 100}')
        
        df = load_daily(code)
        
        # 1. 日线熵诊断
        entropy_df = diagnose_daily_entropy(code, name, df)
        print(f'\n--- 日线熵指标 (compressed=True的日期标★) ---')
        print(f'{"日期":<12s} {"收盘":>7s} {"PE20":>6s} {"PE60":>6s} {"差":>6s} '
              f'{"百分位":>6s} {"PI":>7s} {"|DE|":>6s} {"varL":>5s} {"压缩":>4s}')
        for _, r in entropy_df.iterrows():
            flag = '★' if r['compressed'] else ' '
            print(f'{r["date"]:<12s} {r["close"]:>7.2f} {r["PE20"]:>6.3f} {r["PE60"]:>6.3f} '
                  f'{r["gap"]:>6.3f} {r["EP_pctile"]:>6.3f} {r["PI"]:>7.4f} '
                  f'{abs(r["DE"]):>6.3f} {r["var_lift"]:>5.2f} {flag:>4s}')
        
        # 2. 分岔预警诊断
        bif_df = diagnose_bifurcation(code, name, df)
        print(f'\n--- 分岔预警诊断 ---')
        print(f'{"日期":<12s} {"收盘":>7s} {"预筛":>4s} {"信号":>6s} {"总分":>6s} '
              f'{"特征值":>6s} {"熵差":>6s} {"方向":>6s} {"动量":>6s} '
              f'{"|DE|":>6s} {"加速度":>7s} {"流动_M":>7s} {"mom20":>6s} '
              f'{"gEig":>4s} {"gAcc":>4s} {"gLiq":>4s} {"gMom":>4s}')
        
        for _, r in bif_df.iterrows():
            ps = '✓' if r['prescreen'] else '✗'
            sig = r['signal']
            g_marks = (
                f'{"✓" if r["gate_eig"] else "✗":>4s} '
                f'{"✓" if r["gate_accel"] else "✗":>4s} '
                f'{"✓" if r["gate_liq"] else "✗":>4s} '
                f'{"✓" if r["gate_mom"] else "✗":>4s}'
            )
            print(f'{r["date"]:<12s} {r["close"]:>7.2f} {ps:>4s} {sig:>6s} '
                  f'{r["total_score"]:>6.3f} {r["eig_score"]:>6.3f} '
                  f'{r["ent_div_score"]:>6.3f} {r["dir_score"]:>6.3f} '
                  f'{r["mom_score"]:>6.3f} {r["|DE|"]:>6.3f} '
                  f'{r["price_accel"]:>7.4f} {r["liquidity_M"]:>7.1f} '
                  f'{r["mom20"]:>6.3f} {g_marks}')
        
        # 统计
        if len(bif_df) > 0:
            buy_dates = bif_df[bif_df['signal'] == 'buy']
            watch_dates = bif_df[bif_df['signal'] == 'watch']
            prescreen_pass = bif_df[bif_df['prescreen'] == True]
            print(f'\n  统计: {len(bif_df)}天中, '
                  f'预筛通过={len(prescreen_pass)}, '
                  f'buy={len(buy_dates)}, '
                  f'watch={len(watch_dates)}')
            
            if len(prescreen_pass) > 0:
                print(f'  预筛通过日期: {", ".join(prescreen_pass["date"].tolist())}')
            
            # 找到得分最高的5天
            top5 = bif_df.nlargest(5, 'total_score')
            print(f'\n  得分最高的5天:')
            for _, r in top5.iterrows():
                print(f'    {r["date"]}: score={r["total_score"]:.3f} '
                      f'signal={r["signal"]} prescreen={r["prescreen"]} '
                      f'|DE|={r["|DE|"]:.3f} accel={r["price_accel"]:.4f}')
            
            # 分析为什么预筛失败
            fail = bif_df[bif_df['prescreen'] == False]
            if len(fail) > 0:
                eig_fail = (fail['gate_eig'] == False).sum()
                accel_fail = (fail['gate_accel'] == False).sum()
                liq_fail = (fail['gate_liq'] == False).sum()
                mom_fail = (fail['gate_mom'] == False).sum()
                print(f'\n  预筛失败原因统计 (共{len(fail)}天):')
                print(f'    特征值不够: {eig_fail} ({eig_fail/len(fail)*100:.0f}%)')
                print(f'    加速度不够负: {accel_fail} ({accel_fail/len(fail)*100:.0f}%)')
                print(f'    流动性不足: {liq_fail} ({liq_fail/len(fail)*100:.0f}%)')
                print(f'    动量过低: {mom_fail} ({mom_fail/len(fail)*100:.0f}%)')


if __name__ == '__main__':
    main()
