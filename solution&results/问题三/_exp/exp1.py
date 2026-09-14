"""E1: 预测器对比 —— Q3 现有 predict(周末/工作日) vs 问题二最终版(精确星期几, 8周)"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
import battery_core as bc
import solve_q3 as s3

folder = Path('C题/附件')
p, actual = bc.read_data(folder)
forecast, hourly = s3.read_forecasts(folder, actual)

# --- A: 现状 Q3 预测器 (周末/工作日, window=7) ---
def pred_old(actual):
    pred = np.zeros((365, bc.T))
    for d in range(1, 365):
        pred[d] = bc.predict(actual[:d], bc.DATES[d], 7)[0]
    return pred

# --- B: 问题二最终预测器 (精确星期几, 8周) ---
WEEKS = 8
def predict_new(history, date, weeks=WEEKS):
    n = len(history); ids = np.arange(n - min(n, weeks*7), n)
    same = ids[bc.DATES[ids].dayofweek == date.dayofweek]
    if len(same) == 0: same = ids
    return history[same].mean(axis=0)

def pred_new(actual):
    pred = np.zeros((365, bc.T))
    for d in range(1, 365):
        pred[d] = predict_new(actual[:d], bc.DATES[d])[0]
    return pred

def mae(lp):
    """评价期 2/1-12/31 的日均 MAE (kWh/时段)"""
    return float(np.abs(lp[31:] - actual[31:,0]).mean())

for tag, fn in [('A 现状(周末/工作日,7天)', pred_old), ('B 问题二最终(精确星期几,8周)', pred_new)]:
    lp = fn(actual)
    print(f'{tag}: 评价期负载预测 MAE = {mae(lp):.3f} kWh/时段')
    np.save(f'_exp/loadpred_{tag[0]}.npy', lp)
