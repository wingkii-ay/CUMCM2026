"""E2: 全年回测对比预测器的影响（主策略 all，非公平边界）。"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
import battery_core as bc
import solve_q3 as s3

folder = Path('C题/附件')
p, actual = bc.read_data(folder)
forecast, hourly = s3.read_forecasts(folder, actual)

def predict_new(history, date, weeks=8):
    n = len(history); ids = np.arange(n - min(n, weeks*7), n)
    same = ids[bc.DATES[ids].dayofweek == date.dayofweek]
    if len(same) == 0: same = ids
    return history[same].mean(axis=0)

def make(fn):
    pred = np.zeros((365, bc.T))
    for d in range(1, 365): pred[d] = fn(actual[:d], bc.DATES[d])[0]
    return pred

cases = {
 'A_current': make(lambda h,dt: bc.predict(h, dt, 7)),
 'B_weekday8': make(predict_new),
}
out = {}
for tag, lp in cases.items():
    t0 = time.perf_counter()
    res = s3.run_case((p, actual, forecast, lp, 365, 'all', False))
    ds = res['daily']; ev = ds[ds.date >= '2025-02-01']
    out[tag] = {k: float(ev[k].sum()) for k in ['plan_cost_yuan','increase_cost_yuan','decrease_cost_yuan',
                 'emergency_cost_yuan','total_cost_yuan','original_grid_kwh','adjusted_grid_kwh',
                 'increase_kwh','emergency_kwh','unused_kwh','charge_kwh','discharge_kwh']}
    out[tag]['final_kwh'] = float(ev.final_kwh.iloc[-1])
    out[tag]['runtime_s'] = round(time.perf_counter()-t0,1)
    print(tag, json.dumps(out[tag], indent=1, ensure_ascii=False), flush=True)

a, b = out['A_current'], out['B_weekday8']
print('\n=== 影响 ===')
print('总费用 %,.0f -> %,.0f  节省 %,.0f (%.2f%%)' % (a['total_cost_yuan'], b['total_cost_yuan'],
      a['total_cost_yuan']-b['total_cost_yuan'], (a['total_cost_yuan']-b['total_cost_yuan'])/a['total_cost_yuan']*100))
print('原计划费 %,.0f -> %,.0f' % (a['plan_cost_yuan'], b['plan_cost_yuan']))
print('应急费   %,.0f -> %,.0f' % (a['emergency_cost_yuan'], b['emergency_cost_yuan']))
print('未使用   %,.0f -> %,.0f kWh' % (a['unused_kwh'], b['unused_kwh']))
Path('_exp/exp2.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
