"""在 solve_q3 之上叠加两项改造：(1)问题二最终预测器 (2)0时两阶段计划。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import battery_core as bc
import solve_q3 as s3
from ts_plan import plan_day_2s
from battery_core import T, EREF, interval

def predict_new(history, date, weeks=8):
    n = len(history); ids = np.arange(max(0, n - min(n, weeks*7)), n)
    same = ids[bc.DATES[ids].dayofweek == date.dayofweek]
    if len(same) == 0: same = ids
    return history[same].mean(axis=0)

def load_pred_new(actual):
    pred = np.zeros((365, T))
    for d in range(1, 365): pred[d] = predict_new(actual[:d], bc.DATES[d])[0]
    return pred

def simulate_day_v2(p, actual, forecasts, load_pred, d, initial, policy='all', step=100.,
                    eta=.9, count=7, kappa_multiplier=1., refund=False, hard_terminal=False,
                    keep_versions=False, adjust_from=36):
    """与 solve_q3.simulate_day 相同，仅r==0分支换成两阶段计划。"""
    active, source_override = s3.POLICIES[policy]
    kappa = p[0]/eta*kappa_multiplier
    energy = initial; results = []; versions = []
    grid = None; reference = None
    active_grid = np.zeros(T); original = np.zeros(T); ref_all = np.full(T, initial)
    source_all = np.zeros(T, int); decision_all = np.zeros(T, int)
    objective = []
    if d == 0:
        q = np.maximum(actual[d,0]-actual[d,1], 0); w = np.maximum(actual[d,1]-actual[d,0], 0)
        out = np.column_stack([np.full(T, initial), np.full(T, initial), np.zeros((T,2)), q, w])
    else:
        for t in range(T):
            r = t//36
            if t % 36 == 0 and r in active:
                source = source_override.get(r, r)
                scene = s3.make_scene(actual, forecasts, load_pred, d, source, t, count)
                if r == 0:
                    G, reference, Aeff, _, _, _ = plan_day_2s(
                        p, scene, energy, eta, kappa, hard_terminal,
                        adjust_from=(144 if not active else 36))
                    grid = G; original = G.copy()
                    grid_value = Aeff.mean(axis=0)      # 价值表按期望实际购电
                    obj = float(p@G + np.mean(np.sum(1.5*p*(Aeff-G), axis=1))
                                + kappa*max(EREF-reference[-1], 0))
                else:
                    grid, reference, obj = s3.adjust_plan(p[t:], scene, energy, original[t:],
                                                          eta, kappa, refund, hard_terminal)
                    grid_value = grid
                objective.append(obj); start = t
                states, values = bc.value_table(p[t:], grid_value, scene, eta, kappa, step, hard_terminal)
                active_grid[t:] = grid; ref_all[t:] = reference
                source_all[t:] = source*6; decision_all[t:] = r*6
                if keep_versions:
                    for k in range(T-t):
                        versions.append((str(bc.DATES[d].date()), r*6, source*6, t+k+1,
                                         float(original[t+k]), float(grid[k]),
                                         float(reference[k]), float(load_pred[d, t+k]),
                                         float(forecasts[d, source, t+k])))
            energy, c, b, q_, w = bc.action(p[t], grid[t-start], actual[d,0,t], actual[d,1,t],
                                            energy, states, values[t-start+1], eta)
            results.append((initial if not results else results[-1][1], energy, c, b, q_, w))
        out = np.asarray(results)
    inc = np.maximum(active_grid-original, 0); dec = np.maximum(original-active_grid, 0)
    df = pd.DataFrame(out, columns=['storage_start_kwh','storage_end_kwh','charge_kwh','discharge_kwh',
                                    'emergency_kwh','unused_kwh'])
    df.insert(0,'date',str(bc.DATES[d].date())); df.insert(1,'slot',np.arange(1,T+1))
    df['interval']=[interval(t) for t in range(T)]
    df['price_yuan_per_kwh']=p; df['load_kwh']=actual[d,0]; df['pv_kwh']=actual[d,1]
    df['original_grid_kwh']=original; df['adjusted_grid_kwh']=active_grid
    df['increase_kwh']=inc; df['decrease_kwh']=dec; df['reference_storage_kwh']=ref_all
    df['forecast_release_hour']=source_all; df['decision_hour']=decision_all
    df['plan_cost_yuan']=p*original; df['increase_cost_yuan']=1.5*p*inc
    df['decrease_cost_yuan']=(-.5 if refund else .5)*p*dec
    df['emergency_cost_yuan']=5*p*df.emergency_kwh
    df['total_cost_yuan']=df[['plan_cost_yuan','increase_cost_yuan','decrease_cost_yuan','emergency_cost_yuan']].sum(axis=1)
    df['forecast_load_kwh']=load_pred[d]
    df['forecast_pv_kwh']=[forecasts[d,int(source_all[t]/6),t] for t in range(T)]
    df['history_latest_date']='' if d==0 else str(bc.DATES[d-1].date())
    daily={'date':str(bc.DATES[d].date()),'policy':policy,'initial_kwh':initial,'final_kwh':float(out[-1,1]),
           'emergency_slots':int((df.emergency_kwh>1e-6).sum()),'solve_count':len(objective)}
    for col in ['original_grid_kwh','adjusted_grid_kwh','increase_kwh','decrease_kwh','emergency_kwh','charge_kwh',
                'discharge_kwh','unused_kwh','plan_cost_yuan','increase_cost_yuan','decrease_cost_yuan',
                'emergency_cost_yuan','total_cost_yuan']:
        daily[col]=float(df[col].sum())
    return df, daily, versions

def run_case_v2(payload):
    p, actual, forecasts, load_pred, days, policy, fair, use_2s = payload
    energy = EREF; tables=[]; daily=[]; versions=[]
    start = 31 if fair else 0
    for d in range(start, days):
        fn = simulate_day_v2 if use_2s else s3.simulate_day
        kw = {'hard_terminal': fair and d==days-1}
        if use_2s: kw['keep_versions'] = policy=='all' and not fair
        else: kw = dict(step=100., eta=.9, count=7, kappa_multiplier=1., refund=False,
                        hard_terminal=fair and d==days-1, keep_versions=policy=='all' and not fair)
        table, ds, vs = fn(p, actual, forecasts, load_pred, d, energy, policy, **kw)
        energy = ds['final_kwh']; tables.append(table); daily.append(ds); versions.extend(vs)
    table = pd.concat(tables, ignore_index=True); checks = s3.validate(table)
    if fair: assert abs(energy-EREF) < 1e-5
    return {'policy':policy,'fair':fair,'daily':pd.DataFrame(daily),'table':table,
            'versions':versions,'validation':checks}
