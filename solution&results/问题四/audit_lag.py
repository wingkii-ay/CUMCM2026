"""How much does the one declared idealisation buy?

Q4 assumes the current 10-minute slot's own averaged load / PV / price are
observable when that slot's action is chosen ("段内即时响应是理想化").  Use of
*a later* slot is what would count as leakage; the current slot is a stated
modelling shortcut, not a leak.  This script measures its price tag by rerunning
the year with every real-time observation pushed back one slot.

lag=0 must reproduce core.simulate_day bit-for-bit, which is asserted first, so
the lagged column below is a like-for-like comparison and not a rewrite.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import core
from core import INPUT, DATES, T, read_data, predictions, make_scene, plan, value_table, action
ROOT = Path(__file__).resolve().parent


def simulate_day_lagged(data, fc, base, prices, d, initial, branch=2, policy='all',
                        eta=.9, step=100., lag=0):
    """core.simulate_day with the real-time observation of slot t taken from t-lag."""
    energy = initial
    original = np.zeros(T); active = np.zeros(T)
    point_used = np.zeros((3, T)); decision = np.zeros(T, int); source_used = np.full(T, -1, int)
    rows = []
    o = lambda t: max(0, t - lag)
    if d == 0:
        for t in range(T):
            rows.append([energy, energy, 0., 0., max(data[d, 0, t] - data[d, 1, t], 0), max(data[d, 1, t] - data[d, 0, t], 0)])
        basegrid = np.zeros(T)
    else:
        kappa = base[d, 2, 0] / eta
        adjustable = 36 if (branch == 3 and policy != 'only0') else None
        for t in range(T):
            if t % 36 == 0:
                r = t // 36
                source = None if branch == 2 else (0 if policy in ('only0', 'stale_all') else (2 if policy == 'stale18' and r == 3 else r))
                scene, point = make_scene(data, fc, base, prices, d, r, source)
                if t == 0:
                    grid, ref, obj = plan(scene, energy, kappa, eta=eta, adj_from=adjustable)
                    original = grid.copy(); active[:] = grid; basegrid = core.baseline(scene)
                elif branch == 3 and policy != 'only0':
                    grid, ref, obj = plan(scene, energy, kappa, original[t:], eta)
                    active[t:] = grid
                states, values = value_table(scene, active[t:], kappa, eta, step)
                point_used[:, t:] = point; start = t
            old = energy
            energy, c, b, e, w = action(data[d, 2, o(t)], active[t], data[d, 0, o(t)], data[d, 1, o(t)],
                                         energy, states, values[t - start + 1], eta)
            rows.append([old, energy, c, b, e, w])
    df = pd.DataFrame(rows, columns=['storage_start_kwh', 'storage_end_kwh', 'charge_kwh', 'discharge_kwh',
                                     'emergency_kwh', 'unused_kwh'])
    df.insert(0, 'date', str(DATES[d].date())); df.insert(1, 'slot', np.arange(1, 145))
    df['interval'] = [core.interval(t) for t in range(T)]
    for i, col in enumerate(['load_kwh', 'pv_kwh', 'price_yuan_per_kwh']):
        df[col] = data[d, i]
    df['original_grid_kwh'] = original; df['adjusted_grid_kwh'] = active
    p = data[d, 2]
    df['total_cost_yuan'] = p * (original + 1.5 * np.maximum(active - original, 0)
                                 + .5 * np.maximum(original - active, 0) + 5 * df.emergency_kwh)
    return df


def year(data, fc, base, prices, branch, lag):
    rows = []; energy = 6000.0
    for d in range(365):
        df = simulate_day_lagged(data, fc, base, prices, d, energy, branch, lag=lag)
        energy = float(df.storage_end_kwh.iloc[-1])
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=INPUT)
    p.add_argument('--output', type=Path, default=ROOT / 'results')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--lags', type=int, nargs='+', default=[1, 2])
    a = p.parse_args()
    data, fc = read_data(a.input)
    base, prices = predictions(data)
    out = {}
    for branch in (2, 3):
        delivered = pd.read_csv(a.output / f'q4{branch}_dispatch_all.csv')
        ev0 = delivered[delivered.date >= '2025-02-01']
        rep = {'branch': branch,
               'cost_lag0': float(ev0.total_cost_yuan.sum()),
               'emergency_lag0': float(ev0.emergency_kwh.sum()),
               'lag0_source': f'q4{branch}_dispatch_all.csv (delivered artefact)'}
        real = None
        for lag in a.lags:
            if real is None:
                real = year(data, fc, base, prices, branch, lag=0)
                # cross-check: the reimplementation must reproduce the delivered ledger
                mine = float(real[real.date >= '2025-02-01'].total_cost_yuan.sum())
                assert abs(mine - rep['cost_lag0']) < 1e-4, (mine, rep['cost_lag0'])
                rep['lag0_vs_delivered_error_yuan'] = abs(mine - rep['cost_lag0'])
            got = year(data, fc, base, prices, branch, lag=lag)
            chk = float(np.max(np.abs(real[['storage_end_kwh', 'adjusted_grid_kwh']].to_numpy()
                                      - got[['storage_end_kwh', 'adjusted_grid_kwh']].to_numpy())))
            evg = got[got.date >= '2025-02-01']
            rep[f'cost_lag{lag}'] = float(evg.total_cost_yuan.sum())
            rep[f'emergency_lag{lag}'] = float(evg.emergency_kwh.sum())
            rep[f'cost_delta_pct_lag{lag}'] = 100 * (rep[f'cost_lag{lag}'] / rep['cost_lag0'] - 1)
            rep[f'max_state_change_lag{lag}'] = chk
        out[f'q4{branch}'] = rep
        print(json.dumps(rep, ensure_ascii=False, indent=2), flush=True)
    out['note'] = ('lag>=1 is strictly causal-only observation; the gap is the value of the declared '
                   'within-slot idealisation. It is a modelling shortcut, not a use of future slots.')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'lag_sensitivity.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
