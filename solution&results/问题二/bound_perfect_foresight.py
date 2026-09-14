"""完美预知下界：全时段联合LP，量化当前滚动策略与理论最优的差距。

只读 dispatch.csv（与主策略完全相同的负载/光伏/电价口径），不修改任何既有产出。
变量：g=计划购电, c=充电, b=放电, q=应急购电, e=储能电量。
完美预知意味着不存在"承诺未用"，故不设 unused 变量，代之以平衡不等式。
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack as sparse_vstack

ROOT = Path(__file__).resolve().parent
EMIN, EMAX, EREF = 1200., 10800., 6000.
POWER = 5000 / 6          # 单时段最大充/放电量 kWh
ETA = 0.9

d = pd.read_csv(ROOT / 'dispatch.csv')
price = d.price_yuan_per_kwh.to_numpy(float)
load = d.load_kwh.to_numpy(float)
pv = d.pv_kwh.to_numpy(float)
n = len(price)
E0 = float(d.storage_start_kwh.iloc[0])

ig, ic, ib, iq, ie = (np.arange(k * n, (k + 1) * n) for k in range(5))
nv = 5 * n

# ---- 目标：min sum p*g + 5*p*q ----
obj = np.zeros(nv)
obj[ig] = price
obj[iq] = 5 * price

# ---- 等式：储能动态 e[t] - e[t-1] - eta*c[t] + b[t]/eta = (E0 if t==0 else 0)
# 行 = 方程编号 0..n-1，列 = 变量编号
rows = np.concatenate([np.arange(n), np.arange(1, n), np.arange(n), np.arange(n)])
cols = np.concatenate([ie, ie[:-1], ic, ib])
vals = np.concatenate([np.ones(n), -np.ones(n - 1), -np.full(n, ETA), np.full(n, 1 / ETA)])
A_eq = coo_matrix((vals, (rows, cols)), shape=(n, nv)).tocsr()
b_eq = np.zeros(n)
b_eq[0] = E0

# ---- 不等式1：平衡 g+b+q-c >= load-pv  ->  -g-b-q+c <= pv-load
r1 = np.arange(n)
rows = np.concatenate([r1, r1, r1, r1])
cols = np.concatenate([ig, ib, iq, ic])
vals = np.concatenate([-np.ones(n), -np.ones(n), -np.ones(n), np.ones(n)])

# ---- 不等式2：充电只能来自普通电源或光伏 c-g <= pv
r2 = np.arange(n, 2 * n)
rows = np.concatenate([rows, r2, r2])
cols = np.concatenate([cols, ic, ig])
vals = np.concatenate([vals, np.ones(n), -np.ones(n)])

A_ub = coo_matrix((vals, (rows, cols)), shape=(2 * n, nv)).tocsr()
b_ub = np.concatenate([pv - load, pv])

bounds = [(0, None)] * nv
for i in range(n):
    bounds[ic[i]] = bounds[ib[i]] = (0, POWER)
    bounds[ie[i]] = (EMIN, EMAX)


def solve(final_min):
    """final_min=None 表示期末自由；否则要求期末库存不低于该值。"""
    A, b = A_ub, b_ub
    if final_min is not None:
        # -e[-1] <= -final_min  <=>  e[-1] >= final_min
        extra = coo_matrix(([-1.], ([0], [ie[-1]])), shape=(1, nv)).tocsr()
        A = sparse_vstack([A, extra]).tocsr()
        b = np.append(b, -final_min)
    res = linprog(obj, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method='highs',
                  options={'primal_feasibility_tolerance': 1e-8,
                           'dual_feasibility_tolerance': 1e-8})
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x
    return {
        'status': 'optimal',
        'total_cost_yuan': float(price @ x[ig] + 5 * price @ x[iq]),
        'plan_cost_yuan': float(price @ x[ig]),
        'emergency_cost_yuan': float(5 * price @ x[iq]),
        'emergency_kwh': float(x[iq].sum()),
        'grid_kwh': float(x[ig].sum()),
        'charge_kwh': float(x[ic].sum()),
        'discharge_kwh': float(x[ib].sum()),
        'final_kwh': float(x[ie[-1]]),
        'min_soc': float(x[ie].min()),
        'max_soc': float(x[ie].max()),
        'duration_seconds': float(res.execution_time) if hasattr(res, 'execution_time') else None,
    }


if __name__ == '__main__':
    out = {'period': f'{d.date.iloc[0]} ~ {d.date.iloc[-1]}',
           'slots': int(n), 'initial_kwh': E0}
    print(f'slots={n}  E0={E0}')
    out['final_free'] = solve(None)
    print('期末自由      :', f"{out['final_free']['total_cost_yuan']:,.2f}")
    out['final_at_eref'] = solve(EREF)
    print('期末>=6000    :', f"{out['final_at_eref']['total_cost_yuan']:,.2f}")

    import json as _json
    s = _json.loads((ROOT / 'summary.json').read_text(encoding='utf-8'))
    actual = s['evaluation']['total_cost_yuan']
    out['main_strategy_total_cost_yuan'] = actual
    out['gap_yuan'] = actual - out['final_at_eref']['total_cost_yuan']
    out['gap_pct'] = out['gap_yuan'] / actual * 100
    print(f'主策略        : {actual:,.2f}')
    print(f'可优化空间    : {out["gap_yuan"]:,.2f}  ({out["gap_pct"]:.2f}%)')
    (ROOT / 'results' / 'perfect_foresight_bound.json').write_text(
        _json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已写入 results/perfect_foresight_bound.json')
