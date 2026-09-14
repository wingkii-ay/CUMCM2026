"""问题三完美预知下界：全时段联合LP，量化滚动策略与理论最优的差距。

完全预知意味着不存在承诺与实际购电的偏差，故式(10)中 u=v=e=0，总费用即 Σp·g。
这是任何因果策略（含本策略）都无法逾越的下界，不是可达成绩。
只读 dispatch.csv（与主策略完全相同的负载/光伏/电价口径），不修改任何既有产出。
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack as sparse_vstack

ROOT = Path(__file__).resolve().parent
EMIN, EMAX, EREF, POWER, ETA = 1200., 10800., 6000., 5000 / 6, .9

d = pd.read_csv(ROOT / 'dispatch.csv')
price = d.price_yuan_per_kwh.to_numpy(float)
load = d.load_kwh.to_numpy(float)
pv = d.pv_kwh.to_numpy(float)
n = len(price)
E0 = float(d.storage_start_kwh.iloc[0])       # 评价期实际期初库存
Ef = float(d.storage_end_kwh.iloc[-1])        # 评价期实际期末库存

ig, ic, ib, ie = (np.arange(k * n, (k + 1) * n) for k in range(4))
nv = 4 * n
obj = np.zeros(nv); obj[ig] = price

# 储能动态 e[t] - e[t-1] - eta*c[t] + b[t]/eta = (E0 if t==0 else 0)
rows = np.concatenate([np.arange(n), np.arange(1, n), np.arange(n), np.arange(n)])
cols = np.concatenate([ie, ie[:-1], ic, ib])
vals = np.concatenate([np.ones(n), -np.ones(n - 1), -np.full(n, ETA), np.full(n, 1 / ETA)])
A_eq = coo_matrix((vals, (rows, cols)), shape=(n, nv)).tocsr()
b_eq = np.zeros(n); b_eq[0] = E0

# 平衡 g + b - c >= load - pv
r1 = np.arange(n)
A_ub = coo_matrix((np.concatenate([-np.ones(n), -np.ones(n), np.ones(n)]),
                   (np.concatenate([r1, r1, r1]), np.concatenate([ig, ib, ic]))),
                  shape=(n, nv)).tocsr()
b_ub = pv - load
# 充电只能来自普通电源或光伏 c - g <= pv
r2 = np.arange(n)
A_ub = sparse_vstack([A_ub, coo_matrix((np.concatenate([np.ones(n), -np.ones(n)]),
        (np.concatenate([r2, r2]), np.concatenate([ic, ig]))), shape=(n, nv)).tocsr()]).tocsr()
b_ub = np.concatenate([b_ub, pv])

bounds = [(0, None)] * nv
for i in range(n):
    bounds[ic[i]] = bounds[ib[i]] = (0, POWER)
    bounds[ie[i]] = (EMIN, EMAX)


def solve(final_min):
    """final_min=None 表示期末自由；否则要求期末库存不低于该值。"""
    A, b = A_ub, b_ub
    if final_min is not None:
        A = sparse_vstack([A, coo_matrix(([-1.], ([0], [ie[-1]])), shape=(1, nv)).tocsr()]).tocsr()
        b = np.append(b, -final_min)
    res = linprog(obj, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs',
                  options={'primal_feasibility_tolerance': 1e-8,
                           'dual_feasibility_tolerance': 1e-8})
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x
    return {'status': 'optimal', 'total_cost_yuan': float(price @ x[ig]),
            'grid_kwh': float(x[ig].sum()), 'charge_kwh': float(x[ic].sum()),
            'discharge_kwh': float(x[ib].sum()), 'final_kwh': float(x[ie][-1]),
            'min_soc': float(x[ie].min()), 'max_soc': float(x[ie].max())}


if __name__ == '__main__':
    out = {'period': f'{d.date.iloc[0]} ~ {d.date.iloc[-1]}', 'slots': int(n),
           'initial_kwh': E0, 'actual_final_kwh': Ef,
           'final_free': solve(None), 'final_ge_actual': solve(Ef)}
    print(f'slots={n}  E0={E0}  E_actual_final={Ef}')
    print('期末自由    :', f"{out['final_free']['total_cost_yuan']:,.2f}")
    print('期末>=实际  :', f"{out['final_ge_actual']['total_cost_yuan']:,.2f}")

    s = json.loads((ROOT / 'summary.json').read_text(encoding='utf-8'))
    actual = s['evaluation']['total_cost_yuan']
    out['main_strategy_total_cost_yuan'] = actual
    out['gap_yuan'] = actual - out['final_ge_actual']['total_cost_yuan']
    out['gap_pct'] = out['gap_yuan'] / actual * 100
    print(f'主策略        : {actual:,.2f}')
    print(f'可优化空间    : {out["gap_yuan"]:,.2f}  ({out["gap_pct"]:.2f}%)')
    (ROOT / 'results' / 'perfect_foresight_bound.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已写入 results/perfect_foresight_bound.json')
