"""问题三完美预知下界：全时段联合LP。完全预知下无承诺偏差，故u=v=e=0，费用=Σp·g。"""
from pathlib import Path
import numpy as np, pandas as pd, json
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack as sparse_vstack
ROOT = Path(__file__).resolve().parent.parent
EMIN, EMAX, EREF, POWER, ETA = 1200., 10800., 6000., 5000/6, .9
d = pd.read_csv(ROOT/'dispatch.csv')
price = d.price_yuan_per_kwh.to_numpy(float); load = d.load_kwh.to_numpy(float); pv = d.pv_kwh.to_numpy(float)
n = len(price); E0 = float(d.storage_start_kwh.iloc[0]); Ef = float(d.storage_end_kwh.iloc[-1])
ig, ic, ib, ie = (np.arange(k*n, (k+1)*n) for k in range(4)); nv = 4*n
obj = np.zeros(nv); obj[ig] = price
rows = np.concatenate([np.arange(n), np.arange(1, n), np.arange(n), np.arange(n)])
cols = np.concatenate([ie, ie[:-1], ic, ib])
vals = np.concatenate([np.ones(n), -np.ones(n-1), -np.full(n, ETA), np.full(n, 1/ETA)])
A_eq = coo_matrix((vals, (rows, cols)), shape=(n, nv)).tocsr(); b_eq = np.zeros(n); b_eq[0] = E0
r1 = np.arange(n)
A_ub = coo_matrix((np.concatenate([-np.ones(n), -np.ones(n), np.ones(n)]),
                   (np.concatenate([r1, r1, r1]), np.concatenate([ig, ib, ic]))), shape=(n, nv)).tocsr()
b_ub = pv - load
r2 = np.arange(n)
A_ub = sparse_vstack([A_ub, coo_matrix((np.concatenate([np.ones(n), -np.ones(n)]),
        (np.concatenate([r2, r2]), np.concatenate([ic, ig]))), shape=(n, nv)).tocsr()]).tocsr()
b_ub = np.concatenate([b_ub, pv])
bounds = [(0, None)]*nv
for i in range(n): bounds[ic[i]] = bounds[ib[i]] = (0, POWER); bounds[ie[i]] = (EMIN, EMAX)
def solve(fmin):
    A, b = A_ub, b_ub
    if fmin is not None:
        A = sparse_vstack([A, coo_matrix(([-1.], ([0], [ie[-1]])), shape=(1, nv)).tocsr()]).tocsr()
        b = np.append(b, -fmin)
    res = linprog(obj, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs',
                  options={'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    if not res.success: raise RuntimeError(res.message)
    x = res.x
    return {'status':'optimal','total_cost_yuan':float(price@x[ig]),'grid_kwh':float(x[ig].sum()),
            'charge_kwh':float(x[ic].sum()),'discharge_kwh':float(x[ib].sum()),'final_kwh':float(x[ie[-1]]),
            'min_soc':float(x[ie].min()),'max_soc':float(x[ie].max())}
out = {'period':f'{d.date.iloc[0]} ~ {d.date.iloc[-1]}','slots':int(n),'initial_kwh':E0,
       'actual_final_kwh':Ef,'final_free':solve(None),'final_ge_actual':solve(Ef)}
print('期末自由   :', f"{out['final_free']['total_cost_yuan']:,.2f}")
print('期末>=实际 :', f"{out['final_ge_actual']['total_cost_yuan']:,.2f}")
(ROOT/'_exp'/'bound_q3.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
