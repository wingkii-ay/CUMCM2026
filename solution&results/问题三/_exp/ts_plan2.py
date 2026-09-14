"""0时两阶段随机规划 + 字典序平滑择优。

在 ts_plan.plan_day_2s 基础上加一步：先求纯经济最优值 OPT，再在
"经济目标 <= OPT + slack" 的约束下最小化 G 的全变差，用于打破 LP 退化顶点。
这样平滑项不可能改变经济含义（代价上界 slack 元/日）。
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from battery_core import EMIN, EMAX, EREF, POWER


def _build(p, scene, initial, eta, kappa, hard_terminal, adjust_from, with_smooth):
    s, n = scene.shape[0], len(p)
    ig = np.arange(0, n)
    ic = np.arange(n, 2 * n); ib = np.arange(2 * n, 3 * n); ie = np.arange(3 * n, 4 * n)
    base = 4 * n
    iA = np.arange(base, base + s * n).reshape(s, n)
    iq = np.arange(base + s * n, base + 2 * s * n).reshape(s, n)
    ih = base + 2 * s * n; nv = ih + 1
    adj = np.zeros(n, bool); adj[adjust_from:] = True
    nfix = int((~adj).sum())

    obj = np.zeros(nv)
    obj[ig] = p - 1.5 * p * adj
    obj[iA] = 1.5 * p[None, :] * adj[None, :] / s
    obj[iq] = 5 * p[None, :] / s
    obj[ih] = kappa
    obj[ic] = obj[ib] = 1e-8

    bounds = [(0, None)] * nv
    for t in range(n):
        bounds[ic[t]] = bounds[ib[t]] = (0, POWER / 1000)
        bounds[ie[t]] = (EMIN / 1000, EMAX / 1000)
    if hard_terminal:
        bounds[ie[-1]] = (EREF / 1000, EREF / 1000)
        for t in range(max(0, n - 7), n): bounds[ig[t]] = (POWER / 1000, None)

    neq = n + s * nfix
    A_eq = lil_matrix((neq, nv + (n - 1 if with_smooth else 0))); b_eq = np.zeros(neq)
    for t in range(n):
        A_eq[t, ie[t]] = 1; A_eq[t, ic[t]] = -eta; A_eq[t, ib[t]] = 1 / eta
        if t: A_eq[t, ie[t - 1]] = -1
        else: b_eq[t] = initial / 1000
    fx = np.flatnonzero(~adj)
    for j in range(s):
        for k, t in enumerate(fx):
            A_eq[n + j * nfix + k, iA[j, t]] = 1
            A_eq[n + j * nfix + k, ig[t]] = -1

    if not with_smooth:
        nrow = s * n * 2 + s * int(adj.sum()) + 1
        idv = None
    else:
        idv = np.arange(nv, nv + n - 1)
        nrow = s * n * 2 + s * int(adj.sum()) + 1 + 2 * (n - 1)
    A_ub = lil_matrix((nrow, nv + (n - 1 if with_smooth else 0))); b_ub = np.zeros(nrow)
    r = 0
    for j in range(s):
        for t in range(n):
            A_ub[r, ic[t]] = 1.; A_ub[r, ib[t]] = -1.
            A_ub[r, iA[j, t]] = -1.; A_ub[r, iq[j, t]] = -1.
            b_ub[r] = (scene[j, 1, t] - scene[j, 0, t]) / 1000; r += 1
            A_ub[r, ic[t]] = 1.; A_ub[r, iA[j, t]] = -1.
            b_ub[r] = scene[:, 1, t].min() / 1000; r += 1
            if adj[t]:
                A_ub[r, ig[t]] = 1.; A_ub[r, iA[j, t]] = -1.; r += 1
    A_ub[r, ie[-1]] = -1.; A_ub[r, ih] = -1.; b_ub[r] = -EREF / 1000; r += 1
    if with_smooth:
        for t in range(1, n):
            A_ub[r, ig[t]] = 1.; A_ub[r, ig[t - 1]] = -1.; A_ub[r, idv[t - 1]] = -1.; r += 1
            A_ub[r, ig[t]] = -1.; A_ub[r, ig[t - 1]] = 1.; A_ub[r, idv[t - 1]] = -1.; r += 1
    assert r == nrow
    return dict(ig=ig, ic=ic, ib=ib, ie=ie, iA=iA, iq=iq, ih=ih, nv=nv, idv=idv,
                adj=adj, obj=obj, bounds=bounds, A_eq=A_eq.tocsr(), b_eq=b_eq,
                A_ub=A_ub.tocsr(), b_ub=b_ub, nrow=nrow, s=s, n=n)


def _solve(D, obj, extra_row=None, extra_rhs=None):
    A_ub, b_ub = D['A_ub'], D['b_ub']
    if extra_row is not None:
        from scipy.sparse import coo_matrix, vstack
        data, cols = extra_row
        A_ub = vstack([A_ub, coo_matrix((data, (np.zeros_like(cols), cols)),
                                        shape=(1, D['A_ub'].shape[1]))]).tocsr()
        b_ub = np.append(b_ub, extra_rhs)
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=D['A_eq'], b_eq=D['b_eq'],
                  bounds=D['bounds'], method='highs',
                  options={'primal_feasibility_tolerance': 1e-8,
                           'dual_feasibility_tolerance': 1e-8})
    if not res.success: raise RuntimeError(res.message)
    return res


def plan_day_2s(p, scene, initial, eta=.9, kappa=None, hard_terminal=False,
                adjust_from=36, smooth_slack=1e-2):
    """返回 (G, 参考库存, 场景实际购电A, 充电, 放电, 经济最优值)。"""
    if kappa is None: kappa = p[0] / eta
    D = _build(p, scene, initial, eta, kappa, hard_terminal, adjust_from, False)
    nv, n, s = D['nv'], D['n'], D['s']
    res = _solve(D, D['obj'])
    opt = float(res.fun * 1000)

    D2 = _build(p, scene, initial, eta, kappa, hard_terminal, adjust_from, True)
    idv = D2['idv']
    nvt = nv + (n - 1)
    D2['bounds'] = D2['bounds'] + [(0, None)] * (n - 1)
    obj2 = np.zeros(nvt); obj2[:nv] = D2['obj']; obj2[idv] = 1.0
    # 字典序约束：经济目标 <= OPT + slack（变量单位 MWh，故 RHS 除以 1000）
    nz = np.flatnonzero(D2['obj'])
    extra = (D2['obj'][nz] / 1.0, nz)
    r2 = _solve(D2, obj2, extra, (opt + smooth_slack) / 1000)
    x = r2.x * 1000
    grid = np.maximum(x[D2['ig']], 0); Aeff = np.maximum(x[D2['iA']], 0)
    cancel = np.minimum(x[D2['ic']], x[D2['ib']] / eta ** 2)
    charge = np.maximum(x[D2['ic']] - cancel, 0); discharge = np.maximum(x[D2['ib']] - eta ** 2 * cancel, 0)
    return grid, x[D2['ie']], Aeff, charge, discharge, opt


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from pathlib import Path
    import battery_core as bc, solve_q3 as s3
    from ts_plan import plan_day_2s as old
    p, actual = bc.read_data(Path('C题/附件'))
    fc, _ = s3.read_forecasts(Path('C题/附件'), actual)
    lp = np.load('_exp/loadpred_B.npy')
    for ds in ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']:
        d = bc.DATES.get_loc(ds)
        sc = s3.make_scene(actual, fc, lp, d, 0, 0)
        init = 6200. if ds != '2025-09-23' else 5979.173955
        g0, _, _, _, _, o0 = old(p, sc, init, .9, p[0] / .9, False, 36)
        g1, st, A, c, b, o1 = plan_day_2s(p, sc, init, .9, p[0] / .9, False, 36)
        print(f'--- {ds} ---')
        print(f'  旧 经济最优 {o0:12,.2f} | 全变差 {np.abs(np.diff(g0)).sum():10,.0f}')
        print(f'  新 经济最优 {o1:12,.2f} | 全变差 {np.abs(np.diff(g1)).sum():10,.0f}')
        print(f'  ΣG 旧 {g0.sum():.1f} 新 {g1.sum():.1f}  ΔE(A-G) 新 {np.mean((A-g1).sum(axis=1)):.1f}')
        print(f'  E[A]-G 差值 旧 {np.abs(A.mean(axis=0)-g0).sum():.0f} 新 {np.abs(A.mean(axis=0)-g1).sum():.0f}')
        idx = [60, 72, 84, 96, 108, 120]
        print('  指定6段 旧:', np.round(g0[idx], 1))
        print('  指定6段 新:', np.round(g1[idx], 1))
