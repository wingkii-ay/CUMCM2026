"""0时两阶段随机规划：G是全时段承诺，可调整时段的最终购电A随场景>=G。"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from battery_core import EMIN, EMAX, EREF, POWER

def plan_day_2s(p, scene, initial, eta=.9, kappa=None, hard_terminal=False, adjust_from=36):
    if kappa is None: kappa = p[0]/eta
    s, n = scene.shape[0], len(p)
    ig = np.arange(0, n)
    ic = np.arange(n, 2*n); ib = np.arange(2*n, 3*n); ie = np.arange(3*n, 4*n)
    base = 4*n
    iA = np.arange(base, base+s*n).reshape(s, n)
    iq = np.arange(base+s*n, base+2*s*n).reshape(s, n)
    ih = base + 2*s*n; nv = ih + 1
    adj = np.zeros(n, bool); adj[adjust_from:] = True
    nfix = int((~adj).sum())

    obj = np.zeros(nv)
    obj[ig] = p - 1.5*p*adj          # G 的净边际系数
    obj[iA] = 1.5*p[None, :]*adj[None, :]/s   # 不可调整时段 A^s=G，该增量恒为0
    obj[iq] = 5*p[None, :]/s
    obj[ih] = kappa
    obj[ic] = obj[ib] = 1e-8

    bounds = [(0, None)]*nv
    for t in range(n):
        bounds[ic[t]] = bounds[ib[t]] = (0, POWER/1000)
        bounds[ie[t]] = (EMIN/1000, EMAX/1000)
    if hard_terminal:
        bounds[ie[-1]] = (EREF/1000, EREF/1000)
        for t in range(max(0, n-7), n): bounds[ig[t]] = (POWER/1000, None)

    # ---- 等式：储能动态 n 行 + 不可调整时段 A^s = G (s*nfix 行)
    neq = n + s*nfix
    A_eq = lil_matrix((neq, nv)); b_eq = np.zeros(neq)
    for t in range(n):
        A_eq[t, ie[t]] = 1; A_eq[t, ic[t]] = -eta; A_eq[t, ib[t]] = 1/eta
        if t: A_eq[t, ie[t-1]] = -1
        else: b_eq[t] = initial/1000
    fx = np.flatnonzero(~adj)
    for j in range(s):
        for k, t in enumerate(fx):
            A_eq[n + j*nfix + k, iA[j, t]] = 1
            A_eq[n + j*nfix + k, ig[t]] = -1

    # ---- 不等式：每场景每时段 2 行 + 可调整时段 1 行 A>=G；末行终端
    nrow = s*n*2 + s*int(adj.sum()) + 1
    A_ub = lil_matrix((nrow, nv)); b_ub = np.zeros(nrow)
    r = 0
    for j in range(s):
        for t in range(n):
            # (a) c - A - b - q <= (V-L)/1000   [变量单位MWh，右端单位MWh]
            A_ub[r, ic[t]] = 1.; A_ub[r, ib[t]] = -1.
            A_ub[r, iA[j, t]] = -1.; A_ub[r, iq[j, t]] = -1.
            b_ub[r] = (scene[j, 1, t]-scene[j, 0, t])/1000; r += 1
            # (b) c - A <= V_min/1000
            A_ub[r, ic[t]] = 1.; A_ub[r, iA[j, t]] = -1.
            b_ub[r] = scene[:, 1, t].min()/1000; r += 1
            # (c) G - A <= 0  (可调整时段)
            if adj[t]:
                A_ub[r, ig[t]] = 1.; A_ub[r, iA[j, t]] = -1.; r += 1
    A_ub[r, ie[-1]] = -1.; A_ub[r, ih] = -1.; b_ub[r] = -EREF/1000; r += 1
    assert r == nrow

    res = linprog(obj, A_ub=A_ub.tocsr(), b_ub=b_ub, A_eq=A_eq.tocsr(), b_eq=b_eq, bounds=bounds,
                  method='highs', options={'primal_feasibility_tolerance': 1e-8, 'dual_feasibility_tolerance': 1e-8})
    if not res.success: raise RuntimeError(res.message)
    x = res.x*1000
    grid = np.maximum(x[ig], 0); Aeff = np.maximum(x[iA], 0)
    cancel = np.minimum(x[ic], x[ib]/eta**2)
    charge = np.maximum(x[ic]-cancel, 0); discharge = np.maximum(x[ib]-eta**2*cancel, 0)
    return grid, x[ie], Aeff, charge, discharge, float(res.fun*1000)
