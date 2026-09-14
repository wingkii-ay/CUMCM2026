"""问题二：历史联合场景、日前承诺、因果储能反馈。原始附件只读。"""
from pathlib import Path
from datetime import datetime, timedelta
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
import openpyxl
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parent
INPUT = Path('D:/Users/NGALAM/OneDrive/桌面/CUMCM2026Problems/C题/附件')
DATES = pd.date_range('2025-01-01', '2025-12-31')
SELECTED = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
EMIN, EMAX, EREF, POWER = 1200., 10800., 6000., 5000 / 6
T = 144

def label(slot):
    m = slot * 10
    return f'{m//60:02d}:{m%60:02d}'

def interval(slot):
    return label(slot) + '-' + label(slot+1)

def minute(x):
    if hasattr(x, 'hour'):
        return x.hour*60+x.minute
    s = str(x); h, m = map(int, s.split('+')[0].split(':')[:2])
    return 60*h+m+(1440 if '+1' in s else 0)

def read_data(folder):
    w = openpyxl.load_workbook(folder/'附件1.xlsx', read_only=True, data_only=True)
    rows = list(w.active.values); w.close()
    assert len(rows) == 145 and rows[0][1] == '电价'
    assert [minute(r[0]) for r in rows[1:]] == list(range(10,1441,10))
    p = np.array([r[1] for r in rows[1:]],float)
    w = openpyxl.load_workbook(folder/'附件2.xlsx', read_only=True, data_only=True)
    arrays = []
    for name in ['小区负载','光伏发电实际功率']:
        rows = list(w[name].values)
        assert len(rows)==366 and all(len(r)==145 for r in rows)
        assert [minute(x) for x in rows[0][1:]] == list(range(10,1441,10))
        assert pd.DatetimeIndex([r[0] for r in rows[1:]]).equals(DATES)
        a = np.asarray([r[1:] for r in rows[1:]],float)/6
        assert a.shape==(365,144) and np.isfinite(a).all() and (a>=0).all()
        arrays.append(a)
    w.close()
    assert np.isfinite(p).all() and (p>0).all()
    return p, np.stack(arrays,axis=1)  # 日、负载/光伏、10分钟；单位kWh

def predict(history, date, window):
    """只接收截至昨天的历史数组。周末按星期定义，不假造调休信息。"""
    n = len(history)
    assert n > 0
    ids = np.arange(max(0,n-window),n)
    same = ids[(DATES[ids].dayofweek >= 5) == (date.dayofweek >= 5)]
    ids = same if len(same)>=2 else ids
    return history[ids].mean(axis=0)

def scenarios(pred, residuals, history, count=7):
    if len(residuals) >= 2:
        return np.maximum(0, pred[None,:,:]+np.asarray(residuals[-count:]))
    return history[-count:].copy()

def plan_day(p, scene, initial, eta=.9, kappa=None, hard_terminal=False):
    """共享电池参考轨迹的场景LP。内部以MWh缩放，目标单位千元。"""
    if kappa is None: kappa=p[0]/eta
    s,n = scene.shape[0],len(p)
    g,c,b,e = [np.arange(k*n,(k+1)*n) for k in range(4)]
    q = np.arange(4*n,(4+s)*n).reshape(s,n)
    h=(4+s)*n; nv=h+1
    obj=np.zeros(nv);obj[g]=p;obj[q]=5*p[None,:]/s;obj[h]=kappa
    obj[c]=obj[b]=1e-8  # 小于1分钱的吞吐量择优扰动，另报告原始经济目标。
    bounds=[(0,None)]*nv
    for t in range(n):
        bounds[c[t]]=bounds[b[t]]=(0,POWER/1000)
        bounds[e[t]]=(EMIN/1000,EMAX/1000)
    if hard_terminal:
        bounds[e[-1]]=(EREF/1000,EREF/1000)
        # 独立公平边界实验：最后7段保留足够普通电源，确保实际闭环可达。
        for t in range(n-7,n): bounds[g[t]]=(POWER/1000,None)
    eq=lil_matrix((n,nv));rhs=np.zeros(n)
    for t in range(n):
        eq[t,e[t]]=1;eq[t,c[t]]=-eta;eq[t,b[t]]=1/eta
        if t: eq[t,e[t-1]]=-1
        else: rhs[t]=initial/1000
    a=lil_matrix((s*n+n+1,nv));ub=np.zeros(s*n+n+1)
    for j in range(s):
        for t in range(n):
            row=j*n+t
            a[row,g[t]]=-1;a[row,b[t]]=-1;a[row,c[t]]=1;a[row,q[j,t]]=-1
            ub[row]=(scene[j,1,t]-scene[j,0,t])/1000
    for t in range(n):
        a[s*n+t,c[t]]=1;a[s*n+t,g[t]]=-1
        ub[s*n+t]=scene[:,1,t].min()/1000
    a[-1,e[-1]]=-1;a[-1,h]=-1;ub[-1]=-EREF/1000
    res=linprog(obj,A_ub=a.tocsr(),b_ub=ub,A_eq=eq.tocsr(),b_eq=rhs,bounds=bounds,
                method='highs',options={'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    if not res.success: raise RuntimeError(res.message)
    x=res.x*1000
    # 任意同时充放解可消去无效循环：状态不变，剩余电量增加，费用不升。
    cancel=np.minimum(x[c],x[b]/eta**2)
    charge=np.maximum(0,x[c]-cancel); discharge=np.maximum(0,x[b]-eta**2*cancel)
    grid=np.maximum(0,x[g]); storage=x[e]
    short=np.maximum(scene[:,0,:]+charge-grid-scene[:,1,:]-discharge,0)
    economic=float(p@grid+np.mean(short@(5*p))+kappa*max(EREF-storage[-1],0))
    err=max(float(np.abs(np.r_[initial,storage[:-1]]+eta*charge-discharge/eta-storage).max()),
            float(np.maximum(charge-grid-scene[:,1,:].min(axis=0),0).max()))
    assert err < 1e-4 and np.max(np.minimum(charge,discharge))<1e-5
    return grid, storage, {'plan_objective_yuan':economic,'lp_objective_yuan':float(res.fun*1000),
                           'lp_iterations':int(res.nit),'plan_violation_kwh':err}

def value_table(p, grid, scene, eta=.9, kappa=None, step=100., hard_terminal=False):
    """共享动作的边际场景价值近似；不冒充保留全部时序相关性的精确多阶段DP。"""
    if kappa is None:kappa=p[0]/eta
    states=np.arange(EMIN,EMAX+step/2,step)
    assert abs(states[-1]-EMAX)<1e-8
    delta=states[None,:]-states[:,None]
    charge=np.maximum(delta,0)/eta; discharge=np.maximum(-delta,0)*eta
    bus=charge-discharge
    valid=(charge<=POWER+1e-8)&(discharge<=POWER+1e-8)
    value=np.empty((len(p)+1,len(states)))
    value[-1]=kappa*np.maximum(EREF-states,0)
    if hard_terminal:value[-1]=np.where(np.abs(states-EREF)<1e-8,0,np.inf)
    for t in range(len(p)-1,-1,-1):
        net=scene[:,0,t]-scene[:,1,t]-grid[t]
        cost=np.maximum(net[:,None,None]+bus[None,:,:],0).mean(axis=0)*(5*p[t])
        # 仅约束普通电源可供充电；应急用于负载，约束与日前模型一致。
        allowed=valid & (charge<=grid[t]+scene[:,1,t].min()+1e-8)
        total=cost+value[t+1][None,:]
        total[~allowed]=np.inf
        value[t]=np.min(total,axis=1)
    return states,value

def action(p, grid, load, pv, energy, states, future_value, eta=.9):
    """只使用本段观测，不接收当天未来实际值；储能在网格间作分段线性插值。"""
    lo=max(EMIN,energy-POWER/eta)
    hi=min(EMAX,energy+eta*min(POWER,grid+pv))
    net=load-pv-grid
    balanced=energy-net/eta if net>=0 else energy-net*eta
    candidates=np.unique(np.r_[states[(states>=lo)&(states<=hi)],lo,hi,
                               np.clip(energy,lo,hi),np.clip(balanced,lo,hi)])
    change=candidates-energy
    c=np.maximum(change,0)/eta;b=np.maximum(-change,0)*eta
    emergency=np.maximum(net+c-b,0)
    values=5*p*emergency+np.interp(candidates,states,future_value)
    finite=np.isfinite(values)
    if not finite.any():raise RuntimeError('No feasible real-time action')
    best=values[finite].min()
    # 数值等价时最小吞吐量；不把实际吞吐惩罚计入费用。
    ids=np.flatnonzero(finite & (values<=best+1e-7))
    j=ids[np.argmin((c+b)[ids])]
    unused=max(grid+pv+b[j]+emergency[j]-load-c[j],0)
    return float(candidates[j]),float(c[j]),float(b[j]),float(emergency[j]),float(unused)

def execute(p, grid, scene, actual, initial, eta=.9, kappa=None, step=100., hard_terminal=False):
    states,value=value_table(p,grid,scene,eta,kappa,step,hard_terminal)
    energy=initial;rows=[]
    for t in range(len(p)):
        start=energy
        energy,c,b,q,w=action(p[t],grid[t],actual[0,t],actual[1,t],start,states,value[t+1],eta)
        rows.append((start,energy,c,b,q,w))
    return np.array(rows)

def baseline_grid(scene):
    # min pG+5p E[(L-V-G)+] 的经验0.8分位数；不含电池。
    x=np.sort(scene[:,0,:]-scene[:,1,:],axis=0)
    return np.maximum(0,x[int(np.ceil(.8*len(x)))-1])

def episodes(table):
    rows=[]
    for date,d in table.groupby('date',sort=False):
        q=d.emergency_kwh.to_numpy();start=None
        for t in range(T+1):
            active=t<T and q[t]>1e-6
            if active and start is None:start=t
            if not active and start is not None:
                rows.append({'date':date,'interval':label(start)+'-'+label(t),'emergency_kwh':float(q[start:t].sum())})
                start=None
    return pd.DataFrame(rows,columns=['date','interval','emergency_kwh'])

