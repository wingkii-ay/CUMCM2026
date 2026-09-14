"""CUMCM C Q4: causal floating-price scheduling and realized feedback.
Energy kWh, price CNY/kWh; LP internally MWh and thousand CNY.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

INPUT=Path(r'D:\Users\NGALAM\OneDrive\桌面\CUMCM2026Problems\C题\附件')
DATES=pd.date_range('2025-01-01','2025-12-31')
SELECTED=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']
T=144; EMIN=1200.; EMAX=10800.; EREF=6000.; POWER=5000/6
PARAMS=dict(load_weeks=8,price_days=28,price_observations=6,decay_slots=36,
            scenes=7,step=100.,eta=.9,price_floor=.001,seed=2026)
SHORT_ADJ=1.5   # 节点后可调整时段缺额的边际代价倍数（合同增购价），见plan

def label(t):return f'{t//6:02d}:{t%6*10:02d}'
def interval(t):return label(t)+'-'+label(t+1)
def minute(x):
    if hasattr(x,'hour'):return x.hour*60+x.minute
    s=str(x); h,m=map(int,s.split('+')[0].split(':')[:2])
    return h*60+m+(1440 if '+1' in s else 0)

def read_data(folder):
    arrays=[]
    for file,sheet,scale in [('附件2.xlsx','小区负载',6),('附件2.xlsx','光伏发电实际功率',6),('附件4.xlsx',None,1)]:
        wb=openpyxl.load_workbook(folder/file,read_only=True,data_only=True)
        ws=wb[sheet] if sheet else wb.active; rows=list(ws.values);wb.close()
        assert len(rows)==366 and all(len(r)==145 for r in rows)
        assert [minute(v) for v in rows[0][1:]]==list(range(10,1441,10))
        assert pd.DatetimeIndex([r[0] for r in rows[1:]]).equals(DATES)
        a=np.array([r[1:] for r in rows[1:]],float)/scale
        assert np.isfinite(a).all() and (a>=0).all()
        arrays.append(a)
    data=np.stack(arrays,axis=1);assert (data[:,2]>0).all()
    wb=openpyxl.load_workbook(folder/'附件3.xlsx',read_only=True,data_only=True)
    rows=list(wb.active.values);wb.close()
    assert len(rows)==1461 and list(rows[0])==['日期','预报时刻']+[f'预报{i}小时' for i in range(1,25)]
    fc=np.full((365,4,T),np.nan);date=None
    for k,row in enumerate(rows[1:]):
        d,r=divmod(k,4);off=r*36
        if row[0] not in ('',None):date=pd.Timestamp(row[0])
        assert date==DATES[d] and minute(row[1])==r*360 and len(row)==26
        v=np.asarray(row[2:],float);assert np.isfinite(v).all() and (v>=0).all()
        anchor=data[d,1,off-1]*6 if r else (data[d-1,1,-1]*6 if d else 0.)
        edges=np.interp(np.arange(T+1)/6,np.arange(25),np.r_[anchor,v])
        fc[d,r,off:]=((edges[:-1]+edges[1:])/12)[:T-off]
    return data,fc

def predictions(data):
    """Each day / release built exclusively from information available then."""
    base=np.zeros_like(data); prices=np.full((365,4,T),np.nan)
    for d in range(1,365):
        ids=np.arange(max(0,d-56),d); same=ids[DATES[ids].dayofweek==DATES[d].dayofweek]
        base[d,:2]=data[same if len(same) else ids,:2].mean(axis=0)
        base[d,2]=data[max(0,d-28):d,2].mean(axis=0)
        for r in range(4):
            off=r*36
            correction=float((data[d,2,max(0,off-6):off]-base[d,2,max(0,off-6):off]).mean()) if off else 0.
            prices[d,r,off:]=np.maximum(.001,base[d,2,off:]+correction*np.exp(-np.arange(T-off)/36))
    return base,prices

def make_scene(data,fc,base,prices,d,r,source,count=7):
    off=r*36;ids=np.arange(max(1,d-count),d)
    point=np.stack([base[d,0,off:],base[d,1,off:] if source is None else fc[d,source,off:],prices[d,r,off:]])
    if not len(ids):scene=point[None].copy()
    else:
        past=base[ids,:,off:].copy();past[:,2]=prices[ids,r,off:]
        if source is not None:past[:,1]=fc[ids,source,off:]
        scene=point+data[ids,:,off:]-past
    scene[:,:2]=np.maximum(scene[:,:2],0);scene[:,2]=np.maximum(scene[:,2],.001)
    assert np.isfinite(scene).all()
    return scene,point

def plan(scene,initial,kappa,original=None,eta=.9,hard=False,adj_from=None,adj_coef=SHORT_ADJ):
    """0时共享承诺LP；original给出时是节点调整LP（A>=G0，增购按1.5p）。

    adj_from把"节点后可调整"的时段缺额从5p应急改按adj_coef的增购价计价。0时若
    一律按5p给缺额定阶，等价于把承诺抬到80%分位，而合同下增购只需1.5p、最优分位
    是1/1.5≈67%，于是系统性多承诺。缺额系数与增购成本必须同价，否则0时计划与节点
    实际面临的价格不一致。adj_from=None表示全天不可调整（Q4-2及only0对照）。
    """
    s,_,n=scene.shape; price=scene[:,2]; p=price.mean(0)
    g,c,b,e=[np.arange(k*n,(k+1)*n) for k in range(4)]
    q=np.arange(4*n,(4+s)*n).reshape(s,n);h=(4+s)*n;nv=h+1
    coef=np.full(n,5.) if adj_from is None else np.where(np.arange(n)>=adj_from,adj_coef,5.)
    obj=np.zeros(nv);obj[g]=(1.5 if original is not None else 1.)*p
    obj[q]=coef[None,:]*price/s;obj[h]=kappa;obj[c]=obj[b]=1e-8
    bounds=[(0,None)]*nv
    for t in range(n):
        bounds[c[t]]=bounds[b[t]]=(0,POWER/1000);bounds[e[t]]=(EMIN/1000,EMAX/1000)
        if original is not None:bounds[g[t]]=(original[t]/1000,None)
    if hard:
        bounds[e[-1]]=(6.,6.)
        for t in range(max(0,n-7),n):bounds[g[t]]=(max(POWER/1000,bounds[g[t]][0]),None)
    eq=lil_matrix((n,nv));rhs=np.zeros(n)
    for t in range(n):
        eq[t,e[t]]=1;eq[t,c[t]]=-eta;eq[t,b[t]]=1/eta
        if t:eq[t,e[t-1]]=-1
        else:rhs[t]=initial/1000
    mat=lil_matrix((s*n+n+1,nv));ub=np.zeros(s*n+n+1)
    for j in range(s):
        for t in range(n):
            i=j*n+t;mat[i,g[t]]=-1;mat[i,b[t]]=-1;mat[i,c[t]]=1;mat[i,q[j,t]]=-1
            ub[i]=(scene[j,1,t]-scene[j,0,t])/1000
    for t in range(n):
        mat[s*n+t,c[t]]=1;mat[s*n+t,g[t]]=-1;ub[s*n+t]=scene[:,1,t].min()/1000
    mat[-1,e[-1]]=-1;mat[-1,h]=-1;ub[-1]=-6.
    res=linprog(obj,A_ub=mat.tocsr(),b_ub=ub,A_eq=eq.tocsr(),b_eq=rhs,bounds=bounds,method='highs',
                options={'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    if not res.success:raise RuntimeError(res.message)
    x=res.x*1000;delta=np.minimum(x[c],x[b]/eta**2)
    ch=np.maximum(0,x[c]-delta);dis=np.maximum(0,x[b]-eta**2*delta);grid=np.maximum(x[g],0)
    short=np.maximum(scene[:,0]+ch-grid-scene[:,1]-dis,0)
    economic=float(p@grid*(1.5 if original is not None else 1.)+(coef[None,:]*price*short).sum()/s+kappa*max(6000-x[e[-1]],0))
    if original is not None:economic-=float(1.5*p@original)
    assert np.max(abs(np.r_[initial,x[e][:-1]]+eta*ch-dis/eta-x[e]))<1e-4
    return grid,x[e],economic

def value_table(scene,grid,kappa,eta=.9,step=100.,hard=False):
    states=np.arange(EMIN,EMAX+step/2,step);n=len(grid)
    delta=states[None,:]-states[:,None];c=np.maximum(delta,0)/eta;b=np.maximum(-delta,0)*eta
    valid=(c<=POWER+1e-8)&(b<=POWER+1e-8);bus=c-b
    values=np.empty((n+1,len(states)));values[-1]=kappa*np.maximum(6000-states,0)
    if hard:values[-1]=np.where(abs(states-6000)<1e-8,0,np.inf)
    for t in range(n-1,-1,-1):
        net=scene[:,0,t]-scene[:,1,t]-grid[t]
        cost=(np.maximum(net[:,None,None]+bus,0)*5*scene[:,2,t,None,None]).mean(0)
        allowed=valid&(c<=grid[t]+scene[:,1,t].min()+1e-8)
        costs=cost+values[t+1];costs[~allowed]=np.inf;values[t]=costs.min(axis=1)
    return states,values

def action(price,grid,load,pv,energy,states,future,eta=.9):
    lo=max(EMIN,energy-POWER/eta);hi=min(EMAX,energy+eta*min(POWER,grid+pv));net=load-pv-grid
    balanced=energy-net/eta if net>=0 else energy-net*eta
    candidates=np.unique(np.r_[states[(states>=lo)&(states<=hi)],lo,hi,np.clip(energy,lo,hi),np.clip(balanced,lo,hi)])
    change=candidates-energy;c=np.maximum(change,0)/eta;b=np.maximum(-change,0)*eta
    emergency=np.maximum(net+c-b,0);cost=5*price*emergency+np.interp(candidates,states,future)
    finite=np.isfinite(cost)
    if not finite.any():raise RuntimeError('No feasible real-time action')
    ids=np.flatnonzero(finite&(cost<=cost[finite].min()+1e-7));j=ids[np.argmin((c+b)[ids])]
    return candidates[j],c[j],b[j],emergency[j],max(grid+pv+b[j]+emergency[j]-load-c[j],0)

def baseline(scene):
    net=scene[:,0]-scene[:,1];price=scene[:,2];idx=np.argsort(net,axis=0)
    x=np.take_along_axis(net,idx,axis=0);weights=np.take_along_axis(price,idx,axis=0)
    row=np.argmax(weights.cumsum(0)>=.8*weights.sum(0),axis=0)
    return np.maximum(0,x[row,np.arange(net.shape[1])])

def simulate_day(data,fc,base,prices,d,initial,branch=2,policy='all',eta=.9,step=100.,hard=False,short_adj=SHORT_ADJ):
    energy=initial;original=np.zeros(T);active=np.zeros(T);reference=np.full(T,initial)
    point_used=np.zeros((3,T));decision=np.zeros(T,int);source_used=np.full(T,-1,int);versions=[];rows=[]
    if d==0:
        for t in range(T):rows.append([energy,energy,0.,0.,max(data[d,0,t]-data[d,1,t],0),max(data[d,1,t]-data[d,0,t],0)])
        basegrid=np.zeros(T)
    else:
        kappa=base[d,2,0]/eta
        # 分支3首个调整节点在t=36；only0对照不做节点调整，故仍按全天5p
        adjustable=36 if (branch==3 and policy!='only0') else None
        for t in range(T):
            if t%36==0:
                r=t//36
                source=None if branch==2 else (0 if policy in ('only0','stale_all') else (2 if policy=='stale18' and r==3 else r))
                scene,point=make_scene(data,fc,base,prices,d,r,source)
                if t==0:
                    grid,ref,obj=plan(scene,energy,kappa,eta=eta,hard=hard,adj_from=adjustable,adj_coef=short_adj)
                    original=grid.copy();active[:]=grid;reference[:]=ref;basegrid=baseline(scene)
                elif branch==3 and policy!='only0':
                    grid,ref,obj=plan(scene,energy,kappa,original[t:],eta,hard)
                    active[t:]=grid;reference[t:]=ref
                states,values=value_table(scene,active[t:],kappa,eta,step,hard)
                point_used[:,t:]=point;start=t;source_used[t:]=-1 if source is None else source*6
                if t==0 or (branch==3 and policy!='only0'):
                    decision[t:]=r*6
                    for k in range(t,T):versions.append([str(DATES[d].date()),r*6,-1 if source is None else source*6,k+1,original[k],active[k]])
            old=energy
            energy,c,b,e,w=action(data[d,2,t],active[t],data[d,0,t],data[d,1,t],energy,states,values[t-start+1],eta)
            rows.append([old,energy,c,b,e,w])
    df=pd.DataFrame(rows,columns=['storage_start_kwh','storage_end_kwh','charge_kwh','discharge_kwh','emergency_kwh','unused_kwh'])
    df.insert(0,'date',str(DATES[d].date()));df.insert(1,'slot',np.arange(1,145));df['interval']=[interval(t) for t in range(T)]
    for i,col in enumerate(['load_kwh','pv_kwh','price_yuan_per_kwh']):df[col]=data[d,i]
    for i,col in enumerate(['forecast_load_kwh','forecast_pv_kwh','forecast_price']):df[col]=point_used[i]
    df['original_grid_kwh']=original;df['adjusted_grid_kwh']=active;df['reference_storage_kwh']=reference
    df['decision_hour']=decision;df['forecast_release_hour']=source_used
    df['increase_kwh']=np.maximum(active-original,0);df['decrease_kwh']=np.maximum(original-active,0)
    p=data[d,2];df['plan_cost_yuan']=p*original;df['increase_cost_yuan']=1.5*p*df.increase_kwh
    df['decrease_cost_yuan']=.5*p*df.decrease_kwh;df['emergency_cost_yuan']=5*p*df.emergency_kwh
    df['total_cost_yuan']=df[['plan_cost_yuan','increase_cost_yuan','decrease_cost_yuan','emergency_cost_yuan']].sum(axis=1)
    df['baseline_cost_yuan']=p*(basegrid+5*np.maximum(data[d,0]-data[d,1]-basegrid,0))
    df['history_latest_date']='' if d==0 else str(DATES[d-1].date())
    daily={'date':str(DATES[d].date()),'initial_kwh':initial,'final_kwh':energy,'branch':branch,'policy':policy}
    for col in ['original_grid_kwh','adjusted_grid_kwh','increase_kwh','decrease_kwh','emergency_kwh','charge_kwh','discharge_kwh','unused_kwh','plan_cost_yuan','increase_cost_yuan','decrease_cost_yuan','emergency_cost_yuan','total_cost_yuan','baseline_cost_yuan']:
        daily[col]=float(df[col].sum())
    daily['emergency_slots']=int((df.emergency_kwh>1e-6).sum())
    return df,daily,versions

def validate(df,eta=.9):
    measures={
        'balance':float(abs(df.adjusted_grid_kwh+df.pv_kwh+df.discharge_kwh+df.emergency_kwh-df.load_kwh-df.charge_kwh-df.unused_kwh).max()),
        'dynamics':float(abs(df.storage_start_kwh+eta*df.charge_kwh-df.discharge_kwh/eta-df.storage_end_kwh).max()),
        'continuity':float(abs(df.storage_start_kwh.to_numpy()[1:]-df.storage_end_kwh.to_numpy()[:-1]).max()),
        'mutual_exclusion':float(np.minimum(df.charge_kwh,df.discharge_kwh).max()),
        'source':float(np.maximum(df.charge_kwh-df.adjusted_grid_kwh-df.pv_kwh,0).max()),
        'power':float(max(0,df[['charge_kwh','discharge_kwh']].max().max()*6-5000)),
        'account':float(abs(df.total_cost_yuan-df.price_yuan_per_kwh*(df.original_grid_kwh+1.5*df.increase_kwh+.5*df.decrease_kwh+5*df.emergency_kwh)).max())}
    assert max(measures.values())<1e-5,measures
    assert df[['storage_start_kwh','storage_end_kwh']].min().min()>=1200-1e-5
    assert df[['storage_start_kwh','storage_end_kwh']].max().max()<=10800+1e-5
    assert df[['original_grid_kwh','adjusted_grid_kwh','charge_kwh','discharge_kwh','unused_kwh','emergency_kwh']].min().min()>=-1e-5
    assert np.isfinite(df.select_dtypes('number')).all().all()
    assert (df.decrease_kwh<1e-5).all()
    return measures

def episodes(df):
    out=[]
    for date,day in df.groupby('date',sort=False):
        e=day.emergency_kwh.to_numpy();start=None
        for t in range(145):
            active=t<144 and e[t]>1e-6
            if active and start is None:start=t
            if not active and start is not None:
                out.append([date,label(start)+'-'+label(t),float(e[start:t].sum())]);start=None
    return pd.DataFrame(out,columns=['date','interval','emergency_kwh'])
