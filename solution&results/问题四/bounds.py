"""Perfect-information LOWER bounds with the realized initial/final inventory."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
ROOT=Path(__file__).resolve().parent

def bound(df):
    p=df.price_yuan_per_kwh.to_numpy();load=df.load_kwh.to_numpy();pv=df.pv_kwh.to_numpy();n=len(p)
    g,c,b,e=[np.arange(i*n,(i+1)*n) for i in range(4)];obj=np.zeros(4*n);obj[g]=p
    rows=np.r_[np.arange(n),np.arange(1,n),np.arange(n),np.arange(n)]
    cols=np.r_[e,e[:-1],c,b];vals=np.r_[np.ones(n),-np.ones(n-1),np.full(n,-.9),np.full(n,1/.9)]
    ae=coo_matrix((vals,(rows,cols)),shape=(n,4*n)).tocsr();be=np.zeros(n);be[0]=df.storage_start_kwh.iloc[0]
    rows=np.tile(np.arange(n),4);cols=np.r_[g,c,b,e];vals=np.r_[-np.ones(n),np.ones(n),-np.ones(n),np.zeros(n)]
    rows=np.r_[rows,np.arange(n,2*n),np.arange(n,2*n)];cols=np.r_[cols,c,g];vals=np.r_[vals,np.ones(n),-np.ones(n)]
    au=coo_matrix((vals,(rows,cols)),shape=(2*n,4*n)).tocsr();bu=np.r_[pv-load,pv]
    limits=[(0,None)]*n+[(0,5000/6)]*(2*n)+[(1200,10800)]*n
    limits[-1]=(df.storage_end_kwh.iloc[-1],df.storage_end_kwh.iloc[-1])
    result=linprog(obj,A_ub=au,b_ub=bu,A_eq=ae,b_eq=be,bounds=limits,method='highs')
    if not result.success:raise RuntimeError(result.message)
    error=max(float(abs(ae@result.x-be).max()),float(np.maximum(au@result.x-bu,0).max()))
    assert error<1e-4
    main=float(df.total_cost_yuan.sum());lower=float(result.fun);assert lower<=main+1e-4
    return dict(lower_bound=lower,main_cost=main,gap_yuan=main-lower,gap_over_lower_pct=(main/lower-1)*100,initial=float(be[0]),final=float(result.x[e[-1]]),residual=error,status='optimal')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args();r=args.output/'results'
    results={}
    for branch in [2,3]:
        results[f'q4{branch}']=bound(pd.read_csv(r/f'q4{branch}_dispatch.csv'));print(branch,results[f'q4{branch}'],flush=True)
    (r/'bounds.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
