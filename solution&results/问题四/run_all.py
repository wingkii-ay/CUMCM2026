"""Rebuild Q4 outputs. --finish-only uses existing solved ledgers explicitly."""
from pathlib import Path
import argparse,subprocess,sys,json,time
from core import INPUT,PARAMS
ROOT=Path(__file__).resolve().parent

def skill_root():
    for p in [Path.home()/'.claude/skills/math-modeling',Path.home()/'.codex/skills/math-modeling',
              Path(r'C:\Users\NGALAM\.codex\skills\math-modeling')]:
        if (p/'tools').is_dir():return p
    raise RuntimeError('未找到 math-modeling skill 目录，请在 skill_root() 中补上本机路径')
SKILL=skill_root()

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=INPUT);p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--finish-only',action='store_true');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);r=out/'results';r.mkdir(exist_ok=True);steps=[]
    def run(script,*args,log=None):
        started=time.time();cmd=[sys.executable,str(script),*map(str,args)]
        if log:
            with (r/log).open('w',encoding='utf-8') as f:proc=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        else:proc=subprocess.run(cmd,cwd=ROOT)
        steps.append(dict(script=str(script),exit_code=proc.returncode,seconds=time.time()-started))
        if proc.returncode:raise RuntimeError(f'{script} exited {proc.returncode}; log={log}')
    run(SKILL/'references/roles/编程手/scripts/check_env.py','--features','data','visualization','optimization','excel',log='environment.json')
    run(ROOT/'solve.py','--input',a.input,'--output',r/'smoke','--smoke',log='smoke.log')
    if not a.finish_only:run(ROOT/'solve.py','--input',a.input,'--output',out,'--workers',a.workers,log='solve.log')
    run(ROOT/'verify.py','--input',a.input,'--output',out,log='verify.log')
    run(ROOT/'bounds.py','--output',out,log='bounds.log')
    run(ROOT/'study_shortcoef.py','--input',a.input,'--output',r,'--workers',a.workers,log='study_shortcoef.log')
    for b in [2,3]:run(SKILL/'tools/figure/scripts/profile_data.py',r/f'q4{b}_dispatch.csv','--json',log=f'profile_q4{b}.json')
    run(ROOT/'present.py','--input',a.input,'--output',out,log='present.log')
    run(SKILL/'tools/figure/scripts/check_figure.py',str(out/'figures/*.png'),str(out/'figures/*.svg'),'--width-in','6.3','--height-in','3.8','--strict',log='check_figures.log')
    run(SKILL/'references/roles/编程手/scripts/figure_audit.py',out/'figures','--questions','q1','q2','--strict',log='figure_audit.json')
    inputs=[a.input/'附件2.xlsx',a.input/'附件3.xlsx',a.input/'附件4.xlsx',a.input/'附件5/result4-2.xlsx',a.input/'附件5/result4-3.xlsx']
    args=[]
    for f in inputs:args+=['--input',f]
    params={**PARAMS,'settlement':'floating delivery-slot actual prices, original nonrefundable, final net adjustment',
            'development_backtest':True,'branches':['Q4-2','Q4-3'],'evaluation_days':334}
    command=f'python run_all.py --input "{a.input}"'
    run(ROOT/'utils/repro_manifest.py','--project-root',out,*args,'--seed',2026,'--parameters',json.dumps(params,ensure_ascii=False),'--command',command,
        '--package','numpy','--package','pandas','--package','scipy','--package','openpyxl','--package','matplotlib','--overwrite',log='manifest.log')
    (r/'execution.json').write_text(json.dumps(dict(status='PASS',finish_only=a.finish_only,steps=steps),ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: all execution and figure checks',flush=True)

if __name__=='__main__':main()
