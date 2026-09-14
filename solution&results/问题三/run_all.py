"""唯一复现入口：全年求解、核验、答案、插图和清单。"""
from pathlib import Path
import argparse,json,subprocess,sys
from solve_q3 import INPUT
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=INPUT);parser.add_argument('--workers',type=int,default=2);parser.add_argument('--finish-only',action='store_true',help='仅处理已有求解结果，常规复现不要使用');args=parser.parse_args()
def run(script,*extra):subprocess.run([sys.executable,str(root/script),*map(str,extra)],check=True,cwd=root)
if not args.finish_only:run('solve_q3.py','--input',args.input,'--output',root,'--full','--workers',args.workers)
run('verify_outputs.py');run('write_answer.py');run('plot_q3.py');run('bound_perfect_foresight.py')
params={'load_lookback_weeks':8,'load_match':'exact weekday','scenario_count':7,'state_grid_kwh':100,'eta_charge':.9,'eta_discharge':.9,'delta_h':1/6,'initial_kwh':6000,'release_hours':[0,6,12,18],'hour0_planner':'two-stage stochastic program (shared commitment G, scenario purchase A^s>=G)','hour0_adjust_from_slot':36,'hour0_shortfall_price':'1.5p for t>=36, 5p emergency for t<36','bootstrap_seed':2026,'bootstrap_repeats':2000,'settlement':'original payment plus final net adjustment, no refund','evaluation':'2025-02-01 through 2025-12-31'}
run('utils/repro_manifest.py','--project-root',root,'--input',args.input/'附件1.xlsx','--input',args.input/'附件2.xlsx','--input',args.input/'附件3.xlsx','--input',args.input/'附件5/result3.xlsx','--seed',2026,'--parameters',json.dumps(params,ensure_ascii=False),'--command',f'python run_all.py --input "{args.input}"','--package','numpy','--package','pandas','--package','scipy','--package','openpyxl','--package','matplotlib','--overwrite')
