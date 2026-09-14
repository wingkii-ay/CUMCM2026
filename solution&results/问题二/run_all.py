"""唯一复现入口：求解、核验、答案、图表、复现清单。"""
from pathlib import Path
import argparse,json,subprocess,sys
from solve_q2 import INPUT, LOOKBACK_WEEKS

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=INPUT)
args=parser.parse_args()
def run(script,*extra):
    subprocess.run([sys.executable,str(root/script),*map(str,extra)],check=True,cwd=root)
run('solve_q2.py','--input',args.input,'--output',root,'--full')
run('verify_outputs.py')
run('write_answer.py')
run('plot_q2.py')
params={'lookback_weeks':LOOKBACK_WEEKS,'scenario_count':7,
        'state_grid_kwh':100,'eta_charge':.9,'eta_discharge':.9,'delta_h':1/6,
        'january_initial_kwh':6000,'forecast_cutoff':'previous complete day','kappa':'p[0]/eta',
        'evaluation':'2025-02-01 through 2025-12-31','feedback':'marginal-scenario approximate value table',
        'deterministic':True}
run('utils/repro_manifest.py','--project-root',root,
    '--input',args.input/'附件1.xlsx','--input',args.input/'附件2.xlsx','--input',args.input/'附件5/result2.xlsx',
    '--seed',0,'--parameters',json.dumps(params,ensure_ascii=False),
    '--command',f'python run_all.py --input "{args.input}"',
    '--package','numpy','--package','pandas','--package','scipy','--package','openpyxl','--package','matplotlib',
    '--overwrite')
