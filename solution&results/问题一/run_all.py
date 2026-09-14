"""运行问题一求解、绘图与技能复现清单：python run_all.py。"""
from pathlib import Path
import subprocess,sys,argparse
from solve_q1 import INPUT
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=INPUT)
args=parser.parse_args()
subprocess.run([sys.executable,str(root/'solve_q1.py'),'--input',str(args.input),'--output',str(root)],check=True)
subprocess.run([sys.executable,str(root/'plot_q1.py')],check=True)
subprocess.run([sys.executable,str(root/'utils/repro_manifest.py'),'--project-root',str(root),
               '--input',str(args.input/'附件1.xlsx'),'--input',str(args.input/'附件5/result1.xlsx'),
               '--seed','0','--parameters','{"eta_c":0.9,"eta_d":0.9,"delta_h":0.16666666666666666,"initial_kwh":6000,"deterministic":true}',
               '--command',f'python run_all.py --input "{args.input}"',
               '--package','numpy','--package','pandas','--package','scipy','--package','openpyxl','--package','matplotlib',
               '--overwrite'],check=True)
