"""量化 solve_q3.py:42 的预报截断代价：每个发布时刻的 24h 窗口有多少被丢弃。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import openpyxl
from battery_core import INPUT

T = 144
w = openpyxl.load_workbook(INPUT / '附件3.xlsx', read_only=True, data_only=True)
rows = list(w.active.values); w.close()
h = np.empty((365, 4, 24))
for k, row in enumerate(rows[1:]):
    d, r = divmod(k, 4)
    h[d, r] = np.asarray(row[2:], float)

per = {}
for r in range(4):
    keep = T - r * 36
    tail = h[:, r, keep // 6:]           # 被丢弃的窗口尾部
    per[r] = tail.sum()
    print(f'{r*6:02d}时发布: 保留 {keep//6:>2}h 丢弃 {24-keep//6:>2}h  丢弃预报电量 {tail.sum():>13,.1f} kWh')
print(f'合计丢弃 {sum(per.values()):,.1f} kWh，日均 {sum(per.values())/365:,.1f} kWh')

idx = np.arange(31, 365)                 # 评价期 2/1 起
print(f'评价期丢弃 {sum(h[idx, r, (T-r*36)//6:].sum() for r in range(4)):,.1f} kWh')
print(f'2025-03-20 18时窗口尾部(次日00:00-18:00) {h[78,3,6:].sum():,.1f} kWh')
print(f'2025-03-20 全日实际光伏 {sum(h[78,r,(T-r*36)//6:].sum() for r in range(4)):,.1f} kWh(仅作对照)')
