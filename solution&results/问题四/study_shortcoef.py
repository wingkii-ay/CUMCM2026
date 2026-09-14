"""Q4-3"节点后缺额边际代价"取值的稳健性检验。

0时计划里给"节点后可调整时段"的缺额定价，原先一律按5倍应急。合同下这些时段
实际可按1.5倍增购（见 core.plan 的 adj_coef），5倍等价于把承诺抬到80%分位。
本脚本把该系数从1.5扫到5，量出它对全年实际费用的影响，用于判断主结果对该建模
选择的敏感程度。

注意：本脚本只做稳健性诊断，**主结果固定取 adj_coef=1.5**（合同增购价，
与问题三 plan_day_2s 对可调整时段的定价一致）。按回测费用挑系数属于事后择优，
不得据此改主结果。公平边界协议（2月1日与12月31日均固定6000kWh）与 solve.py 一致。

Run:  python study_shortcoef.py --input "<C题附件目录>"
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, json
import numpy as np
import pandas as pd
from core import INPUT, DATES, T, read_data, predictions, simulate_day, validate

ROOT = Path(__file__).resolve().parent
COEFS = [5.0, 3.0, 2.0, 1.5]
POLICIES = ['all', 'stale_all']


def run_fair(payload):
    data, fc, base, prices, branch, policy, coef = payload
    days = []
    energy = 6000.
    for d in range(31, 365):
        df, day, _ = simulate_day(data, fc, base, prices, d, energy, branch, policy,
                                  hard=(d == 364), short_adj=coef)
        validate(df)
        energy = day['final_kwh']
        days.append(day)
    daily = pd.DataFrame(days)
    evaluation = daily[daily.date >= '2025-02-01']
    return dict(branch=branch, policy=policy, short_adj=coef,
                cost=float(evaluation.total_cost_yuan.sum()),
                plan_cost=float(evaluation.plan_cost_yuan.sum()),
                increase_cost=float(evaluation.increase_cost_yuan.sum()),
                emergency_cost=float(evaluation.emergency_cost_yuan.sum()),
                emergency_kwh=float(evaluation.emergency_kwh.sum()),
                grid_kwh=float(evaluation.original_grid_kwh.sum()),
                adjusted_kwh=float(evaluation.adjusted_grid_kwh.sum()),
                unused_kwh=float(evaluation.unused_kwh.sum()),
                final_kwh=float(daily.final_kwh.iloc[-1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=INPUT)
    p.add_argument('--output', type=Path, default=ROOT / 'results')
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    data, fc = read_data(a.input)
    base, prices = predictions(data)
    jobs = [(data, fc, base, prices, 3, pol, c) for pol in POLICIES for c in COEFS]
    # Q4-2 has no post-node recourse: run once, it must not move with the coefficient
    jobs.append((data, fc, base, prices, 2, 'all', 1.5))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(run_fair, jobs):
            rows.append(r)
            print(f"Q4-{r['branch']} {r['policy']:10s} coef={r['short_adj']:.1f} "
                  f"cost={r['cost']:,.0f} emergency={r['emergency_kwh']:,.0f} kWh", flush=True)
    df = pd.DataFrame(rows).sort_values(['branch', 'policy', 'short_adj'], ascending=[True, True, False])
    a.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output / 'short_coef_sensitivity.csv', index=False, encoding='utf-8-sig')
    base31 = df[(df.branch == 3) & (df.policy == 'all') & (df.short_adj == 1.5)].cost.iloc[0]
    out = dict(note='主结果取 adj_coef=1.5（合同增购价）；本表只作稳健性诊断，不用于选参。',
               reference_q42_fair=float(df[(df.branch == 2)].cost.iloc[0]),
               rows=df.to_dict('records'),
               max_range_pct_all=float(100 * (df[(df.branch == 3) & (df.policy == 'all')].cost.max() / base31 - 1)))
    (a.output / 'short_coef_sensitivity.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
