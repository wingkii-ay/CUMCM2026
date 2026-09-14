"""一次性稳健性检验：预测回看深度对全年总费用的影响。

结果写入 results/lookback_robustness.json，供 write_answer.py 引用。
该检验不参与主参数反选，只用于说明主结果对回看深度不敏感。
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solve_q2 as S

ROOT = Path(__file__).resolve().parent
WEEKS = [4, 8, 12, 16, 24]


def mixed_predict(history, date, window=None):
    """改动前的做法：只区分工作日/周末，最近 7 天取均值。"""
    n = len(history)
    ids = np.arange(max(0, n - 7), n)
    same = ids[(S.DATES[ids].dayofweek >= 5) == (date.dayofweek >= 5)]
    ids = same if len(same) >= 2 else ids
    return history[ids].mean(axis=0)


def make_split(weeks):
    """按精确星期几匹配，回看 weeks 周。"""
    def predict(history, date, window=None):
        n = len(history)
        ids = np.arange(n - min(n, weeks * 7), n)
        same = ids[S.DATES[ids].dayofweek == date.dayofweek]
        if len(same) == 0: same = ids
        return history[same].mean(axis=0)
    return predict


def run(fn):
    S.predict = fn
    _, daily, _ = S.run_period(p, data, days=365, lookback_weeks=S.LOOKBACK_WEEKS)
    off = daily[daily.date >= '2025-02-01']
    return float(off.total_cost_yuan.sum())


p, data = S.read_data(S.INPUT)
print('数据载入完成', flush=True)

base = run(mixed_predict)
print(f'混合预测基线: {base:,.2f}', flush=True)

study = []
for w in WEEKS:
    total = run(make_split(w))
    study.append({'weeks': w, 'total_cost_yuan': total,
                  'saving_yuan': base - total, 'saving_pct': (base - total) / base * 100})
    print(f'{w:2d}周: {total:>14,.2f}  省 {base - total:>12,.0f}'
          f' ({study[-1]["saving_pct"]:5.2f}%)', flush=True)

spread = max(r['saving_pct'] for r in study) - min(r['saving_pct'] for r in study)
out = {'baseline_mixed_total_cost_yuan': base,
       'baseline_description': '工作日/周末混合、7天窗口',
       'production_weeks': S.LOOKBACK_WEEKS, 'study': study,
       'saving_pct_spread': spread}
(ROOT / 'results' / 'lookback_robustness.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'\n回看深度 {WEEKS[0]}—{WEEKS[-1]} 周内节省比例极差 {spread:.2f} 个百分点')
print('已写入 results/lookback_robustness.json')
