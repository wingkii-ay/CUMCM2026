"""Independent ex-ante causality / data-leakage audit for Q4.

The Q4 deliverable claims decisions at time t use only information available at
t.  This script attacks that claim directly instead of trusting solve.py's own
smoke test: it corrupts the *future* in several ways and demands that every
already-taken decision stays bit-identical.

Run:  python audit_leakage.py --input "<C题附件目录>"
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from core import INPUT, DATES, T, read_data, predictions, simulate_day, make_scene

TRACK = ['storage_end_kwh', 'charge_kwh', 'discharge_kwh', 'emergency_kwh',
         'unused_kwh', 'adjusted_grid_kwh', 'original_grid_kwh']


def maxdiff(a, b, cols):
    return float(np.max(np.abs(a[cols].to_numpy(float) - b[cols].to_numpy(float)))) if len(a) else 0.0


def corrupt(data, d, cut, rng, mode):
    """Return a copy of data with slot >= cut of day d and all later days ruined."""
    alt = data.copy()
    shape = alt[d, :, cut:].shape
    if mode == 'scale':                      # multiplicative blow-up (keeps ordering)
        alt[d, :, cut:] *= 3.7
    elif mode == 'zero':                     # hard zero
        alt[d, :, cut:] = 0.0
    elif mode == 'noise':                    # i.i.d. uniform, no structure preserved
        alt[d, :, cut:] = rng.uniform(0.0, 60.0, size=shape)
    elif mode == 'shuffle':                  # permute time, breaks any trending signal
        alt[d][:, cut:] = alt[d][:, cut:][:, rng.permutation(shape[1])]
    else:
        raise ValueError(mode)
    fut = alt[d + 1:].shape
    if fut[0]:
        alt[d + 1:] = rng.uniform(0.0, 60.0, size=fut)
    return alt


def run(data, fc, base, prices, days, branch, policy):
    """Perturbation battery.  Returns list of result rows."""
    rng = np.random.default_rng(20260913)
    rows = []
    for d in days:
        energy = 6000.0            # inventory is exogenous to the perturbation
        ref, _, _ = simulate_day(data, fc, base, prices, d, energy, branch, policy)
        # --- control: identical inputs must reproduce identically
        ctl, _, _ = simulate_day(data, fc, base, prices, d, energy, branch, policy)
        rows.append(dict(test='control', branch=branch, policy=policy, day=d,
                         cut=None, mode='-', leak=maxdiff(ref, ctl, TRACK)))
        for cut in (36, 72, 108):          # decision nodes 6h / 12h / 18h
            for mode in ('scale', 'zero', 'noise', 'shuffle'):
                alt = corrupt(data, d, cut, rng, mode)
                ab, ap = predictions(alt)
                # the forecast the simulation would really have had, too:
                # predictions() must itself be causal; checked separately below.
                got, _, _ = simulate_day(alt, fc, ab, ap, d, energy, branch, policy)
                leak = maxdiff(ref.iloc[:cut], got.iloc[:cut], TRACK)
                # the 00:00 plan is a decision taken at t=0: the whole day of G must be frozen
                leak = max(leak, float(np.max(np.abs(ref.original_grid_kwh - got.original_grid_kwh))))
                rows.append(dict(test='future-corruption', branch=branch, policy=policy,
                                 day=d, cut=cut, mode=mode, leak=leak))
        # --- future days only: the whole of day d must be untouched
        alt = data.copy()
        alt[d + 1:] = rng.uniform(0.0, 60.0, size=alt[d + 1:].shape)
        ab, ap = predictions(alt)
        got, _, _ = simulate_day(alt, fc, ab, ap, d, energy, branch, policy)
        rows.append(dict(test='later-days-only', branch=branch, policy=policy, day=d,
                         cut=T, mode='noise', leak=maxdiff(ref, got, TRACK)))
    return rows


def forecast_audit(data, days):
    """predictions() must not read the future either."""
    rng = np.random.default_rng(11)
    out = []
    base, prices = predictions(data)
    for d in days:
        alt = data.copy()
        alt[d] = rng.uniform(0.0, 60.0, size=data[d].shape)          # ruin all of day d
        alt[d + 1:] = rng.uniform(0.0, 60.0, size=data[d + 1:].shape)  # and all later days
        ab, ap = predictions(alt)
        # load/pv history mean for day d must be untouched
        out.append(dict(check='base_load_pv', day=d,
                        error=float(np.max(np.abs(ab[d, :2] - base[d, :2])))))
        # price forecast at release r may only use day-d slots strictly before 36r
        for r in range(4):
            off = 36 * r
            out.append(dict(check=f'price_r{r}_prefix', day=d,
                            error=float(np.max(np.abs(ap[d, r, :off] - prices[d, r, :off]))) if off else 0.0))
    # price forecast at release r must depend ONLY on slots < 36r of day d
    d = days[0]
    for r in range(1, 4):
        off = 36 * r
        alt = data.copy()
        alt[d, 2, off:] = rng.uniform(0.0, 5.0, size=T - off)
        ab, ap = predictions(alt)
        out.append(dict(check=f'price_r{r}_independent_of_future', day=d,
                        error=float(np.max(np.abs(ap[d, r] - prices[d, r])))))
    return out


def scene_audit(data, fc, base, prices, days):
    """make_scene must draw residuals only from completed days < d, and its point
    forecast only from the already-observed prefix of day d (slots < 36r)."""
    rng = np.random.default_rng(5)
    out = []
    for d in days:
        for r in range(4):
            off = 36 * r
            # ruin day d from the release instant onward, plus every later day.
            # slots < off stay intact: they are legitimately observable at release r.
            alt = data.copy()
            if off < T:
                alt[d][:, off:] = rng.uniform(0.0, 60.0, size=(3, T - off))
            if d + 1 < len(alt):
                alt[d + 1:] = rng.uniform(0.0, 60.0, size=alt[d + 1:].shape)
            ab, ap = predictions(alt)
            for src in (None, 0, r):
                sc, pt = make_scene(data, fc, base, prices, d, r, src)
                sc2, pt2 = make_scene(alt, fc, ab, ap, d, r, src)
                out.append(dict(check='scene', day=d, release=r * 6,
                                source=-1 if src is None else src,
                                error=float(np.max(np.abs(sc - sc2))),
                                point_error=float(np.max(np.abs(pt - pt2)))))
    return out


def chain_audit(data, fc, base, prices, cut, branch, policy):
    """Whole-year carry-over test.

    The per-day battery pins the initial inventory, so a leak that travels
    through the state of charge would slip past it.  Here the state of charge
    is carried exactly as in the real pipeline: run the full year on the real
    data, then again with every day >= cut replaced by noise.  Days < cut must
    come out bit-identical, inventory included.
    """
    rng = np.random.default_rng(4242)
    alt = data.copy()
    alt[cut:] = rng.uniform(0.0, 60.0, size=alt[cut:].shape)
    ab, ap = predictions(alt)
    out = []
    for tag, (dd, bb, pp) in [('real', (data, base, prices)), ('ruined-future', (alt, ab, ap))]:
        rows = []
        energy = 6000.0
        for d in range(365):
            df, day, _ = simulate_day(dd, fc, bb, pp, d, energy, branch, policy)
            energy = day['final_kwh']
            rows.append(df)
        out.append(pd.concat(rows, ignore_index=True))
    real, got = out
    head_r = real[real.date < str(DATES[cut].date())].reset_index(drop=True)
    head_g = got[got.date < str(DATES[cut].date())].reset_index(drop=True)
    assert len(head_r) == len(head_g) == cut * 144
    return dict(check='whole-year-carryover', branch=branch, policy=policy,
                cut_day=cut, days_compared=cut,
                leak=maxdiff(head_r, head_g, TRACK),
                final_soc_real=float(real.storage_end_kwh.iloc[-1]),
                final_soc_ruined=float(got.storage_end_kwh.iloc[-1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=INPUT)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    p.add_argument('--days', type=int, default=12)
    p.add_argument('--chain', action='store_true')
    p.add_argument('--chain-cut', type=int, default=334)
    a = p.parse_args()
    data, fc = read_data(a.input)
    base, prices = predictions(data)
    days = [int(v) for v in np.linspace(32, 363, a.days).round()]   # evaluation period only

    if a.chain:
        chains = [chain_audit(data, fc, base, prices, a.chain_cut, b, 'all') for b in (2, 3)]
        bad = [c for c in chains if c['leak'] > 1e-9]
        report = dict(mode='chain', cut_day=a.chain_cut, chains=chains,
                      status='PASS' if not bad else 'FAIL')
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / 'leakage_chain.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    rows = []
    for branch, policy in [(2, 'all'), (3, 'all'), (3, 'only0'), (3, 'stale_all'), (3, 'stale18')]:
        rows += run(data, fc, base, prices, days, branch, policy)
    sim = pd.DataFrame(rows)
    fc_audit = forecast_audit(data, days)
    sc_audit = scene_audit(data, fc, base, prices, days)

    leaks = sim[(sim.test != 'control') & (sim.leak > 1e-9)]
    bad_fc = [r for r in fc_audit if r['error'] > 1e-12]
    bad_sc = [r for r in sc_audit if max(r['error'], r['point_error']) > 1e-12]
    report = dict(
        days_tested=days, configs=sim.groupby(['branch', 'policy']).ngroups,
        perturbation_runs=int(len(sim) - len(sim[sim.test == 'control'])),
        worst_control_leak=float(sim[sim.test == 'control'].leak.max()),
        leak_violations=leaks.to_dict('records'),
        forecast_violations=bad_fc, scene_violations=bad_sc,
        status='PASS' if not len(leaks) and not bad_fc and not bad_sc else 'FAIL')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'leakage_audit.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('leak_violations', 'forecast_violations', 'scene_violations')},
                     ensure_ascii=False, indent=2))
    if len(leaks):
        print('\nLEAKS:\n', leaks.head(30).to_string())
    if bad_fc:
        print('\nFORECAST LEAKS:\n', pd.DataFrame(bad_fc).head(20).to_string())
    if bad_sc:
        print('\nSCENE LEAKS:\n', pd.DataFrame(bad_sc).head(20).to_string())


if __name__ == '__main__':
    main()
