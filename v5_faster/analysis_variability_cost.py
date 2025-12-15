#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate variability (SRQ2) and cost-performance (SRQ5) metrics.

Input: sweep_agg.csv (Chapter3) OR grid_summary.csv (Chapter4)
Outputs:
  - variability_table.csv (size -> CV metrics, loss variance)
  - cost_pareto.csv (non-dominated points for success_rate & episodes_per_min)
  - textual summary printed to console

Usage:
  python analysis_variability_cost.py --csv ./v5_faster/v5_exp_data/replay_sweep/rb122k/sweep_summary.csv
  python analysis_variability_cost.py --csv ./v5_faster/v5_exp_data/replay_grid/rb_bs_grid/grid_summary.csv --grid
"""
from __future__ import annotations
import csv, math, argparse, statistics, os
from collections import defaultdict

FLOAT_FIELDS = {
    'success_rate','avg_score','avg_loss','episodes_per_min','time_min','update_loss_variance',
    'update_loss_variance_last100','td_mean_last100','td_std_last100'
}
INT_FIELDS = {
    'episodes','successes','total_samples_collected','first_success_episode','learning_steps_to_first_success'
}


def load_rows(path: str):
    rows = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows


def to_num(v):
    if v in (None, '', 'N/A'): return None
    try:
        if '.' in str(v): return float(v)
        return int(v)
    except Exception:
        try:
            return float(v)
        except Exception:
            return None


def group_by(rows, key):
    g = defaultdict(list)
    for r in rows:
        k = r.get(key)
        if k is None: continue
        try:
            g[int(k)].append(r)
        except Exception:
            continue
    return g


def coeff_var(vals):
    vals = [v for v in vals if isinstance(v,(int,float))]
    if len(vals) < 2: return None
    m = statistics.mean(vals)
    if m == 0: return None
    return statistics.stdev(vals) / m


def variability_analysis(rows, is_grid=False):
    # For grid, we aggregate per size across all batch sizes; still answer capacity variability.
    key = 'size'
    grouped = group_by(rows, key)
    out_rows = []
    for size, grp in sorted(grouped.items()):
        sr = [to_num(r.get('success_rate')) for r in grp]
        score = [to_num(r.get('avg_score')) for r in grp]
        loss = [to_num(r.get('avg_loss')) for r in grp]
        ulv = [to_num(r.get('update_loss_variance')) for r in grp]
        ulv_last = [to_num(r.get('update_loss_variance_last100')) for r in grp]
        cv_sr = coeff_var(sr)
        cv_score = coeff_var(score)
        cv_loss = coeff_var(loss)
        mean_ulv = statistics.mean([v for v in ulv if isinstance(v,(int,float))]) if ulv else None
        mean_ulv_last = statistics.mean([v for v in ulv_last if isinstance(v,(int,float))]) if ulv_last else None
        out_rows.append({
            'size': size,
            'runs': len(grp),
            'cv_success_rate': f"{cv_sr:.4f}" if cv_sr is not None else '',
            'cv_avg_score': f"{cv_score:.4f}" if cv_score is not None else '',
            'cv_avg_loss': f"{cv_loss:.4f}" if cv_loss is not None else '',
            'mean_update_loss_variance': f"{mean_ulv:.6f}" if mean_ulv is not None else '',
            'mean_update_loss_variance_last100': f"{mean_ulv_last:.6f}" if mean_ulv_last is not None else ''
        })
    return out_rows


def pareto_front(rows):
    # Maximize success_rate and episodes_per_min simultaneously.
    pts = []
    for r in rows:
        sr = to_num(r.get('success_rate'))
        epm = to_num(r.get('episodes_per_min'))
        size = to_num(r.get('size'))
        if sr is None or epm is None or size is None:
            continue
        pts.append((size, sr, epm))
    frontier = []
    for i,(sz_i, sr_i, epm_i) in enumerate(pts):
        dominated = False
        for j,(sz_j, sr_j, epm_j) in enumerate(pts):
            if i==j: continue
            if (sr_j >= sr_i and epm_j >= epm_i) and (sr_j > sr_i or epm_j > epm_i):
                dominated = True
                break
        if not dominated:
            frontier.append((sz_i, sr_i, epm_i))
    return frontier


def write_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Input sweep or grid summary CSV')
    ap.add_argument('--out_dir', default='./v5_faster/v5_exp_data/analysis', help='Output directory')
    ap.add_argument('--grid', action='store_true', help='Input is grid_summary (contains batch_size column)')
    args = ap.parse_args()

    rows = load_rows(args.csv)
    var_rows = variability_analysis(rows, is_grid=args.grid)
    var_header = ['size','runs','cv_success_rate','cv_avg_score','cv_avg_loss','mean_update_loss_variance','mean_update_loss_variance_last100']
    write_csv(os.path.join(args.out_dir,'variability_table.csv'), var_rows, var_header)

    frontier = pareto_front(rows)
    pareto_rows = [{'size': sz, 'success_rate': sr, 'episodes_per_min': epm} for (sz,sr,epm) in frontier]
    pareto_header = ['size','success_rate','episodes_per_min']
    write_csv(os.path.join(args.out_dir,'cost_pareto.csv'), pareto_rows, pareto_header)

    print('\n[Summary] Variability (lower CV better):')
    for r in var_rows:
        print(r)
    print('\n[Summary] Pareto frontier points:')
    for r in pareto_rows:
        print(r)

if __name__ == '__main__':
    main()
