#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot replay buffer sweep results (simple) for v5_faster.

This simplified script only draws four episode-based curves (no variance shading):
    1) success_rate vs episode (mean over reps)
    2) reward vs episode (mean over reps)
    3) epsilon vs episode (mean over reps)
    4) loss vs episode (mean over reps)

Notes
- Auto-discovers runs under base_dir/tag: size_<N>/rep_<i>/
- Uses a headless Matplotlib backend (Agg) so it can run on servers without a display.

Usage
        python plot_replay_sweep.py \
                --base_dir ./v5_faster/v5_exp_data/replay_sweep \
                --tag rb122k \
                --max_ep 1000 \
                --smooth 5

Outputs
- All figures saved under: <base_dir>/<tag>/plots/
"""
from __future__ import annotations
import os
import re
import math
import argparse
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt

# (Optional pandas previously used for CSV export; no longer required in the simple version)

# --------------------------
# Data structures
# --------------------------

Series = Dict[str, List[Tuple[int, float]]]

@dataclass
class RunCurves:
    size: int
    rep: int
    series: Series = field(default_factory=dict)

    def get(self, key: str) -> List[Tuple[int, float]]:
        return self.series.get(key, [])

@dataclass
class RunSummary:
    size: int
    rep: int
    metrics: Dict[str, float | int | str | None] = field(default_factory=dict)

# --------------------------
# Parsing helpers
# --------------------------

def parse_curve_data(path: str) -> Series:
    """Parse curve_data.txt into a dict: metric -> list[(episode, value)]."""
    series: Series = defaultdict(list)
    if not os.path.isfile(path):
        return dict(series)
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Expected: v5,<metric>,<episode>,<value>
            parts = line.split(',')
            if len(parts) != 4:
                continue
            _, metric, ep_str, val_str = parts
            try:
                ep = int(ep_str)
                val = float(val_str)
            except ValueError:
                continue
            series[metric].append((ep, val))
    # Sort by episode
    for k in list(series.keys()):
        series[k].sort(key=lambda x: x[0])
    return dict(series)

_SUMMARY_KEY_FLOAT = {
    'success_rate','avg_score','std_score','avg_loss','avg_epsilon','avg_tries','tries_std',
    'time_min','episodes_per_min','avg_samples_per_episode'
}
_SUMMARY_KEY_INT = {
    'episodes','successes','total_samples_collected','first_success_episode','samples_at_first_success',
    'eps_min_reached_at_samples','buffer_capacity','final_buffer_occupancy','samples_dropped_overwritten',
    'effective_learn_starts'
}


def parse_summary(path: str) -> Dict[str, float | int | str | None]:
    """Parse summary.txt block produced by DQN_CAR_v5_Chapter3.py."""
    if not os.path.isfile(path):
        return {}
    out: Dict[str, float | int | str | None] = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('===='):
                continue
            if ':' not in line:
                continue
            k, v = [x.strip() for x in line.split(':', 1)]
            if v == 'N/A':
                out[k] = None
            elif k in _SUMMARY_KEY_INT:
                try:
                    out[k] = int(v)
                except Exception:
                    # Some keys could be floats but look like ints; be tolerant
                    try:
                        out[k] = int(float(v))
                    except Exception:
                        out[k] = None
            elif k in _SUMMARY_KEY_FLOAT:
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = None
            else:
                # For unknown keys keep raw string
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = v
    return out

# --------------------------
# Discovery
# --------------------------

def discover_runs(base_dir: str, tag: Optional[str]) -> Tuple[str, List[Tuple[int,int,str]]]:
    """Return (tag_dir, runs) where runs is list of (size, rep, run_dir)."""
    if tag:
        tag_dir = os.path.join(base_dir, str(tag))
    else:
        # If tag not provided, use the single directory under base_dir (if unique)
        candidates = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if len(candidates) == 1:
            tag_dir = os.path.join(base_dir, candidates[0])
        else:
            raise RuntimeError(f"Multiple tags found under {base_dir}, please specify --tag")
    if not os.path.isdir(tag_dir):
        raise FileNotFoundError(f"Tag directory not found: {tag_dir}")

    runs: List[Tuple[int,int,str]] = []
    size_pat = re.compile(r'^size_(\d+)$')
    rep_pat = re.compile(r'^rep_(\d+)$')
    for d in sorted(os.listdir(tag_dir)):
        m = size_pat.match(d)
        if not m:
            continue
        size = int(m.group(1))
        size_dir = os.path.join(tag_dir, d)
        for rd in sorted(os.listdir(size_dir)):
            mr = rep_pat.match(rd)
            if not mr:
                continue
            rep = int(mr.group(1))
            run_dir = os.path.join(size_dir, rd)
            runs.append((size, rep, run_dir))
    if not runs:
        raise RuntimeError(f"No runs discovered in {tag_dir}")
    return tag_dir, runs

# --------------------------
# Aggregation
# --------------------------

@dataclass
class AggregatedSeries:
    size: int
    metric: str
    episodes: np.ndarray  # int
    mean: np.ndarray      # float
    std: np.ndarray       # float
    count: np.ndarray     # int


def aggregate_metric(runs: List[RunCurves], metric: str, max_episode: Optional[int] = None,
                     min_rep_threshold: int = 1) -> List[AggregatedSeries]:
    """Aggregate metric across reps per size by episode index.

    For each episode index e, we average values from all runs that have that index.
    """
    # Group runs by size
    by_size: Dict[int, List[RunCurves]] = defaultdict(list)
    for r in runs:
        by_size[r.size].append(r)

    agg_list: List[AggregatedSeries] = []
    for size, rs in sorted(by_size.items()):
        # Build map episode -> list of values
        ep_vals: Dict[int, List[float]] = defaultdict(list)
        for r in rs:
            for ep, val in r.get(metric):
                if max_episode is not None and ep > max_episode:
                    continue
                ep_vals[ep].append(float(val))
        if not ep_vals:
            continue
        episodes_sorted = sorted(ep_vals.keys())
        vals_mean, vals_std, vals_cnt = [], [], []
        for ep in episodes_sorted:
            vs = ep_vals[ep]
            if len(vs) < min_rep_threshold:
                continue
            arr = np.asarray(vs, dtype=float)
            vals_mean.append(float(np.mean(arr)))
            vals_std.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)
            vals_cnt.append(int(len(arr)))
        if not vals_mean:
            continue
        agg_list.append(AggregatedSeries(
            size=size,
            metric=metric,
            episodes=np.asarray(episodes_sorted[:len(vals_mean)], dtype=int),
            mean=np.asarray(vals_mean, dtype=float),
            std=np.asarray(vals_std, dtype=float),
            count=np.asarray(vals_cnt, dtype=int),
        ))
    return agg_list

# --------------------------
# Plotting
# --------------------------

COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']


def _maybe_smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    w = min(window, len(y))
    if w <= 1:
        return y
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(y, kernel, mode='same')


def plot_aggregated_series(aggs: List[AggregatedSeries], metric: str, out_dir: str,
                           title: str, ylabel: str, smooth: int = 1,
                           ylimit: Optional[Tuple[float, float]] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for i, agg in enumerate(sorted(aggs, key=lambda a: a.size)):
        ys = _maybe_smooth(agg.mean, smooth)
        color = COLORS[i % len(COLORS)]
        plt.plot(agg.episodes, ys, label=f'size={agg.size}', color=color, linewidth=2)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    if ylimit:
        plt.ylim(*ylimit)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(out_dir, f'{metric}_by_size.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_box_final_success_rate(curves: List[RunCurves], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    # Collect final success rate for each run (last episode's value if exists)
    by_size_vals: Dict[int, List[float]] = defaultdict(list)
    for r in curves:
        sr = r.get('success_rate')
        if not sr:
            continue
        last_val = float(sr[-1][1])
        by_size_vals[r.size].append(last_val)
    if not by_size_vals:
        return ''
    sizes = sorted(by_size_vals.keys())
    data = [by_size_vals[s] for s in sizes]

    plt.figure(figsize=(10,6))
    b = plt.boxplot(data, labels=[str(s) for s in sizes], patch_artist=True)
    for patch, color in zip(b['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    plt.title('Final success_rate distribution by replay size')
    plt.xlabel('Replay buffer size')
    plt.ylabel('Final success_rate')
    plt.grid(True, axis='y', alpha=0.3)
    out_path = os.path.join(out_dir, 'final_success_rate_boxplot.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_bars_from_summary(summaries: List[RunSummary], metric: str, out_dir: str,
                           title: Optional[str] = None, lower_is_better: bool = False) -> str:
    os.makedirs(out_dir, exist_ok=True)
    by_size: Dict[int, List[float]] = defaultdict(list)
    for s in summaries:
        val = s.metrics.get(metric, None)
        if isinstance(val, (int, float)):
            by_size[s.size].append(float(val))
    if not by_size:
        return ''
    sizes = sorted(by_size.keys())
    means = []
    stds = []
    for s in sizes:
        arr = np.asarray(by_size[s], dtype=float)
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)

    x = np.arange(len(sizes))
    plt.figure(figsize=(10,6))
    bars = plt.bar(x, means, yerr=stds, capsize=4, color=[COLORS[i % len(COLORS)] for i in range(len(sizes))], alpha=0.8)
    # Annotate values
    for xi, m in zip(x, means):
        plt.text(xi, m, f"{m:.3f}", ha='center', va='bottom', fontsize=8)
    plt.xticks(x, [str(s) for s in sizes])
    plt.xlabel('Replay buffer size')
    plt.ylabel(metric)
    plt.title(title or f'{metric} by size (mean ± std over reps)')
    if lower_is_better:
        plt.gca().invert_yaxis()
    plt.grid(True, axis='y', alpha=0.3)
    out_path = os.path.join(out_dir, f'summary_{metric}_by_size.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_scatter_size_vs_metrics(summaries: List[RunSummary], metrics: List[str], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sizes = sorted({s.size for s in summaries})
    plt.figure(figsize=(10,6))
    for i, m in enumerate(metrics):
        xs, ys = [], []
        for s in summaries:
            v = s.metrics.get(m, None)
            if isinstance(v, (int, float)):
                xs.append(s.size)
                ys.append(float(v))
        if xs:
            plt.scatter(xs, ys, color=COLORS[i % len(COLORS)], alpha=0.6, label=m)
    plt.xlabel('Replay buffer size')
    plt.ylabel('Metric value')
    plt.title('Replay size vs metrics (per run points)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    out_path = os.path.join(out_dir, 'scatter_size_vs_metrics.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

# --------------------------
# Main
# --------------------------


def load_all(base_dir: str, tag: Optional[str]) -> Tuple[str, List[RunCurves], List[RunSummary]]:
    tag_dir, discovered = discover_runs(base_dir, tag)
    curves: List[RunCurves] = []
    sums: List[RunSummary] = []
    for size, rep, run_dir in discovered:
        curve_path = os.path.join(run_dir, 'curve_data.txt')
        summary_path = os.path.join(run_dir, 'summary.txt')
        series = parse_curve_data(curve_path)
        curves.append(RunCurves(size=size, rep=rep, series=series))
        metrics = parse_summary(summary_path)
        sums.append(RunSummary(size=size, rep=rep, metrics=metrics))
    return tag_dir, curves, sums


def main():
    ap = argparse.ArgumentParser(description='Plot replay sweep results (simple: 3 metrics)')
    ap.add_argument('--base_dir', type=str, default='./v5_faster/v5_exp_data/replay_sweep', help='Base sweep directory')
    ap.add_argument('--tag', type=str, default=None, help='Tag subdirectory name (e.g., rb122k)')
    ap.add_argument('--max_ep', type=int, default=None, help='Max episode to plot for curves')
    ap.add_argument('--smooth', type=int, default=1, help='Moving average window for mean curves')
    args = ap.parse_args()

    tag_dir, curves, _summaries_unused = load_all(args.base_dir, args.tag)
    out_dir = os.path.join(tag_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    # Only aggregate and plot three time series metrics (no std wording because shading removed)
    for metric, ylabel, ylim in [
        ('success_rate', 'Success rate', (0.0, 1.05)),
        ('reward', 'Reward', None),
        ('epsilon', 'Epsilon', (0.0, 1.05)),
        ('loss', 'Loss', None),
    ]:
        aggs = aggregate_metric(curves, metric, max_episode=args.max_ep)
        if aggs:
            path = plot_aggregated_series(aggs, metric, out_dir,
                                          title=f'{metric} vs episode (mean over reps)',
                                          ylabel=ylabel, smooth=args.smooth, ylimit=ylim)
            print(f'[plot] {path}')
        else:
            print(f'[warn] No data for metric: {metric}')

if __name__ == '__main__':
    main()
