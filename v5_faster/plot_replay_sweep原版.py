#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot replay buffer sweep results for v5_faster.

Features
- Auto-discover runs under base_dir/tag: size_<N>/rep_<i>/{curve_data.txt, summary.txt}
- Parse curve_data.txt for episode-wise series: reward, loss, epsilon, success_rate
- Parse summary.txt for run-level metrics
- Aggregate across reps for each buffer size and draw:
  1) success_rate vs episode (mean ± std)
  2) reward vs episode (mean ± std)
  3) epsilon vs episode (mean ± std)
  4) Final success rate per run boxplot by size
  5) Bars with error bars for key summary metrics by size: success_rate, episodes_per_min, avg_score, avg_loss,
     total_samples_collected, avg_samples_per_episode, first_success_episode (lower is better)
  6) Optional scatter: buffer size vs success_rate/episodes_per_min

Notes
- Works without pandas; if pandas is present, it will be used for convenience in CSV export only.
- Uses a headless Matplotlib backend (Agg) so it can run on servers without a display.

Usage
    python plot_replay_sweep.py \
        --base_dir ./v5_faster/v5_exp_data/replay_sweep \
        --tag rb123k \
        --max_ep 1000 \
        --smooth 5

Outputs
- All figures saved under: <base_dir>/<tag>/plots/
- Optionally writes derived CSVs when pandas is available.
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

# Try optional pandas
try:
    import pandas as pd  # type: ignore
    HAS_PD = True
except Exception:
    HAS_PD = False

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
    'time_min','episodes_per_min','avg_samples_per_episode','time_sec',
    'update_loss_variance','update_loss_variance_last100','td_mean_last100','td_std_last100',
    # 收敛与方差指标（浮点）
    'convergence_speed','reward_variance_full','reward_variance_last100',
    'success_rate_variance_full','success_rate_variance_last100',
    'loss_variance_full','loss_variance_last100',
    'convergence_threshold'
}
_SUMMARY_KEY_INT = {
    'episodes','successes','total_samples_collected','first_success_episode','samples_at_first_success',
    'eps_min_reached_at_samples','buffer_capacity','final_buffer_occupancy','samples_dropped_overwritten',
    'effective_learn_starts','learning_steps_to_first_success',
    # 收敛相关整数指标
    'episodes_to_convergence','samples_to_convergence',
    'convergence_patience','convergence_min_episodes','early_stop_episode','overwritten_count'
}
# 布尔型 (0/1) 指标
_SUMMARY_KEY_BOOL = {'early_stop_triggered'}


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
                continue
            if k in _SUMMARY_KEY_BOOL:
                try:
                    out[k] = 1 if float(v) != 0 else 0
                except Exception:
                    lv = v.strip().lower()
                    out[k] = 1 if lv in ('true', 'yes') else 0
                continue
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
    # Structured logging: group by size and list repeats
    by_size: Dict[int, List[int]] = defaultdict(list)
    for size, rep, _ in runs:
        by_size[size].append(rep)
    print(f"[discover] total_runs={len(runs)} tag_dir={tag_dir}")
    for size in sorted(by_size.keys()):
        reps_sorted = sorted(by_size[size])
        print(f"[discover] size={size} repeats={len(reps_sorted)} reps={reps_sorted}")
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
        # Shaded ± std
        # ylo = ys - agg.std
        # yhi = ys + agg.std
        # plt.fill_between(agg.episodes, ylo, yhi, color=color, alpha=0.15, linewidth=0)
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
        print(f"[load] size={size} rep={rep} curve_data={'OK' if series else 'EMPTY'} summary={'OK' if metrics else 'EMPTY'}")
    # Final summary log
    size_set = sorted({c.size for c in curves})
    print(f"[load] loaded_runs={len(curves)} sizes={size_set}")
    return tag_dir, curves, sums


def main():
    ap = argparse.ArgumentParser(description='Plot replay sweep results (v5_faster)')
    ap.add_argument('--base_dir', type=str, default='./v5_faster/v5_exp_data/replay_sweep', help='Base sweep directory')
    ap.add_argument('--tag', type=str, default=None, help='Tag subdirectory name (e.g., rb123k)')
    ap.add_argument('--max_ep', type=int, default=None, help='Max episode to plot for curves')
    ap.add_argument('--smooth', type=int, default=1, help='Moving average window for mean curves')
    ap.add_argument('--write_csv', action='store_true', help='Export aggregated CSVs (requires pandas)')
    args = ap.parse_args()

    tag_dir, curves, summaries = load_all(args.base_dir, args.tag)
    out_dir = os.path.join(tag_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    # ===== 核心时序曲线 (研究问题相关) =====
    print("\n[PHASE 1] Plotting time-series curves...")
    for metric, ylabel, ylim in [
        ('success_rate', 'Success Rate', (0.0, 1.05)),
        ('loss', 'Loss', None),
        ('td_mean', 'TD Error Mean', None),
        ('td_std', 'TD Error Std', None),
    ]:
        aggs = aggregate_metric(curves, metric, max_episode=args.max_ep)
        if aggs:
            path = plot_aggregated_series(aggs, metric, out_dir,
                                          title=f'{metric} vs episode (mean ± std over {len(summaries)} runs)',
                                          ylabel=ylabel, smooth=args.smooth, ylimit=ylim)
            print(f'[plot] {path}')
        else:
            print(f'[warn] No data for metric: {metric}')

    # Final success_rate distribution by size
    box_path = plot_box_final_success_rate(curves, out_dir)
    if box_path:
        print(f'[plot] {box_path}')

    # ===== 基础性能条形图 (快速概览) =====
    print("\n[PHASE 2] Plotting basic performance bars...")
    for metric, lower_better in [
        ('episodes_to_convergence', True),       # SRQ1.1
        ('samples_to_convergence', True),        # SRQ1.1
        ('success_rate', False),                 # SRQ1.1
        ('loss_variance_last100', True),         # SRQ1.2
        ('td_mean_last100', True),               # Off-policy bias
    ]:
        p = plot_bars_from_summary(summaries, metric, out_dir, lower_is_better=lower_better)
        if p:
            print(f'[plot] {p}')
        else:
            print(f'[warn] No summary data for: {metric}')

    # ===== SRQ1.1: 样本效率分析 (Sample Efficiency Analysis) =====
    print("\n[PHASE 3] SRQ1.1: Sample Efficiency Analysis...")
    
    # 图1: Episodes vs Samples to Convergence (scatter with trend line)
    conv_data = []
    for s in summaries:
        ep_conv = s.metrics.get('episodes_to_convergence', None)
        samp_conv = s.metrics.get('samples_to_convergence', None)
        if isinstance(ep_conv, (int,float)) and isinstance(samp_conv, (int,float)):
            conv_data.append((s.size, float(ep_conv), float(samp_conv)))
    
    if conv_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图: Buffer size vs Episodes to convergence
        by_size_ep = defaultdict(list)
        for sz, ep, _ in conv_data:
            by_size_ep[sz].append(ep)
        sizes = sorted(by_size_ep.keys())
        mean_eps = [float(np.mean(by_size_ep[sz])) for sz in sizes]
        std_eps = [float(np.std(by_size_ep[sz], ddof=1)) if len(by_size_ep[sz]) > 1 else 0.0 for sz in sizes]
        
        ax1.errorbar(sizes, mean_eps, yerr=std_eps, fmt='o-', capsize=5, linewidth=2.5, markersize=10, color='#1f77b4')
        for sz, m, s in zip(sizes, mean_eps, std_eps):
            ax1.text(sz, m, f'{m:.1f}±{s:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.set_xlabel('Replay Buffer Size', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Episodes to Convergence', fontsize=12, fontweight='bold')
        ax1.set_title('SRQ1.1a: Convergence Speed (Episodes)', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axhline(y=np.mean(mean_eps), color='red', linestyle='--', alpha=0.5, label=f'Mean={np.mean(mean_eps):.1f}')
        ax1.legend()
        
        # 右图: Buffer size vs Samples to convergence  
        by_size_samp = defaultdict(list)
        for sz, _, samp in conv_data:
            by_size_samp[sz].append(samp)
        mean_samps = [float(np.mean(by_size_samp[sz])) for sz in sizes]
        std_samps = [float(np.std(by_size_samp[sz], ddof=1)) if len(by_size_samp[sz]) > 1 else 0.0 for sz in sizes]
        
        ax2.errorbar(sizes, mean_samps, yerr=std_samps, fmt='o-', capsize=5, linewidth=2.5, markersize=10, color='#2ca02c')
        for sz, m, s in zip(sizes, mean_samps, std_samps):
            ax2.text(sz, m, f'{m:.0f}±{s:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.set_xlabel('Replay Buffer Size', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Samples to Convergence', fontsize=12, fontweight='bold')
        ax2.set_title('SRQ1.1b: Sample Efficiency', fontsize=14, fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.axhline(y=np.mean(mean_samps), color='red', linestyle='--', alpha=0.5, label=f'Mean={np.mean(mean_samps):.0f}')
        ax2.legend()
        
        plt.tight_layout()
        out_conv_eff = os.path.join(out_dir, 'SRQ1.1_convergence_efficiency.png')
        plt.savefig(out_conv_eff, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'[plot] {out_conv_eff}')
    
    # 图2: Convergence Speed vs Buffer Size (直方图对比)
    conv_speed_data = defaultdict(list)
    for s in summaries:
        cs = s.metrics.get('convergence_speed', None)
        if isinstance(cs, (int,float)) and cs > 0:
            conv_speed_data[s.size].append(float(cs))
    
    if conv_speed_data:
        sizes = sorted(conv_speed_data.keys())
        means = [float(np.mean(conv_speed_data[sz])) for sz in sizes]
        stds = [float(np.std(conv_speed_data[sz], ddof=1)) if len(conv_speed_data[sz]) > 1 else 0.0 for sz in sizes]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(sizes))
        bars = ax.bar(x, means, yerr=stds, capsize=6, color=[COLORS[i % len(COLORS)] for i in range(len(sizes))],
                      alpha=0.85, edgecolor='black', linewidth=1.5)
        
        for i, (sz, m, s) in enumerate(zip(sizes, means, stds)):
            ax.text(i, m + s, f'{m:.6f}\n±{s:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(sz) for sz in sizes], fontsize=11)
        ax.set_xlabel('Replay Buffer Size', fontsize=13, fontweight='bold')
        ax.set_ylabel('Convergence Speed (1/episodes)', fontsize=13, fontweight='bold')
        ax.set_title('SRQ1.1c: Convergence Speed Comparison (Higher is Better)', fontsize=15, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加最优标记
        best_idx = np.argmax(means)
        ax.annotate('BEST', xy=(best_idx, means[best_idx]), xytext=(best_idx, means[best_idx] + stds[best_idx] * 1.5),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2), fontsize=12, color='red', fontweight='bold',
                   ha='center')
        
        plt.tight_layout()
        out_conv_speed = os.path.join(out_dir, 'SRQ1.1_convergence_speed.png')
        plt.savefig(out_conv_speed, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'[plot] {out_conv_speed}')
    
    # ===== SRQ1.2: 训练稳定性与方差分析 (Stability & Variance Analysis) =====
    print("\n[PHASE 4] SRQ1.2: Stability & Variance Analysis...")
    
    # 图3: 多指标方差热力图 (3x5 grid)
    variance_metrics = [
        ('success_rate_variance_last100', 'Success Rate\nVariance'),
        ('reward_variance_last100', 'Reward\nVariance'),
        ('loss_variance_last100', 'Loss\nVariance')
    ]
    
    var_data = {}
    for metric, label in variance_metrics:
        by_size_var = defaultdict(list)
        for s in summaries:
            v = s.metrics.get(metric, None)
            if isinstance(v, (int,float)):
                by_size_var[s.size].append(float(v))
        if by_size_var:
            var_data[label] = by_size_var
    
    if var_data:
        sizes = sorted(list(var_data.values())[0].keys())
        n_metrics = len(var_data)
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 5 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (label, by_size) in enumerate(var_data.items()):
            ax = axes[idx]
            means = [float(np.mean(by_size[sz])) for sz in sizes]
            stds = [float(np.std(by_size[sz], ddof=1)) if len(by_size[sz]) > 1 else 0.0 for sz in sizes]
            
            x = np.arange(len(sizes))
            bars = ax.bar(x, means, yerr=stds, capsize=5, color='#d62728' if 'Loss' in label else '#ff7f0e' if 'Reward' in label else '#2ca02c',
                         alpha=0.8, edgecolor='black', linewidth=1.2)
            
            for i, (sz, m, s) in enumerate(zip(sizes, means, stds)):
                ax.text(i, m + s, f'{m:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels([str(sz) for sz in sizes], fontsize=10)
            ax.set_xlabel('Replay Buffer Size', fontsize=12, fontweight='bold')
            ax.set_ylabel(label, fontsize=12, fontweight='bold')
            ax.set_title(f'SRQ1.2: {label} (Lower is Better = More Stable)', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # 标记最稳定配置
            min_idx = np.argmin(means)
            ax.annotate('MOST STABLE', xy=(min_idx, means[min_idx]), xytext=(min_idx, means[min_idx] + stds[min_idx] * 2),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2), fontsize=11, color='green', fontweight='bold',
                       ha='center')
        
        plt.tight_layout()
        out_var_heat = os.path.join(out_dir, 'SRQ1.2_variance_heatmap.png')
        plt.savefig(out_var_heat, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'[plot] {out_var_heat}')
    
    # 图4: CoV (Coefficient of Variation) 跨种子稳定性
    cov_metrics = ['success_rate', 'avg_score', 'episodes_to_convergence', 'samples_to_convergence']
    
    # CoV统一绘制在一张图上
    cov_data = {}
    for cov_metric in cov_metrics:
        by_size_vals = defaultdict(list)
        for s in summaries:
            v = s.metrics.get(cov_metric, None)
            if isinstance(v, (int,float)):
                by_size_vals[s.size].append(float(v))
        if not by_size_vals:
            continue
        sizes = sorted(by_size_vals.keys())
        cvs = []
        for sz in sizes:
            arr = np.asarray(by_size_vals[sz], dtype=float)
            if arr.size > 1 and abs(arr.mean()) > 1e-9:
                cvs.append(float(arr.std(ddof=1) / abs(arr.mean())))
            else:
                cvs.append(0.0)
        cov_data[cov_metric] = (sizes, cvs)
    
    if cov_data:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (metric, (sizes, cvs)) in enumerate(cov_data.items()):
            ax = axes[idx]
            x = np.arange(len(sizes))
            bars = ax.bar(x, cvs, color=COLORS[idx % len(COLORS)], alpha=0.85, edgecolor='black', linewidth=1.2)
            
            for i, (sz, cv) in enumerate(zip(sizes, cvs)):
                ax.text(i, cv, f"{cv:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in sizes], fontsize=10)
            ax.set_ylabel(f'CoV', fontsize=12, fontweight='bold')
            ax.set_xlabel('Replay Buffer Size', fontsize=12, fontweight='bold')
            ax.set_title(f'SRQ1.2: CoV of {metric} (Lower = More Stable)', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # 标记最低CoV
            if cvs:
                min_idx = np.argmin(cvs)
                ax.annotate('MIN', xy=(min_idx, cvs[min_idx]), xytext=(min_idx, cvs[min_idx] * 1.3),
                           arrowprops=dict(arrowstyle='->', color='green', lw=1.5), fontsize=10, color='green',
                           fontweight='bold', ha='center')
        
        plt.suptitle('SRQ1.2: Coefficient of Variation (CoV) Analysis Across Seeds', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        out_cov_all = os.path.join(out_dir, 'SRQ1.2_CoV_analysis.png')
        plt.savefig(out_cov_all, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'[plot] {out_cov_all}')

    # ===== 补充分析图表 =====
    print("\n[PHASE 5] Additional Analysis...")
    
    # 图5: Buffer覆盖率分析
    buffer_data = []
    for s in summaries:
        sz = s.metrics.get('buffer_capacity', None)
        ow = s.metrics.get('overwritten_count', None)
        tot = s.metrics.get('total_samples_collected', None)
        sr = s.metrics.get('success_rate', None)
        if all(isinstance(x, (int,float)) for x in [sz, ow, tot, sr]):
            coverage = float(ow) / float(tot) if tot > 0 else 0.0
            buffer_data.append((s.size, float(sz), float(ow), float(tot), coverage, float(sr)))
    
    if buffer_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图: Buffer refresh rate vs buffer size
        by_size_cov = defaultdict(list)
        for (sz, _, _, _, cov, _) in buffer_data:
            by_size_cov[sz].append(cov)
        sizes = sorted(by_size_cov.keys())
        mean_covs = [float(np.mean(by_size_cov[sz])) for sz in sizes]
        std_covs = [float(np.std(by_size_cov[sz], ddof=1)) if len(by_size_cov[sz]) > 1 else 0.0 for sz in sizes]
        
        ax1.errorbar(sizes, mean_covs, yerr=std_covs, fmt='o-', capsize=5, linewidth=2.5, markersize=10, color='#9467bd')
        for sz, m in zip(sizes, mean_covs):
            ax1.text(sz, m, f'{m:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.set_xlabel('Replay Buffer Size', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Sample Overwrite Rate', fontsize=12, fontweight='bold')
        ax1.set_title('Buffer Refresh Rate vs Size (Higher = Fresher Samples)', fontsize=13, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 右图: Coverage vs Success rate scatter
        sizes_u = sorted(set(bd[0] for bd in buffer_data))
        for sz in sizes_u:
            pts = [(cov, sr) for (s, _, _, _, cov, sr) in buffer_data if s == sz]
            if pts:
                covs, srs = zip(*pts)
                color_idx = sizes_u.index(sz)
                ax2.scatter(covs, srs, s=100, alpha=0.7, label=f'size={sz}',
                           c=[COLORS[color_idx % len(COLORS)]], edgecolors='black', linewidths=1.5)
        
        ax2.set_xlabel('Sample Overwrite Rate', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Final Success Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Buffer Refresh vs Performance', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        out_buf = os.path.join(out_dir, 'buffer_coverage_analysis.png')
        plt.savefig(out_buf, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'[plot] {out_buf}')
    
    # 图6: TD Error分析 (Off-policy bias)
    td_data = defaultdict(lambda: {'mean': [], 'std': []})
    for s in summaries:
        tdm = s.metrics.get('td_mean_last100', None)
        tds = s.metrics.get('td_std_last100', None)
        if isinstance(tdm, (int,float)) and isinstance(tds, (int,float)):
            td_data[s.size]['mean'].append(float(tdm))
            td_data[s.size]['std'].append(float(tds))
    
    if td_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sizes = sorted(td_data.keys())
        mean_td_means = [float(np.mean(td_data[sz]['mean'])) for sz in sizes]
        mean_td_stds = [float(np.mean(td_data[sz]['std'])) for sz in sizes]
        std_td_means = [float(np.std(td_data[sz]['mean'], ddof=1)) if len(td_data[sz]['mean']) > 1 else 0.0 for sz in sizes]
        std_td_stds = [float(np.std(td_data[sz]['std'], ddof=1)) if len(td_data[sz]['std']) > 1 else 0.0 for sz in sizes]
        
        # 左图: TD error magnitude
        ax1.errorbar(sizes, mean_td_means, yerr=std_td_means, fmt='o-', capsize=5, linewidth=2.5, markersize=10, color='#d62728')
        for sz, m in zip(sizes, mean_td_means):
            ax1.text(sz, m, f'{m:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.set_xlabel('Replay Buffer Size', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean |TD Error| (last 100 updates)', fontsize=12, fontweight='bold')
        ax1.set_title('Off-policy Bias: TD Error Magnitude', fontsize=13, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 右图: TD error volatility
        ax2.errorbar(sizes, mean_td_stds, yerr=std_td_stds, fmt='o-', capsize=5, linewidth=2.5, markersize=10, color='#9467bd')
        for sz, m in zip(sizes, mean_td_stds):
            ax2.text(sz, m, f'{m:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Replay Buffer Size', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean TD Error Std (last 100 updates)', fontsize=12, fontweight='bold')
        ax2.set_title('Off-policy Bias: TD Error Volatility', fontsize=13, fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        out_td = os.path.join(out_dir, 'td_error_analysis.png')
        plt.savefig(out_td, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'[plot] {out_td}')
    
    # 最终总结
    print("\n" + "="*60)
    print("[SUMMARY] All research plots generated successfully!")
    print("="*60)

    # Optional CSV exports
    if args.write_csv and HAS_PD:
        # Curves export: stacked long table
        rows = []
        for r in curves:
            for m, seq in r.series.items():
                for ep, val in seq:
                    rows.append({'size': r.size, 'rep': r.rep, 'metric': m, 'episode': ep, 'value': val})
        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(out_dir, 'curves_long.csv')
            df.to_csv(csv_path, index=False)
            print(f'[csv] {csv_path}')
        # Summaries export
        rows = []
        for s in summaries:
            row = {'size': s.size, 'rep': s.rep}
            for k, v in s.metrics.items():
                row[k] = v
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(out_dir, 'summaries.csv')
            df.to_csv(csv_path, index=False)
            print(f'[csv] {csv_path}')
    elif args.write_csv and not HAS_PD:
        print('[warn] pandas not available, skip CSV export')

if __name__ == '__main__':
    main()
