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
        --tag rb122k \
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
                     min_rep_threshold: int = 1,
                     align_mode: str = 'median',  # 默认使用中位数对齐，避免显示稀疏数据
                     min_sample_coverage: float = 0.5) -> List[AggregatedSeries]:
    """Aggregate metric across reps per size by episode index.

    Args:
        runs: List of RunCurves
        metric: Metric name to aggregate
        max_episode: Maximum episode to include
        min_rep_threshold: Minimum number of reps required per episode
        align_mode: How to align different run lengths:
            'min': Truncate to shortest run length (most conservative)
            'median': Truncate to median run length (balanced)
            'quantile': Truncate to quantile (e.g., 75th percentile)
            'coverage': Truncate where coverage drops below min_sample_coverage
            'max': Use all available data (current behavior)
        min_sample_coverage: For 'coverage' mode, minimum fraction of runs required

    For each episode index e, we average values from all runs that have that index.
    """
    # Group runs by size
    by_size: Dict[int, List[RunCurves]] = defaultdict(list)
    for r in runs:
        by_size[r.size].append(r)

    agg_list: List[AggregatedSeries] = []
    for size, rs in sorted(by_size.items()):
        # Step 1: Collect max episode for each run
        run_max_episodes = []
        ep_vals: Dict[int, List[float]] = defaultdict(list)
        
        for r in rs:
            run_eps = [ep for ep, _ in r.get(metric)]
            if run_eps:
                run_max_episodes.append(max(run_eps))
            
            for ep, val in r.get(metric):
                if max_episode is not None and ep > max_episode:
                    continue
                ep_vals[ep].append(float(val))
        
        if not ep_vals or not run_max_episodes:
            continue
        
        # Step 2: Determine cutoff episode based on align_mode
        total_runs = len(rs)
        cutoff_ep = None
        
        if align_mode == 'min':
            cutoff_ep = min(run_max_episodes)
        elif align_mode == 'median':
            cutoff_ep = int(np.median(run_max_episodes))
        elif align_mode == 'quantile':
            cutoff_ep = int(np.percentile(run_max_episodes, 75))  # 75th percentile
        elif align_mode == 'coverage':
            # Find episode where sample count drops below threshold
            episodes_sorted_temp = sorted(ep_vals.keys())
            for ep in episodes_sorted_temp:
                coverage = len(ep_vals[ep]) / total_runs
                if coverage < min_sample_coverage:
                    cutoff_ep = ep - 1
                    break
            if cutoff_ep is None:
                cutoff_ep = max(episodes_sorted_temp)
        elif align_mode == 'max':
            cutoff_ep = None  # No truncation
        else:
            cutoff_ep = None
        
        # Step 3: Apply cutoff
        if cutoff_ep is not None:
            ep_vals = {ep: vals for ep, vals in ep_vals.items() if ep <= cutoff_ep}
        
        if not ep_vals:
            continue
        
        episodes_sorted = sorted(ep_vals.keys())
        vals_mean, vals_std, vals_cnt = [], [], []
        valid_episodes = []
        
        for ep in episodes_sorted:
            vs = ep_vals[ep]
            if len(vs) < min_rep_threshold:
                continue
            arr = np.asarray(vs, dtype=float)
            vals_mean.append(float(np.mean(arr)))
            vals_std.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)
            vals_cnt.append(int(len(arr)))
            valid_episodes.append(ep)
        
        if not vals_mean:
            continue
        
        # Log truncation info
        if cutoff_ep is not None:
            orig_max = max(run_max_episodes) if run_max_episodes else 0
            print(f"  [align] size={size} metric={metric}: truncated from {orig_max} to {cutoff_ep} episodes (mode={align_mode})")
        
        agg_list.append(AggregatedSeries(
            size=size,
            metric=metric,
            episodes=np.asarray(valid_episodes, dtype=int),
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
                           ylimit: Optional[Tuple[float, float]] = None,
                           show_sample_count: bool = True,
                           show_std_band: bool = False) -> str:
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 7))
    
    for i, agg in enumerate(sorted(aggs, key=lambda a: a.size)):
        ys = _maybe_smooth(agg.mean, smooth)
        color = COLORS[i % len(COLORS)]
        
        # Main line with sample count info in legend
        line, = plt.plot(agg.episodes, ys, label=f'size={agg.size} (n={agg.count[0]}→{agg.count[-1]})', 
                        color=color, linewidth=2.5, alpha=0.9)
        
        # Optional std band
        if show_std_band:
            ylo = ys - agg.std
            yhi = ys + agg.std
            plt.fill_between(agg.episodes, ylo, yhi, color=color, alpha=0.15, linewidth=0)
        
        # Mark endpoint with sample count
        if show_sample_count and len(agg.episodes) > 0:
            last_ep = agg.episodes[-1]
            last_y = ys[-1]
            last_n = agg.count[-1]
            plt.scatter([last_ep], [last_y], color=color, s=100, zorder=5, 
                       edgecolors='black', linewidths=1.5)
            plt.text(last_ep * 1.02, last_y, f'n={last_n}', 
                    fontsize=9, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=color, alpha=0.8))
    
    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if ylimit:
        plt.ylim(*ylimit)
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    out_path = os.path.join(out_dir, f'{metric}_by_size.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
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


# ===========================
# 增强版Loss可视化 (100次重复专用)
# ===========================

def plot_loss_enhanced_1_multilevel_ci(curves: List[RunCurves], out_dir: str, 
                                       max_ep: Optional[int] = None, smooth: int = 1) -> str:
    """Loss图1: 均值 + 多层置信区间 (±1σ深色, ±2σ浅色)"""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    by_size: Dict[int, List[RunCurves]] = defaultdict(list)
    for r in curves:
        by_size[r.size].append(r)
    
    for i, (size, runs) in enumerate(sorted(by_size.items())):
        # 收集所有runs的loss数据
        ep_vals: Dict[int, List[float]] = defaultdict(list)
        run_lengths = []
        for r in runs:
            loss_data = r.get('loss')
            if not loss_data:
                continue
            run_lengths.append(max(ep for ep, _ in loss_data))
            for ep, val in loss_data:
                if max_ep is None or ep <= max_ep:
                    ep_vals[ep].append(val)
        
        # 智能截断：使用中位数长度，避免显示稀疏尾部
        if run_lengths and max_ep is None:
            median_length = int(np.median(run_lengths))
            ep_vals = {ep: vals for ep, vals in ep_vals.items() if ep <= median_length}
            print(f"[plot_loss_ci] size={size}: auto-truncate to median_length={median_length} (run_lengths: min={min(run_lengths)}, max={max(run_lengths)})")
        
        if not ep_vals:
            continue
        
        # 计算统计量
        episodes = sorted(ep_vals.keys())
        means, stds, counts = [], [], []
        for ep in episodes:
            vals = ep_vals[ep]
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            counts.append(len(vals))
        
        episodes = np.array(episodes)
        means = np.array(means)
        stds = np.array(stds)
        counts = np.array(counts)
        
        # 平滑
        if smooth > 1:
            means = _maybe_smooth(means, smooth)
            stds = _maybe_smooth(stds, smooth)
        
        color = COLORS[i % len(COLORS)]
        
        # 均值线
        ax.plot(episodes, means, color=color, linewidth=3, alpha=0.9,
               label=f'size={size} (n={len(runs)})', zorder=10)
        
        # ±2σ (95% CI) - 浅色
        ax.fill_between(episodes, means - 2*stds, means + 2*stds,
                       color=color, alpha=0.1, linewidth=0, zorder=5)
        
        # ±1σ (68% CI) - 深色
        ax.fill_between(episodes, means - stds, means + stds,
                       color=color, alpha=0.25, linewidth=0, zorder=6)
        
        # 标注终点
        if len(episodes) > 0:
            ax.scatter([episodes[-1]], [means[-1]], color=color, s=120,
                      zorder=15, edgecolors='black', linewidths=2)
            ax.text(episodes[-1] * 1.01, means[-1], f'n={counts[-1]}',
                   fontsize=9, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=color, alpha=0.9))
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('LOSS: Mean ± Confidence Intervals (100 runs per size)',
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 图例
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(facecolor='gray', alpha=0.25, label='±1σ (68% CI)'),
               Patch(facecolor='gray', alpha=0.1, label='±2σ (95% CI)')]
    labels += ['±1σ (68% CI)', '±2σ (95% CI)']
    ax.legend(handles=handles, labels=labels, loc='best', fontsize=10, 
             framealpha=0.95, ncol=2)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'loss_1_mean_multilevel_ci.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def plot_loss_enhanced_2_quantile_bands(curves: List[RunCurves], out_dir: str,
                                       max_ep: Optional[int] = None, smooth: int = 1) -> str:
    """Loss图2: 分位数带 (中位数 + 25-75% IQR + 10-90%)"""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    by_size: Dict[int, List[RunCurves]] = defaultdict(list)
    for r in curves:
        by_size[r.size].append(r)
    
    for i, (size, runs) in enumerate(sorted(by_size.items())):
        ep_vals: Dict[int, List[float]] = defaultdict(list)
        run_lengths = []
        for r in runs:
            loss_data = r.get('loss')
            if not loss_data:
                continue
            run_lengths.append(max(ep for ep, _ in loss_data))
            for ep, val in loss_data:
                if max_ep is None or ep <= max_ep:
                    ep_vals[ep].append(val)
        
        # 智能截断：使用中位数长度
        if run_lengths and max_ep is None:
            median_length = int(np.median(run_lengths))
            ep_vals = {ep: vals for ep, vals in ep_vals.items() if ep <= median_length}
            print(f"[plot_loss_quantile] size={size}: auto-truncate to median_length={median_length}")
        
        if not ep_vals:
            continue
        
        episodes = sorted(ep_vals.keys())
        q50s, q25s, q75s, q10s, q90s = [], [], [], [], []
        for ep in episodes:
            vals = np.array(ep_vals[ep])
            q50s.append(np.percentile(vals, 50))
            q25s.append(np.percentile(vals, 25))
            q75s.append(np.percentile(vals, 75))
            q10s.append(np.percentile(vals, 10))
            q90s.append(np.percentile(vals, 90))
        
        episodes = np.array(episodes)
        q50s = np.array(q50s)
        q25s = np.array(q25s)
        q75s = np.array(q75s)
        q10s = np.array(q10s)
        q90s = np.array(q90s)
        
        # 平滑
        if smooth > 1:
            q50s = _maybe_smooth(q50s, smooth)
            q25s = _maybe_smooth(q25s, smooth)
            q75s = _maybe_smooth(q75s, smooth)
            q10s = _maybe_smooth(q10s, smooth)
            q90s = _maybe_smooth(q90s, smooth)
        
        color = COLORS[i % len(COLORS)]
        
        # 中位数
        ax.plot(episodes, q50s, color=color, linewidth=3, alpha=0.9,
               label=f'size={size} (median)', zorder=10)
        
        # 10-90% (浅色)
        ax.fill_between(episodes, q10s, q90s, color=color, alpha=0.15,
                       linewidth=0, zorder=5)
        
        # 25-75% IQR (深色)
        ax.fill_between(episodes, q25s, q75s, color=color, alpha=0.3,
                       linewidth=0, zorder=6)
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('LOSS: Quantile Evolution (100 runs per size)',
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 图例
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(facecolor='gray', alpha=0.3, label='IQR (25-75%)'),
               Patch(facecolor='gray', alpha=0.15, label='10-90%')]
    labels += ['IQR (25-75%)', '10-90%']
    ax.legend(handles=handles, labels=labels, loc='best', fontsize=10,
             framealpha=0.95, ncol=2)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'loss_2_quantile_bands.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def plot_loss_enhanced_3_cv_evolution(curves: List[RunCurves], out_dir: str,
                                     max_ep: Optional[int] = None, smooth: int = 10) -> str:
    """Loss图3: 变异系数演化 (CV = std/mean) - 完整数据，无截断"""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    by_size: Dict[int, List[RunCurves]] = defaultdict(list)
    for r in curves:
        by_size[r.size].append(r)
    
    for i, (size, runs) in enumerate(sorted(by_size.items())):
        color = COLORS[i % len(COLORS)]
        
        ep_dict: Dict[int, List[float]] = defaultdict(list)
        for r in runs:
            loss_data = r.series.get('loss', [])
            if max_ep is not None:
                loss_data = [x for x in loss_data if x[0] <= max_ep]
            for ep, val in loss_data:
                ep_dict[ep].append(val)
        
        if not ep_dict:
            continue
        
        episodes = sorted(ep_dict.keys())
        cv_list = []
        for ep in episodes:
            vals = ep_dict[ep]
            if len(vals) >= 2:
                mu = np.mean(vals)
                sigma = np.std(vals, ddof=1)
                cv = (sigma / mu) if mu > 1e-6 else 0.0
                cv_list.append(cv)
            else:
                cv_list.append(0.0)
        
        if not cv_list:
            continue
        
        episodes_arr = np.array(episodes)
        cv_arr = _maybe_smooth(np.array(cv_list), smooth)
        
        ax.plot(episodes_arr, cv_arr, color=color, linewidth=2.5,
                label=f'size={size}', alpha=0.85)
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (CV = σ/μ)', fontsize=13, fontweight='bold')
    ax.set_title('LOSS: Relative Variability Evolution (100 runs per size)',
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 参考线
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, linewidth=1.5,
              label='CV=0.1 (low variability)')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, linewidth=1.5,
              label='CV=0.3 (moderate)')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
              label='CV=0.5 (high)')
    
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'loss_3_coefficient_of_variation.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


# ===========================
# 聚焦版本 (0-2000 episodes)
# ===========================

def plot_loss_enhanced_1_multilevel_ci_zoom(curves: List[RunCurves], out_dir: str, 
                                            max_ep: int = 2000, smooth: int = 1) -> str:
    """Loss图1聚焦版: 均值 + 多层置信区间 (0-2000 episodes)"""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    by_size: Dict[int, List[RunCurves]] = defaultdict(list)
    for r in curves:
        by_size[r.size].append(r)
    
    for i, (size, runs) in enumerate(sorted(by_size.items())):
        color = COLORS[i % len(COLORS)]
        
        # 收集所有run的loss数据
        ep_dict: Dict[int, List[float]] = defaultdict(list)
        run_lengths = []
        for r in runs:
            loss_data = [x for x in r.series.get('loss', []) if x[0] <= max_ep]
            if loss_data:
                run_lengths.append(max(ep for ep, _ in loss_data))
            for ep, val in loss_data:
                ep_dict[ep].append(val)
        
        if not ep_dict:
            continue
        
        # 计算中位数截断点
        if run_lengths:
            median_length = int(np.median(run_lengths))
            print(f"  size={size}: {len(runs)} runs, median_length={median_length} (zoom)")
        else:
            median_length = max_ep
        
        # 截断到中位数长度
        episodes = sorted([ep for ep in ep_dict.keys() if ep <= median_length])
        
        means = []
        stds = []
        for ep in episodes:
            vals = ep_dict[ep]
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        
        if not means:
            continue
        
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        episodes_arr = np.array(episodes)
        
        # 平滑
        means_smooth = _maybe_smooth(means_arr, smooth)
        stds_smooth = _maybe_smooth(stds_arr, smooth)
        
        # 绘制均值
        ax.plot(episodes_arr, means_smooth, color=color, linewidth=2.5, 
                label=f'size={size}')
        
        # ±1σ (68% CI) - 深色
        ax.fill_between(episodes_arr, 
                       means_smooth - stds_smooth, 
                       means_smooth + stds_smooth,
                       color=color, alpha=0.25, linewidth=0)
        
        # ±2σ (95% CI) - 浅色
        ax.fill_between(episodes_arr,
                       means_smooth - 2*stds_smooth,
                       means_smooth + 2*stds_smooth,
                       color=color, alpha=0.1, linewidth=0)
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('LOSS: Mean ± CI (0-2000 episodes, zoomed)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, max_ep)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(facecolor='gray', alpha=0.25, label='±1σ (68% CI)'),
               Patch(facecolor='gray', alpha=0.1, label='±2σ (95% CI)')]
    labels += ['±1σ (68% CI)', '±2σ (95% CI)']
    ax.legend(handles=handles, labels=labels, loc='best', fontsize=10, 
             framealpha=0.95, ncol=2)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'loss_1_mean_multilevel_ci_zoom.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def plot_loss_enhanced_2_quantile_bands_zoom(curves: List[RunCurves], out_dir: str,
                                            max_ep: int = 2000, smooth: int = 1) -> str:
    """Loss图2聚焦版: 分位数带 (0-2000 episodes)"""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    by_size: Dict[int, List[RunCurves]] = defaultdict(list)
    for r in curves:
        by_size[r.size].append(r)
    
    for i, (size, runs) in enumerate(sorted(by_size.items())):
        color = COLORS[i % len(COLORS)]
        
        ep_dict: Dict[int, List[float]] = defaultdict(list)
        run_lengths = []
        for r in runs:
            loss_data = [x for x in r.series.get('loss', []) if x[0] <= max_ep]
            if loss_data:
                run_lengths.append(max(ep for ep, _ in loss_data))
            for ep, val in loss_data:
                ep_dict[ep].append(val)
        
        if not ep_dict:
            continue
        
        if run_lengths:
            median_length = int(np.median(run_lengths))
        else:
            median_length = max_ep
        
        episodes = sorted([ep for ep in ep_dict.keys() if ep <= median_length])
        
        p10_list, p25_list, p50_list, p75_list, p90_list = [], [], [], [], []
        for ep in episodes:
            vals = ep_dict[ep]
            p10_list.append(np.percentile(vals, 10))
            p25_list.append(np.percentile(vals, 25))
            p50_list.append(np.percentile(vals, 50))
            p75_list.append(np.percentile(vals, 75))
            p90_list.append(np.percentile(vals, 90))
        
        if not p50_list:
            continue
        
        episodes_arr = np.array(episodes)
        p10_arr = _maybe_smooth(np.array(p10_list), smooth)
        p25_arr = _maybe_smooth(np.array(p25_list), smooth)
        p50_arr = _maybe_smooth(np.array(p50_list), smooth)
        p75_arr = _maybe_smooth(np.array(p75_list), smooth)
        p90_arr = _maybe_smooth(np.array(p90_list), smooth)
        
        ax.plot(episodes_arr, p50_arr, color=color, linewidth=2.5,
                label=f'size={size} (median)')
        
        ax.fill_between(episodes_arr, p25_arr, p75_arr,
                       color=color, alpha=0.3, linewidth=0)
        ax.fill_between(episodes_arr, p10_arr, p90_arr,
                       color=color, alpha=0.15, linewidth=0)
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('LOSS: Quantile Bands (0-2000 episodes, zoomed)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, max_ep)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(facecolor='gray', alpha=0.3, label='IQR (25-75%)'),
               Patch(facecolor='gray', alpha=0.15, label='10-90%')]
    labels += ['IQR (25-75%)', '10-90%']
    ax.legend(handles=handles, labels=labels, loc='best', fontsize=10,
             framealpha=0.95, ncol=2)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'loss_2_quantile_bands_zoom.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def plot_loss_enhanced_3_cv_evolution_zoom(curves: List[RunCurves], out_dir: str,
                                          max_ep: int = 2000, smooth: int = 10) -> str:
    """Loss图3聚焦版: 变异系数演化 (0-2000 episodes)"""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    by_size: Dict[int, List[RunCurves]] = defaultdict(list)
    for r in curves:
        by_size[r.size].append(r)
    
    for i, (size, runs) in enumerate(sorted(by_size.items())):
        color = COLORS[i % len(COLORS)]
        
        ep_dict: Dict[int, List[float]] = defaultdict(list)
        for r in runs:
            loss_data = [x for x in r.series.get('loss', []) if x[0] <= max_ep]
            for ep, val in loss_data:
                ep_dict[ep].append(val)
        
        if not ep_dict:
            continue
        
        episodes = sorted(ep_dict.keys())
        cv_list = []
        for ep in episodes:
            vals = ep_dict[ep]
            if len(vals) >= 2:
                mu = np.mean(vals)
                sigma = np.std(vals, ddof=1)
                cv = (sigma / mu) if mu > 1e-6 else 0.0
                cv_list.append(cv)
            else:
                cv_list.append(0.0)
        
        if not cv_list:
            continue
        
        episodes_arr = np.array(episodes)
        cv_arr = _maybe_smooth(np.array(cv_list), smooth)
        
        ax.plot(episodes_arr, cv_arr, color=color, linewidth=2.5,
                label=f'size={size}', alpha=0.85)
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (CV = σ/μ)', fontsize=13, fontweight='bold')
    ax.set_title('LOSS: CV Evolution (0-2000 episodes, zoomed)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, max_ep)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, linewidth=1.5,
              label='CV=0.1 (low variability)')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, linewidth=1.5,
              label='CV=0.3 (moderate)')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
              label='CV=0.5 (high)')
    
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'loss_3_coefficient_of_variation_zoom.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
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
    ap.add_argument('--tag', type=str, default=None, help='Tag subdirectory name (e.g., rb122k)')
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
        ('td_mean', 'TD Error Mean', None),
        ('td_std', 'TD Error Std', None),
    ]:
        # Use 'median' alignment for main plots - balances data coverage and representativeness
        aggs = aggregate_metric(curves, metric, max_episode=args.max_ep, 
                               align_mode='median', min_rep_threshold=5)
        if aggs:
            path = plot_aggregated_series(aggs, metric, out_dir,
                                          title=f'{metric} vs episode (median-aligned, mean ± std)',
                                          ylabel=ylabel, smooth=args.smooth, ylimit=ylim,
                                          show_sample_count=True, show_std_band=False)
            print(f'[plot] {path}')
        else:
            print(f'[warn] No data for metric: {metric}')
    
    # ===== LOSS增强可视化 (100次重复专用) =====
    print("\n[PHASE 1.5] Plotting ENHANCED LOSS visualizations (100 runs per size)...")
    print("  这些图表专门设计用于展示100次重复实验的loss走势:")
    
    print("  [1/6] 均值 + 多层置信区间 (general view)...")
    p1 = plot_loss_enhanced_1_multilevel_ci(curves, out_dir, args.max_ep, args.smooth)
    print(f'    ✓ {p1}')
    
    print("  [2/6] 均值 + 多层置信区间 (0-2000 episodes, zoomed)...")
    p1z = plot_loss_enhanced_1_multilevel_ci_zoom(curves, out_dir, max_ep=2000, smooth=args.smooth)
    print(f'    ✓ {p1z}')
    
    print("  [3/6] 分位数带演化 (general view)...")
    p2 = plot_loss_enhanced_2_quantile_bands(curves, out_dir, args.max_ep, args.smooth)
    print(f'    ✓ {p2}')
    
    print("  [4/6] 分位数带演化 (0-2000 episodes, zoomed)...")
    p2z = plot_loss_enhanced_2_quantile_bands_zoom(curves, out_dir, max_ep=2000, smooth=args.smooth)
    print(f'    ✓ {p2z}')
    
    print("  [5/6] 变异系数演化 (general view)...")
    p3 = plot_loss_enhanced_3_cv_evolution(curves, out_dir, args.max_ep, smooth=10)
    print(f'    ✓ {p3}')
    
    print("  [6/6] 变异系数演化 (0-2000 episodes, zoomed)...")
    p3z = plot_loss_enhanced_3_cv_evolution_zoom(curves, out_dir, max_ep=2000, smooth=10)
    print(f'    ✓ {p3z}')

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
    
    # ===== 数据质量诊断报告 =====
    print("\n[PHASE 6] Data Quality Diagnostics...")
    
    # 生成episode分布统计
    print("\n" + "="*70)
    print("EPISODE DISTRIBUTION BY BUFFER SIZE")
    print("="*70)
    
    by_size_episodes = defaultdict(list)
    for r in curves:
        # Get max episode from any available metric
        max_ep_in_run = 0
        for metric_name in ['success_rate', 'loss', 'reward']:
            series = r.get(metric_name)
            if series:
                max_ep_in_run = max(max_ep_in_run, max(ep for ep, _ in series))
        if max_ep_in_run > 0:
            by_size_episodes[r.size].append(max_ep_in_run)
    
    for size in sorted(by_size_episodes.keys()):
        eps_list = sorted(by_size_episodes[size])
        if not eps_list:
            continue
        
        print(f"\n  Buffer Size = {size}:")
        print(f"    Total runs: {len(eps_list)}")
        print(f"    Min episodes: {min(eps_list)}")
        print(f"    Q1 (25th): {int(np.percentile(eps_list, 25))}")
        print(f"    Median (50th): {int(np.median(eps_list))}")
        print(f"    Q3 (75th): {int(np.percentile(eps_list, 75))}")
        print(f"    Max episodes: {max(eps_list)}")
        print(f"    Mean: {np.mean(eps_list):.1f} ± {np.std(eps_list):.1f}")
        
        # 统计特殊情况
        early_stop = sum(1 for e in eps_list if e < 1000)
        max_out = sum(1 for e in eps_list if e >= 9999)
        mid_range = len(eps_list) - early_stop - max_out
        
        print(f"    Distribution:")
        print(f"      Early converged (<1000 ep): {early_stop} ({100*early_stop/len(eps_list):.1f}%)")
        print(f"      Normal range (1000-9999 ep): {mid_range} ({100*mid_range/len(eps_list):.1f}%)")
        print(f"      Maxed out (>=10000 ep): {max_out} ({100*max_out/len(eps_list):.1f}%)")
    
    # 绘制episode分布对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 左图：箱线图
    sizes = sorted(by_size_episodes.keys())
    data_for_box = [by_size_episodes[s] for s in sizes]
    
    bp = ax1.boxplot(data_for_box, labels=[str(s) for s in sizes], patch_artist=True)
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    ax1.set_xlabel('Replay Buffer Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Episodes per Run', fontsize=12, fontweight='bold')
    ax1.set_title('Episode Distribution by Buffer Size', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label='1000 ep threshold')
    ax1.legend()
    
    # 右图：直方图堆叠
    for i, size in enumerate(sizes):
        eps_list = by_size_episodes[size]
        ax2.hist(eps_list, bins=20, alpha=0.6, label=f'size={size}', 
                color=COLORS[i % len(COLORS)], edgecolor='black')
    
    ax2.set_xlabel('Episodes per Run', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Episode Distribution Histogram', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    diag_path = os.path.join(out_dir, 'diagnostic_episode_distribution.png')
    plt.savefig(diag_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\n[plot] {diag_path}')
    
    # ===== 数据质量诊断报告 =====
    print("\n" + "="*70)
    print("EPISODE DISTRIBUTION DIAGNOSTICS")
    print("="*70)
    
    by_size_episodes = defaultdict(list)
    for r in curves:
        # Get max episode from any available metric
        max_ep_in_run = 0
        for metric_name in ['success_rate', 'loss', 'reward']:
            series = r.get(metric_name)
            if series:
                max_ep_in_run = max(max_ep_in_run, max(ep for ep, _ in series))
        if max_ep_in_run > 0:
            by_size_episodes[r.size].append(max_ep_in_run)
    
    for size in sorted(by_size_episodes.keys()):
        eps_list = sorted(by_size_episodes[size])
        if not eps_list:
            continue
        
        print(f"\nBuffer Size = {size}:")
        print(f"  Runs: {len(eps_list)}")
        print(f"  Min: {min(eps_list)}, Median: {int(np.median(eps_list))}, Max: {max(eps_list)}")
        print(f"  Q1: {int(np.percentile(eps_list, 25))}, Q3: {int(np.percentile(eps_list, 75))}")
        
        # Count distribution
        early = sum(1 for e in eps_list if e < 1000)
        maxed = sum(1 for e in eps_list if e >= 9999)
        print(f"  Early converged (<1000 ep): {early} ({100*early/len(eps_list):.1f}%)")
        print(f"  Maxed out (≥10000 ep): {maxed} ({100*maxed/len(eps_list):.1f}%)")
    
    # 最终总结
    print("\n" + "="*70)
    print("[SUMMARY] All research plots generated successfully!")
    print("="*70)
    print("\nKEY IMPROVEMENTS:")
    print("  1. Median alignment: Plots truncated to median run length per buffer size")
    print("  2. Sample count indicators: Shows n=start→end at curve endpoints")
    print("  3. Data quality diagnostics: Episode distribution analysis above")
    print("  4. Conservative thresholds: min_rep_threshold=5 to ensure reliability")
    print("\nIMPORTANT NOTES:")
    print("  - X-axis now shows median episode length (more representative)")
    print("  - Curves with n=1 at tail are excluded (unreliable single-run data)")
    print("  - Legend format: 'size=X (n=50→25)' shows sample count drop")
    print("\n" + "="*70)
    print("LOSS可视化推荐 (100次重复实验)")
    print("="*70)
    print("\n推荐使用顺序:")
    print("  1. loss_1_mean_multilevel_ci.png")
    print("     → 论文主图，展示均值趋势 + 68%/95%置信区间")
    print("     → 直观显示不确定性，适合对比不同buffer size")
    print("\n  2. loss_2_quantile_bands.png")
    print("     → 补充图，使用分位数而非标准差")
    print("     → 不受极端值影响，更稳健")
    print("\n  3. loss_3_coefficient_of_variation.png")
    print("     → 分析相对变异性 (CV = σ/μ)")
    print("     → 展示哪个buffer size训练最稳定，完整显示所有数据")
    print("\n关键统计指标:")
    print("  • ±1σ: 包含约68%的数据点 (正态分布假设)")
    print("  • ±2σ: 包含约95%的数据点 (95%置信区间)")
    print("  • IQR (25-75%): 稳健的中间50%数据")
    print("  • CV < 0.1: 低变异性 (高度稳定)")
    print("  • CV > 0.5: 高变异性 (训练不稳定)")
    print("\nRECOMMENDATIONS FOR YOUR PAPER:")
    print("  - Use median-aligned plots for main results (fair comparison)")
    print("  - Report episode distribution statistics in methodology")
    print("  - For LOSS: Use multilevel CI plot as main figure")
    print("  - Mention: 'Shaded regions show 68% (darker) and 95% (lighter) confidence intervals'")
    print("  - Consider excluding size=100000 if instability is too high")
    print("="*70)

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
