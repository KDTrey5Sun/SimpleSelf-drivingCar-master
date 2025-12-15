#!/usr/bin/env python3
"""
RQ2 结果可视化脚本

生成以下图表来回答研究问题：
- SRQ2.1: 不同批量大小下缓冲区性能排名的一致性
- SRQ2.2: 批量大小与缓冲区大小的交互效应

使用方法:
    python plot_rq2_results.py <csv_path> [output_dir]
    
示例:
    python plot_rq2_results.py v5_faster/v5_exp_data/replay_grid/rb133k/grid_summary.csv
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from scipy.stats import f as f_dist

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


def load_data(csv_path):
    """加载并验证数据"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ 加载数据: {len(df)} 行")
    print(f"  缓冲区大小: {sorted(df['buffer_size'].unique())}")
    print(f"  批量大小: {sorted(df['batch_size'].unique())}")
    print(f"  重复次数: {df.groupby(['buffer_size', 'batch_size']).size().min()}-{df.groupby(['buffer_size', 'batch_size']).size().max()}")
    
    return df


def plot_performance_heatmap(df, metric, output_dir):
    """
    图1: 性能热图 (Buffer Size × Batch Size)
    用于直观展示不同组合下的性能
    """
    # 计算每个组合的平均性能
    pivot = df.groupby(['buffer_size', 'batch_size'])[metric].mean().unstack()
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制热图
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    # 标注数值
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax.text(j, i, f'{pivot.values[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    # 设置标签
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Buffer Size', fontsize=12, fontweight='bold')
    
    # 根据指标设置标题
    metric_names = {
        'success_rate': 'Success Rate',
        'episodes_to_convergence': 'Episodes to First Success',
        'samples_to_convergence': 'Samples to First Success',
        'avg_score': 'Average Score',
        'loss_variance_last100': 'Loss Variance (Last 100 Steps)'
    }
    title = f'Performance Heatmap: {metric_names.get(metric, metric)}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_names.get(metric, metric), rotation=270, labelpad=20)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'heatmap_{metric}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  → 保存: {output_path}")


def plot_ranking_consistency(df, metric, output_dir):
    """
    图2: 排名一致性图 (SRQ2.1)
    展示不同批量大小下缓冲区的排名
    """
    # 计算每个批量大小下的排名
    pivot = df.groupby(['buffer_size', 'batch_size'])[metric].mean().unstack()
    rankings = {}
    for col in pivot.columns:
        # 降序排名（值越大排名越高）
        if metric in ['episodes_to_convergence', 'samples_to_convergence', 'loss_variance_last100']:
            # 这些指标越小越好
            rankings[col] = pivot[col].rank(ascending=True, method='average')
        else:
            # 这些指标越大越好
            rankings[col] = pivot[col].rank(ascending=False, method='average')
    
    # 创建排名数据框
    rank_df = pd.DataFrame(rankings)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 为每个批量大小绘制线条
    colors = plt.cm.tab10(np.linspace(0, 1, len(rank_df.columns)))
    for idx, col in enumerate(rank_df.columns):
        ax.plot(range(len(rank_df)), rank_df[col], 
                marker='o', linewidth=2, markersize=8,
                label=f'Batch {col}', color=colors[idx])
    
    # 设置坐标轴
    ax.set_xticks(range(len(rank_df)))
    ax.set_xticklabels(rank_df.index)
    ax.set_xlabel('Buffer Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rank (1=Best)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()  # 反转y轴，使1在顶部
    
    # 设置标题
    metric_names = {
        'success_rate': 'Success Rate',
        'episodes_to_convergence': 'Episodes to First Success',
        'samples_to_convergence': 'Samples to First Success',
        'avg_score': 'Average Score',
        'loss_variance_last100': 'Loss Variance (Last 100 Steps)'
    }
    title = f'Ranking Consistency: {metric_names.get(metric, metric)}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例
    ax.legend(loc='best', frameon=True, shadow=True)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'ranking_{metric}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  → 保存: {output_path}")


def plot_rank_correlation_matrix(df, metric, output_dir):
    """
    图3: 排名相关性矩阵 (SRQ2.1)
    展示不同批量大小间的 Spearman 相关系数
    """
    # 计算每个批量大小下的排名
    pivot = df.groupby(['buffer_size', 'batch_size'])[metric].mean().unstack()
    rankings = {}
    for col in pivot.columns:
        if metric in ['episodes_to_convergence', 'samples_to_convergence', 'loss_variance_last100']:
            rankings[col] = pivot[col].rank(ascending=True, method='average')
        else:
            rankings[col] = pivot[col].rank(ascending=False, method='average')
    
    rank_df = pd.DataFrame(rankings)
    
    # 计算 Spearman 相关矩阵
    corr_matrix = rank_df.corr(method='spearman')
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 绘制热图
    im = ax.imshow(corr_matrix.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels([f'Batch {c}' for c in corr_matrix.columns])
    ax.set_yticklabels([f'Batch {c}' for c in corr_matrix.index])
    
    # 标注数值
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            text = ax.text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                          ha="center", va="center", 
                          color="white" if abs(corr_matrix.values[i, j]) > 0.5 else "black",
                          fontsize=11, fontweight='bold')
    
    # 设置标题
    metric_names = {
        'success_rate': 'Success Rate',
        'episodes_to_convergence': 'Episodes to First Success',
        'samples_to_convergence': 'Samples to First Success',
        'avg_score': 'Average Score',
        'loss_variance_last100': 'Loss Variance (Last 100 Steps)'
    }
    
    # 计算平均相关系数
    triu_indices = np.triu_indices_from(corr_matrix.values, k=1)
    mean_corr = corr_matrix.values[triu_indices].mean()
    
    title = f'Rank Correlation Matrix: {metric_names.get(metric, metric)}\nMean Spearman ρ = {mean_corr:.3f}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spearman ρ', rotation=270, labelpad=20)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'correlation_{metric}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  → 保存: {output_path}")


def plot_interaction_effects(df, metric, output_dir):
    """
    图4: 交互效应图 (SRQ2.2)
    展示批量大小对缓冲区效应的调节作用
    """
    # 计算每个组合的平均性能和标准误差
    grouped = df.groupby(['buffer_size', 'batch_size'])[metric].agg(['mean', 'sem']).reset_index()
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 为每个批量大小绘制线条
    batch_sizes = sorted(grouped['batch_size'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(batch_sizes)))
    
    for idx, bs in enumerate(batch_sizes):
        data = grouped[grouped['batch_size'] == bs].sort_values('buffer_size')
        ax.plot(range(len(data)), data['mean'], 
                marker='o', linewidth=2.5, markersize=10,
                label=f'Batch {bs}', color=colors[idx])
        
        # 添加误差带
        ax.fill_between(range(len(data)),
                        data['mean'] - data['sem'],
                        data['mean'] + data['sem'],
                        alpha=0.2, color=colors[idx])
    
    # 设置坐标轴
    buffer_sizes = sorted(grouped['buffer_size'].unique())
    ax.set_xticks(range(len(buffer_sizes)))
    ax.set_xticklabels(buffer_sizes)
    ax.set_xlabel('Buffer Size', fontsize=12, fontweight='bold')
    
    metric_names = {
        'success_rate': 'Success Rate',
        'episodes_to_convergence': 'Episodes to First Success',
        'samples_to_convergence': 'Samples to First Success',
        'avg_score': 'Average Score',
        'loss_variance_last100': 'Loss Variance (Last 100 Steps)'
    }
    ax.set_ylabel(metric_names.get(metric, metric), fontsize=12, fontweight='bold')
    
    # 设置标题
    title = f'Interaction Effects: {metric_names.get(metric, metric)}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例
    ax.legend(loc='best', frameon=True, shadow=True, ncol=2)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 判断是否存在交叉（强交互的迹象）
    # 如果线条交叉，说明存在交互效应
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'interaction_{metric}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  → 保存: {output_path}")


def plot_anova_summary(df, metrics, output_dir):
    """
    图5: ANOVA 效应大小汇总 (SRQ2.2)
    对比不同指标的主效应和交互效应
    """
    results = []
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        df_metric = df[['buffer_size', 'batch_size', metric]].dropna()
        if len(df_metric) < 10:
            continue
        
        # 计算 ANOVA
        group_means = df_metric.groupby(['buffer_size', 'batch_size'])[metric].mean()
        grand_mean = df_metric[metric].mean()
        n_total = len(df_metric)
        
        # SS Total
        ss_total = ((df_metric[metric] - grand_mean) ** 2).sum()
        
        # SS Buffer
        buffer_means = df_metric.groupby('buffer_size')[metric].mean()
        buffer_counts = df_metric.groupby('buffer_size').size()
        ss_buffer = sum(buffer_counts * (buffer_means - grand_mean) ** 2)
        
        # SS Batch
        batch_means = df_metric.groupby('batch_size')[metric].mean()
        batch_counts = df_metric.groupby('batch_size').size()
        ss_batch = sum(batch_counts * (batch_means - grand_mean) ** 2)
        
        # SS Interaction
        ss_cells = sum(
            df_metric.groupby(['buffer_size', 'batch_size']).size() * 
            (group_means - grand_mean) ** 2
        )
        ss_interaction = ss_cells - ss_buffer - ss_batch
        
        # 计算 η²
        eta2_buffer = ss_buffer / ss_total if ss_total > 0 else 0
        eta2_batch = ss_batch / ss_total if ss_total > 0 else 0
        eta2_interaction = ss_interaction / ss_total if ss_total > 0 else 0
        
        results.append({
            'metric': metric,
            'eta2_buffer': eta2_buffer,
            'eta2_batch': eta2_batch,
            'eta2_interaction': eta2_interaction
        })
    
    if not results:
        print("  ⚠ 无足够数据进行 ANOVA 分析")
        return
    
    results_df = pd.DataFrame(results)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 设置柱状图参数
    x = np.arange(len(results_df))
    width = 0.25
    
    metric_names = {
        'success_rate': 'Success\nRate',
        'episodes_to_convergence': 'Episodes to\nConvergence',
        'samples_to_convergence': 'Samples to\nConvergence',
        'avg_score': 'Average\nScore',
        'loss_variance_last100': 'Loss\nVariance'
    }
    
    # 绘制柱状图
    bars1 = ax.bar(x - width, results_df['eta2_buffer'], width, 
                   label='Buffer Size (Main Effect)', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x, results_df['eta2_batch'], width,
                   label='Batch Size (Main Effect)', color='#A23B72', alpha=0.8)
    bars3 = ax.bar(x + width, results_df['eta2_interaction'], width,
                   label='Buffer × Batch (Interaction)', color='#F18F01', alpha=0.8)
    
    # 添加 η² = 0.1 参考线
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='η² = 0.1 (H2.2 threshold)')
    
    # 设置坐标轴
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Effect Size (η²)', fontsize=12, fontweight='bold')
    ax.set_title('ANOVA Effect Sizes: Main Effects vs. Interaction', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_names.get(m, m) for m in results_df['metric']])
    ax.legend(loc='upper right', frameon=True, shadow=True)
    
    # 添加网格
    ax.grid(True, alpha=0.3, axis='y')
    
    # 标注数值
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'anova_summary.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  → 保存: {output_path}")


def plot_hypothesis_support_summary(df, metrics, output_dir):
    """
    图6: 假设支持度汇总
    可视化 H2.1 和 H2.2 的支持情况
    """
    h21_support = []
    h22_support = []
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        # H2.1: 排名相关性
        pivot = df.groupby(['buffer_size', 'batch_size'])[metric].mean().unstack()
        rankings = {}
        for col in pivot.columns:
            if metric in ['episodes_to_convergence', 'samples_to_convergence', 'loss_variance_last100']:
                rankings[col] = pivot[col].rank(ascending=True, method='average')
            else:
                rankings[col] = pivot[col].rank(ascending=False, method='average')
        
        rank_df = pd.DataFrame(rankings)
        corr_matrix = rank_df.corr(method='spearman')
        triu_indices = np.triu_indices_from(corr_matrix.values, k=1)
        mean_rho = corr_matrix.values[triu_indices].mean()
        
        h21_support.append('YES' if mean_rho > 0.7 else 'PARTIAL' if mean_rho > 0.5 else 'NO')
        
        # H2.2: 交互效应
        df_metric = df[['buffer_size', 'batch_size', metric]].dropna()
        if len(df_metric) >= 10:
            group_means = df_metric.groupby(['buffer_size', 'batch_size'])[metric].mean()
            grand_mean = df_metric[metric].mean()
            
            ss_total = ((df_metric[metric] - grand_mean) ** 2).sum()
            buffer_means = df_metric.groupby('buffer_size')[metric].mean()
            buffer_counts = df_metric.groupby('buffer_size').size()
            ss_buffer = sum(buffer_counts * (buffer_means - grand_mean) ** 2)
            
            batch_means = df_metric.groupby('batch_size')[metric].mean()
            batch_counts = df_metric.groupby('batch_size').size()
            ss_batch = sum(batch_counts * (batch_means - grand_mean) ** 2)
            
            ss_cells = sum(
                df_metric.groupby(['buffer_size', 'batch_size']).size() * 
                (group_means - grand_mean) ** 2
            )
            ss_interaction = ss_cells - ss_buffer - ss_batch
            eta2_interaction = ss_interaction / ss_total if ss_total > 0 else 0
            
            h22_support.append('YES' if eta2_interaction < 0.1 else 'PARTIAL' if eta2_interaction < 0.14 else 'NO')
        else:
            h22_support.append('N/A')
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    metric_names = {
        'success_rate': 'Success\nRate',
        'episodes_to_convergence': 'Episodes to\nConvergence',
        'samples_to_convergence': 'Samples to\nConvergence',
        'avg_score': 'Average\nScore',
        'loss_variance_last100': 'Loss\nVariance'
    }
    
    # H2.1 支持度
    colors_h21 = ['green' if s == 'YES' else 'orange' if s == 'PARTIAL' else 'red' for s in h21_support]
    x = np.arange(len(metrics))
    bars1 = ax1.bar(x, [1]*len(metrics), color=colors_h21, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([metric_names.get(m, m) for m in metrics])
    ax1.set_ylabel('Support', fontsize=12, fontweight='bold')
    ax1.set_title('H2.1: Ranking Consistency\n(Spearman ρ > 0.7)', fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylim(0, 1.2)
    ax1.set_yticks([])
    
    for i, (bar, support) in enumerate(zip(bars1, h21_support)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, support,
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # H2.2 支持度
    colors_h22 = ['green' if s == 'YES' else 'orange' if s == 'PARTIAL' else 'red' if s == 'NO' else 'gray' 
                  for s in h22_support]
    bars2 = ax2.bar(x, [1]*len(metrics), color=colors_h22, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([metric_names.get(m, m) for m in metrics])
    ax2.set_ylabel('Support', fontsize=12, fontweight='bold')
    ax2.set_title('H2.2: Weak Interaction\n(η² < 0.1)', fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([])
    
    for i, (bar, support) in enumerate(zip(bars2, h22_support)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, support,
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='YES (Supported)'),
        Patch(facecolor='orange', alpha=0.7, label='PARTIAL (Partially Supported)'),
        Patch(facecolor='red', alpha=0.7, label='NO (Not Supported)'),
        Patch(facecolor='gray', alpha=0.7, label='N/A (Insufficient Data)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=4, frameon=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    output_path = os.path.join(output_dir, 'hypothesis_support.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  → 保存: {output_path}")


def generate_summary_report(df, metrics, output_dir):
    """生成文本摘要报告"""
    report = []
    report.append("="*80)
    report.append("RQ2: 批量大小敏感性分析 - 结果摘要")
    report.append("="*80)
    report.append("")
    
    # 数据概览
    buffer_sizes = sorted(df['buffer_size'].unique())
    batch_sizes = sorted(df['batch_size'].unique())
    
    report.append("【数据概览】")
    report.append(f"  总运行次数:     {len(df)}")
    report.append(f"  缓冲区大小:     {buffer_sizes}")
    report.append(f"  批量大小:       {batch_sizes}")
    report.append(f"  重复次数:       {df.groupby(['buffer_size', 'batch_size']).size().min()}-{df.groupby(['buffer_size', 'batch_size']).size().max()}")
    report.append("")
    
    # SRQ2.1 结果
    report.append("【SRQ2.1: 性能趋势一致性】")
    report.append("假设 H2.1: 缓冲区排名在不同批量下保持一致 (Spearman ρ > 0.7)")
    report.append("")
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        pivot = df.groupby(['buffer_size', 'batch_size'])[metric].mean().unstack()
        rankings = {}
        for col in pivot.columns:
            if metric in ['episodes_to_convergence', 'samples_to_convergence', 'loss_variance_last100']:
                rankings[col] = pivot[col].rank(ascending=True, method='average')
            else:
                rankings[col] = pivot[col].rank(ascending=False, method='average')
        
        rank_df = pd.DataFrame(rankings)
        corr_matrix = rank_df.corr(method='spearman')
        triu_indices = np.triu_indices_from(corr_matrix.values, k=1)
        mean_rho = corr_matrix.values[triu_indices].mean()
        
        support = '✓ 支持' if mean_rho > 0.7 else '△ 部分支持' if mean_rho > 0.5 else '✗ 不支持'
        
        report.append(f"  {metric}:")
        report.append(f"    平均 Spearman ρ = {mean_rho:.3f}")
        report.append(f"    H2.1 状态: {support}")
        report.append("")
    
    # SRQ2.2 结果
    report.append("【SRQ2.2: 交互效应强度】")
    report.append("假设 H2.2: 批量与缓冲区仅弱交互 (η² < 0.1)")
    report.append("")
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        df_metric = df[['buffer_size', 'batch_size', metric]].dropna()
        if len(df_metric) < 10:
            report.append(f"  {metric}: 数据不足")
            report.append("")
            continue
        
        group_means = df_metric.groupby(['buffer_size', 'batch_size'])[metric].mean()
        grand_mean = df_metric[metric].mean()
        
        ss_total = ((df_metric[metric] - grand_mean) ** 2).sum()
        
        buffer_means = df_metric.groupby('buffer_size')[metric].mean()
        buffer_counts = df_metric.groupby('buffer_size').size()
        ss_buffer = sum(buffer_counts * (buffer_means - grand_mean) ** 2)
        
        batch_means = df_metric.groupby('batch_size')[metric].mean()
        batch_counts = df_metric.groupby('batch_size').size()
        ss_batch = sum(batch_counts * (batch_means - grand_mean) ** 2)
        
        ss_cells = sum(
            df_metric.groupby(['buffer_size', 'batch_size']).size() * 
            (group_means - grand_mean) ** 2
        )
        ss_interaction = ss_cells - ss_buffer - ss_batch
        
        eta2_buffer = ss_buffer / ss_total if ss_total > 0 else 0
        eta2_batch = ss_batch / ss_total if ss_total > 0 else 0
        eta2_interaction = ss_interaction / ss_total if ss_total > 0 else 0
        
        support = '✓ 支持' if eta2_interaction < 0.1 else '△ 部分支持' if eta2_interaction < 0.14 else '✗ 不支持'
        
        report.append(f"  {metric}:")
        report.append(f"    缓冲区主效应 η² = {eta2_buffer:.4f}")
        report.append(f"    批量主效应 η²   = {eta2_batch:.4f}")
        report.append(f"    交互效应 η²     = {eta2_interaction:.4f}")
        report.append(f"    H2.2 状态: {support}")
        report.append("")
    
    # 总结
    report.append("【总体结论】")
    
    h21_yes = 0
    h22_yes = 0
    total = 0
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        # H2.1
        pivot = df.groupby(['buffer_size', 'batch_size'])[metric].mean().unstack()
        rankings = {}
        for col in pivot.columns:
            if metric in ['episodes_to_convergence', 'samples_to_convergence', 'loss_variance_last100']:
                rankings[col] = pivot[col].rank(ascending=True, method='average')
            else:
                rankings[col] = pivot[col].rank(ascending=False, method='average')
        
        rank_df = pd.DataFrame(rankings)
        corr_matrix = rank_df.corr(method='spearman')
        triu_indices = np.triu_indices_from(corr_matrix.values, k=1)
        mean_rho = corr_matrix.values[triu_indices].mean()
        
        if mean_rho > 0.7:
            h21_yes += 1
        
        # H2.2
        df_metric = df[['buffer_size', 'batch_size', metric]].dropna()
        if len(df_metric) >= 10:
            group_means = df_metric.groupby(['buffer_size', 'batch_size'])[metric].mean()
            grand_mean = df_metric[metric].mean()
            
            ss_total = ((df_metric[metric] - grand_mean) ** 2).sum()
            buffer_means = df_metric.groupby('buffer_size')[metric].mean()
            buffer_counts = df_metric.groupby('buffer_size').size()
            ss_buffer = sum(buffer_counts * (buffer_means - grand_mean) ** 2)
            
            batch_means = df_metric.groupby('batch_size')[metric].mean()
            batch_counts = df_metric.groupby('batch_size').size()
            ss_batch = sum(batch_counts * (batch_means - grand_mean) ** 2)
            
            ss_cells = sum(
                df_metric.groupby(['buffer_size', 'batch_size']).size() * 
                (group_means - grand_mean) ** 2
            )
            ss_interaction = ss_cells - ss_buffer - ss_batch
            eta2_interaction = ss_interaction / ss_total if ss_total > 0 else 0
            
            if eta2_interaction < 0.1:
                h22_yes += 1
            
            total += 1
    
    if total > 0:
        h21_rate = 100 * h21_yes / total
        h22_rate = 100 * h22_yes / total
        
        report.append(f"  H2.1 支持率: {h21_yes}/{total} ({h21_rate:.1f}%)")
        report.append(f"  H2.2 支持率: {h22_yes}/{total} ({h22_rate:.1f}%)")
        report.append("")
        
        if h21_rate >= 80 and h22_rate >= 80:
            report.append("  ✓ 结论: 缓冲区和批量大小的效应是独立的，可分别调优")
        elif h21_rate >= 80:
            report.append("  △ 结论: 排名一致但存在一定交互，需注意效应强度的变化")
        else:
            report.append("  ✗ 结论: 存在强交互，需要联合调优这两个参数")
    
    report.append("")
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    # 保存报告
    report_path = os.path.join(output_dir, 'summary_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\n✓ 报告已保存: {report_path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python plot_rq2_results.py <csv_path> [output_dir]")
        print("\n示例:")
        print("  python plot_rq2_results.py v5_faster/v5_exp_data/replay_grid/rb133k/grid_summary.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(csv_path), 'rq2_plots')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("RQ2 结果可视化")
    print("="*80)
    print(f"数据文件: {csv_path}")
    print(f"输出目录: {output_dir}\n")
    
    # 加载数据
    df = load_data(csv_path)
    
    # 关键指标
    metrics = ['success_rate', 'episodes_to_convergence', 'samples_to_convergence', 
               'avg_score', 'loss_variance_last100']
    
    # 生成图表
    print("\n生成图表...")
    
    for metric in metrics:
        if metric not in df.columns:
            print(f"  ⚠ 跳过 {metric} (列不存在)")
            continue
        
        print(f"\n处理指标: {metric}")
        plot_performance_heatmap(df, metric, output_dir)
        plot_ranking_consistency(df, metric, output_dir)
        plot_rank_correlation_matrix(df, metric, output_dir)
        plot_interaction_effects(df, metric, output_dir)
    
    print("\n生成汇总图表...")
    plot_anova_summary(df, metrics, output_dir)
    plot_hypothesis_support_summary(df, metrics, output_dir)
    
    # 生成文本报告
    print("\n生成摘要报告...")
    generate_summary_report(df, metrics, output_dir)
    
    print("\n" + "="*80)
    print("✓ 所有图表和报告已生成完成")
    print("="*80)
    print(f"\n输出位置: {output_dir}")
    print("\n生成的文件:")
    print("  - heatmap_*.png: 性能热图")
    print("  - ranking_*.png: 排名一致性图")
    print("  - correlation_*.png: 排名相关性矩阵")
    print("  - interaction_*.png: 交互效应图")
    print("  - anova_summary.png: ANOVA 效应大小汇总")
    print("  - hypothesis_support.png: 假设支持度汇总")
    print("  - summary_report.txt: 文本摘要报告")
    print("")


if __name__ == '__main__':
    main()
