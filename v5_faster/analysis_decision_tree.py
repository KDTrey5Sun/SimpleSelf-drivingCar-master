#!/usr/bin/env python3
"""
决策树示例：如何综合判断配置是否稳定且性能好

Usage:
    python analysis_decision_tree.py --csv sweep_summary.csv
"""
from typing import Dict, List, Tuple

def classify_configuration(mean: float, std: float, cv: float,
                          mean_threshold: float = 0.7,
                          cv_threshold: float = 0.1) -> Dict[str, str]:
    """
    综合分类函数：避免"低均值+低CoV"陷阱
    
    Args:
        mean: 性能均值
        std: 标准差
        cv: 变异系数 (CoV = std / mean)
        mean_threshold: 性能阈值（默认0.7）
        cv_threshold: 稳定性阈值（默认0.1）
    
    Returns:
        分类结果字典
    """
    high_performance = mean >= mean_threshold
    high_stability = cv <= cv_threshold
    
    if high_performance and high_stability:
        category = 'optimal'
        icon = '✓✓'
        description = 'Optimal: High Performance + High Stability'
        recommendation = '推荐使用 - 适合生产环境部署'
        color = 'green'
    
    elif high_performance and not high_stability:
        category = 'unstable'
        icon = '✓✗'
        description = 'Unstable: High Performance but Variable'
        recommendation = '谨慎使用 - 需要多次运行或集成方法'
        color = 'yellow'
    
    elif not high_performance and high_stability:
        category = 'poor_stable'
        icon = '✗✓'
        description = 'Poor (Stable): Consistent but Inadequate'
        recommendation = '不推荐 - 稳定但性能不足（陷阱配置）'
        color = 'orange'
    
    else:
        category = 'unusable'
        icon = '✗✗'
        description = 'Unusable: Low Performance + High Variance'
        recommendation = '淘汰 - 既不稳定也不高效'
        color = 'red'
    
    return {
        'category': category,
        'icon': icon,
        'description': description,
        'recommendation': recommendation,
        'color': color,
        'metrics': {
            'mean': f'{mean:.3f}',
            'std': f'{std:.3f}',
            'cv': f'{cv:.3f}',
            'high_performance': high_performance,
            'high_stability': high_stability
        }
    }


def analyze_buffer_size(buffer_size: int, success_rates: List[float],
                       mean_threshold: float = 0.7,
                       cv_threshold: float = 0.1) -> None:
    """
    分析单个 buffer size 的稳定性
    
    Args:
        buffer_size: 缓冲区大小
        success_rates: 成功率列表（多次重复）
        mean_threshold: 性能阈值
        cv_threshold: 稳定性阈值
    """
    import numpy as np
    
    mean = np.mean(success_rates)
    std = np.std(success_rates, ddof=1)
    cv = std / mean if mean > 1e-9 else 0.0
    
    result = classify_configuration(mean, std, cv, mean_threshold, cv_threshold)
    
    print(f"\n{'='*80}")
    print(f"Buffer Size: {buffer_size}")
    print(f"{'='*80}")
    print(f"Performance Metrics:")
    print(f"  Mean:     {mean:.4f}  {'✓' if mean >= mean_threshold else '✗'} (threshold: {mean_threshold})")
    print(f"  Std:      {std:.4f}")
    print(f"  CoV:      {cv:.4f}  {'✓' if cv <= cv_threshold else '✗'} (threshold: {cv_threshold})")
    print(f"\nClassification: {result['icon']} {result['category'].upper()}")
    print(f"Description:    {result['description']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"{'='*80}\n")


def multi_buffer_comparison(configs: Dict[int, List[float]]) -> None:
    """
    对比多个 buffer size 配置
    
    Args:
        configs: {buffer_size: [success_rates]}
    """
    import numpy as np
    
    print("\n" + "="*100)
    print("COMPREHENSIVE BUFFER SIZE COMPARISON")
    print("="*100)
    
    results = []
    for size, rates in sorted(configs.items()):
        mean = np.mean(rates)
        std = np.std(rates, ddof=1)
        cv = std / mean if mean > 1e-9 else 0.0
        result = classify_configuration(mean, std, cv)
        results.append((size, mean, std, cv, result))
    
    # 表格头
    print(f"\n{'Size':<10} {'Mean':<10} {'Std':<10} {'CoV':<10} {'Category':<20} {'Icon':<6} {'Recommendation'}")
    print("-"*100)
    
    # 表格内容
    for size, mean, std, cv, result in results:
        print(f"{size:<10} {mean:<10.4f} {std:<10.4f} {cv:<10.4f} "
              f"{result['category']:<20} {result['icon']:<6} {result['recommendation']}")
    
    print("\n" + "="*100)
    
    # 推荐配置
    optimal_configs = [(size, mean, cv) for size, mean, std, cv, r in results 
                      if r['category'] == 'optimal']
    
    if optimal_configs:
        print("\n🎯 RECOMMENDED CONFIGURATIONS (Optimal Zone):")
        for size, mean, cv in optimal_configs:
            print(f"  • Buffer Size {size:>6}: mean={mean:.3f}, CoV={cv:.3f}")
    
    # 陷阱配置警告
    poor_stable = [(size, mean, cv) for size, mean, std, cv, r in results 
                   if r['category'] == 'poor_stable']
    
    if poor_stable:
        print("\n⚠️  WARNING: Stable but Poor Performance (Trap Configurations):")
        for size, mean, cv in poor_stable:
            print(f"  • Buffer Size {size:>6}: mean={mean:.3f}, CoV={cv:.3f} "
                  f"- Avoid using despite low CoV!")
    
    print("\n" + "="*100)


# 示例数据：模拟不同场景
if __name__ == '__main__':
    print("\n" + "="*100)
    print("STABILITY ANALYSIS DECISION TREE - DEMONSTRATION")
    print("="*100)
    
    # 场景1: 低均值 + 低CoV（陷阱）
    print("\n\n📌 Scenario 1: Low Mean + Low CoV (TRAP)")
    analyze_buffer_size(1000, [0.10, 0.12, 0.11, 0.09, 0.10, 0.11])
    
    # 场景2: 高均值 + 低CoV（最优）
    print("\n📌 Scenario 2: High Mean + Low CoV (OPTIMAL)")
    analyze_buffer_size(10000, [0.82, 0.85, 0.84, 0.83, 0.86, 0.84])
    
    # 场景3: 高均值 + 高CoV（不稳定）
    print("\n📌 Scenario 3: High Mean + High CoV (UNSTABLE)")
    analyze_buffer_size(100000, [0.65, 0.90, 0.75, 0.88, 0.70, 0.92])
    
    # 场景4: 低均值 + 高CoV（最差）
    print("\n📌 Scenario 4: Low Mean + High CoV (UNUSABLE)")
    analyze_buffer_size(500, [0.30, 0.55, 0.20, 0.45, 0.35, 0.25])
    
    # 多配置对比
    print("\n\n" + "="*100)
    print("MULTI-CONFIGURATION COMPARISON")
    print("="*100)
    
    configs = {
        1000: [0.42, 0.45, 0.40, 0.43, 0.44, 0.41],  # 低性能但稳定（陷阱）
        5000: [0.72, 0.75, 0.74, 0.71, 0.73, 0.72],  # 及格且稳定
        10000: [0.82, 0.85, 0.84, 0.83, 0.86, 0.84], # 最优
        30000: [0.88, 0.90, 0.87, 0.89, 0.88, 0.90], # 最优
        100000: [0.65, 0.90, 0.75, 0.88, 0.70, 0.92] # 高性能但不稳定
    }
    
    multi_buffer_comparison(configs)
    
    # 关键要点总结
    print("\n\n" + "="*100)
    print("KEY TAKEAWAYS")
    print("="*100)
    print("""
1. ❌ NEVER judge stability by CoV alone
   - Low CoV with low mean = "Consistently failing" (TRAP)
   - Must check if mean meets performance threshold first

2. ✅ Use multi-dimensional evaluation:
   - Step 1: Check mean ≥ threshold (e.g., 0.7)
   - Step 2: Check CoV ≤ threshold (e.g., 0.1)
   - Step 3: Classify into 4 quadrants

3. 🎯 Optimal configurations must satisfy BOTH:
   - High absolute performance (mean ≥ 0.7)
   - High relative stability (CoV ≤ 0.1)

4. 📊 Recommended visualization order:
   a) Pareto classification plot → Identify optimal zone
   b) Dual-axis plot → Verify threshold crossings
   c) Boxplot → Check for outliers
   d) Bar chart → Compare absolute performance

5. 📝 Paper reporting checklist:
   ✓ Report mean AND CoV together
   ✓ Specify performance threshold
   ✓ Classify configurations explicitly
   ✓ Avoid claiming "stable" without performance context
    """)
    print("="*100)
