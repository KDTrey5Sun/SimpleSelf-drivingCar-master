# 📊 综合稳定性分析指南

## 问题：为何不能单看 CoV？

### 陷阱示例
```python
配置 A: mean=0.12, std=0.01, CoV=0.083  # 低CoV但性能差 ❌
配置 B: mean=0.85, std=0.09, CoV=0.106  # 略高CoV但性能优 ✓
```

**问题根源**：CoV 只衡量**相对变异性**，不考虑**绝对性能水平**。  
**解决方案**：必须联合多种图表综合判断。

---

## ✅ 多维度分析框架

### 1️⃣ 双轴图：Performance vs Stability

**文件位置**：`{metric}_performance_stability_dual_axis.png`

**功能**：
- **左轴（实线+误差棒）**：性能均值 ± 标准差
- **右轴（虚线）**：CoV（变异系数）
- **绿色阈值线**：目标性能（如 0.7）
- **橙色阈值线**：稳定性阈值（CoV = 0.1）

**使用方法**：
```python
# 决策树
if 均值 >= 0.7 AND CoV <= 0.1:
    return "✓✓ 最优配置（高性能 + 高稳定）"
elif 均值 >= 0.7 AND CoV > 0.1:
    return "✓✗ 高性能但不稳定（需权衡）"
elif 均值 < 0.7 AND CoV <= 0.1:
    return "✗✓ 稳定但性能差（无价值）"  # ← 避免误判关键
else:
    return "✗✗ 性能差且不稳定（淘汰）"
```

**论文引用**：
> Figure X shows the dual-axis analysis of buffer sizes. While buffer size 1000 exhibits low CoV (0.08), its mean success rate (0.42) falls below the practical threshold (0.7), classifying it as *stable but inadequate*. In contrast, buffer sizes 10,000-50,000 achieve both high performance (>0.8) and low CoV (<0.1), indicating optimal configurations.

---

### 2️⃣ Pareto前沿四象限图

**文件位置**：`{metric}_pareto_classification.png`

**功能**：
- **横轴**：性能均值
- **纵轴**：CoV
- **四个区域**：
  - 🟢 右下：Optimal（高性能 + 低CoV）
  - 🟡 右上：Unstable（高性能但不稳定）
  - 🟠 左下：Poor (Stable)（稳定但性能差）← **识别陷阱关键**
  - 🔴 左上：Unusable（都不好）

**使用方法**：
1. 查看散点分布：目标是**右下绿色区域**
2. 识别"假稳定"：左下橙色区域的配置应淘汰
3. 颜色编码：散点颜色代表 buffer size（log尺度）

**论文引用**：
> Figure Y presents the Pareto front classification. Buffer size 1000 falls in the *Poor (Stable)* quadrant (mean=0.42, CoV=0.08), demonstrating that low variability alone does not guarantee practical utility. Configurations in the *Optimal* quadrant (sizes 10,000-50,000) satisfy both performance (mean ≥ 0.7) and stability (CoV ≤ 0.1) criteria simultaneously.

---

### 3️⃣ 箱线图（现有）

**文件位置**：`final_success_rate_boxplot.png`

**补充价值**：
- 显示**中位数、四分位数、异常值**
- 识别**偏态分布**（如极端成功/失败）

**联合判断**：
```python
if 中位数 >= 0.7 AND 箱体高度小 AND CoV < 0.1:
    return "稳定且性能好 ✓✓"
elif 中位数 < 0.5 AND 箱体高度小 AND CoV < 0.15:
    return "稳定但性能差（陷阱）✗✓"
```

---

### 4️⃣ 柱状图+误差棒（现有）

**文件位置**：`summary_{metric}_by_size.png`

**补充价值**：
- 直观对比**绝对性能差异**
- 误差棒长度 = 标准差（绝对变异性）

**联合判断**：
- 柱高 → 性能均值
- 误差棒 → 绝对稳定性
- CoV图 → 相对稳定性

---

## 📝 论文撰写模板

### Methods 部分

```latex
\subsection{Comprehensive Stability Evaluation}

To avoid misleading conclusions from low-mean configurations appearing 
"stable" due to low absolute variance, we adopt a multi-dimensional 
evaluation framework:

\begin{enumerate}
    \item \textbf{Performance}: Mean success rate $\bar{r}$ across 50 seeds
    \item \textbf{Absolute Stability}: Standard deviation $\sigma$
    \item \textbf{Relative Stability}: Coefficient of Variation 
          $\text{CoV} = \sigma / \bar{r}$
    \item \textbf{Classification}: Pareto front analysis combining 
          performance and stability thresholds
\end{enumerate}

Configurations are classified into four quadrants (Figure X):
\begin{itemize}
    \item \textbf{Optimal}: $\bar{r} \geq 0.7$ and $\text{CoV} \leq 0.1$ 
          (high utility and reproducibility)
    \item \textbf{Unstable}: $\bar{r} \geq 0.7$ and $\text{CoV} > 0.1$ 
          (sufficient performance but high variance)
    \item \textbf{Poor (Stable)}: $\bar{r} < 0.7$ and $\text{CoV} \leq 0.1$ 
          (low variance but inadequate performance)
    \item \textbf{Unusable}: $\bar{r} < 0.7$ and $\text{CoV} > 0.1$ 
          (both performance and stability are poor)
\end{itemize}

This framework ensures stability claims are not confounded by low 
absolute performance levels.
```

### Results 部分

```latex
\subsection{Performance-Stability Trade-off Analysis}

Figure \ref{fig:dual_axis} presents the dual-axis analysis of buffer 
sizes. While buffer size 1000 exhibits the lowest CoV (0.083), its mean 
success rate (0.42) falls significantly below the practical threshold 
(0.7), classifying it as \textit{stable but inadequate}. This demonstrates 
the critical importance of evaluating stability in the context of 
absolute performance.

The Pareto front classification (Figure \ref{fig:pareto}) reveals three 
optimal configurations (buffer sizes 10,000, 30,000, and 50,000) residing 
in the target zone ($\bar{r} > 0.8$, $\text{CoV} < 0.1$). These 
configurations achieve both high performance and reproducibility, making 
them suitable for safety-critical deployment.

In contrast, buffer size 100,000 exhibits high mean performance (0.79) 
but elevated CoV (0.18), indicating \textit{unstable} behavior potentially 
due to overfitting to outdated transitions. Buffer size 1000's low 
absolute performance (despite low CoV) suggests insufficient experience 
diversity for effective learning.

\textbf{Key Finding}: Buffer sizes 10,000-50,000 represent a sweet spot, 
balancing sample efficiency (fast learning) with stability 
(reproducibility across seeds). This finding would be obscured by 
examining only relative variability (CoV) without absolute performance 
metrics.
```

---

## 🔬 实验报告清单

### 必须包含的图表

| 图表类型 | 用途 | 关键信息 |
|---------|------|---------|
| ✅ 双轴图 | 联合展示性能+稳定性 | 识别"假稳定"配置 |
| ✅ Pareto图 | 四象限分类 | 直观区分最优/陷阱配置 |
| ✅ 箱线图 | 分布形态 | 识别异常值和偏态 |
| ✅ 柱状图 | 绝对性能对比 | 快速比较均值差异 |
| ✅ CoV趋势图 | 相对稳定性 | 辅助验证稳定性结论 |

### 必须回答的问题

1. **哪些配置同时满足性能和稳定性？**  
   → 查看 Pareto 图右下绿色区域

2. **低 CoV 的配置是否性能足够？**  
   → 查看双轴图绿色阈值线以上的点

3. **是否存在"稳定但无用"的陷阱配置？**  
   → 查看 Pareto 图左下橙色区域

4. **最优配置的具体数值？**  
   → 查看双轴图中同时满足两条阈值线的点

---

## 🎯 推荐分析流程

### Step 1: 生成所有图表
```bash
cd /nesi/project/uoa04575/SimpleSelf-drivingCar-master/v5_faster
sbatch run_plot_replay_sweep.job rb122k
```

### Step 2: 优先查看 Pareto 图
- 文件：`plots/success_rate_pareto_classification.png`
- 目标：识别绿色"Optimal"区域的配置

### Step 3: 双轴图验证
- 文件：`plots/success_rate_performance_stability_dual_axis.png`
- 验证：最优配置是否同时高于两条阈值线

### Step 4: 箱线图补充
- 文件：`plots/final_success_rate_boxplot.png`
- 检查：最优配置是否存在严重异常值

### Step 5: 撰写结论
使用上述论文模板，强调：
> "我们避免了单独使用 CoV 可能导致的误判，通过联合分析性能均值和相对变异性，识别出既稳定又高性能的配置。"

---

## 📚 相关文献支持

### 核心引用

1. **Henderson et al. (2018)** - Deep Reinforcement Learning that Matters  
   → 强调需要报告均值和方差，避免单一指标误导

2. **Mahmood et al. (2018)** - Benchmarking for Evaluating RL Algorithms  
   → 提出 CoV < 0.1 作为稳定性阈值

3. **Agarwal et al. (2021)** - Deep RL at the Edge of Statistical Precipice  
   → 推荐使用多维度评估框架（IQM, Optimality Gap, CoV）

### 引用示例

```latex
Prior work has emphasized the importance of multi-dimensional stability 
evaluation to avoid misleading conclusions from single metrics 
\citep{henderson2018matters, agarwal2021statistical}. Following 
\citet{mahmood2018benchmarking}, we adopt CoV $\leq$ 0.1 as the 
reproducibility threshold but supplement it with absolute performance 
criteria to ensure practical utility.
```

---

## ✅ 总结

| 维度 | 单独使用问题 | 联合使用优势 |
|------|------------|------------|
| **均值** | 忽略稳定性 | 确保绝对性能达标 |
| **标准差** | 忽略相对性 | 衡量绝对变异程度 |
| **CoV** | 误判低性能为稳定 | 评估相对一致性 |
| **综合** | ❌ | ✅ 避免所有陷阱 |

**关键原则**：  
> 稳定性必须在**高性能区域**内评估才有意义。低水平的稳定性（"稳定地失败"）没有实用价值。

**最佳实践**：  
使用 **Pareto 四象限图** 作为主图，**双轴图** 作为补充，**箱线图** 验证分布，**CoV 趋势图** 支持结论。
