# Loss可视化增强说明 (100次重复实验)

## 概述
为了准确呈现100次重复实验的loss走势，已在 `plot_replay_sweep.py` 中添加了4个专门的可视化函数。

## 使用方法

```bash
python plot_replay_sweep.py \
    --base_dir ./v5_faster/v5_exp_data/replay_sweep \
    --tag your_experiment_tag \
    --max_ep 1000 \
    --smooth 5
```

## 生成的Loss图表

### 1. loss_1_mean_multilevel_ci.png ⭐ **推荐用于论文主图**
**内容:**
- 粗实线: 100次运行的均值
- 深色阴影 (±1σ): 68%置信区间
- 浅色阴影 (±2σ): 95%置信区间

**优点:**
- 最直观展示趋势和不确定性
- 置信区间直接显示数据可靠性
- 适合对比不同buffer size的稳定性

**论文中如何描述:**
> "Figure X shows the mean loss curves with confidence intervals across 100 independent runs. The darker shaded region represents ±1 standard deviation (68% CI), while the lighter region shows ±2 standard deviations (95% CI)."

---

### 2. loss_2_quantile_bands.png **稳健性分析**
**内容:**
- 粗线: 中位数 (50th percentile)
- 深色带: IQR (25-75th percentile)
- 浅色带: 10-90th percentile

**优点:**
- 不受极端值影响
- 更稳健的统计方法
- 适合数据有异常值的情况

**何时使用:**
- 补充图，证明结果的稳健性
- 审稿人质疑极端值影响时

---

### 3. loss_3_individual_trajectories.png **直观展示变异性**
**内容:**
- 灰色半透明线: 随机采样10条个体轨迹
- 红色粗线: 均值 ± 标准差

**优点:**
- 直观看到实际的数据分布
- 展示训练过程的多样性
- 帮助发现异常模式

**何时使用:**
- 附录或补充材料
- 详细分析某个特定buffer size
- 展示训练的真实情况

---

### 4. loss_4_coefficient_of_variation.png **相对稳定性分析**
**内容:**
- CV曲线: 变异系数 (CV = σ/μ)
- 参考线: CV=0.1 (低), 0.3 (中), 0.5 (高)

**优点:**
- 归一化的变异性指标
- 不同量级的loss可比较
- 量化训练稳定性

**解读:**
- CV < 0.1: 训练非常稳定
- CV 0.1-0.3: 中等变异
- CV > 0.5: 训练不稳定，需要调参

---

## 推荐使用流程

### 论文正文
1. **主图**: 使用 `loss_1_mean_multilevel_ci.png`
   - 展示所有buffer size的对比
   - 清晰显示置信区间

2. **补充说明**: 文字描述
   ```
   "Each curve represents the mean loss over 100 independent runs, 
   with shaded regions indicating 68% (darker) and 95% (lighter) 
   confidence intervals."
   ```

### 补充材料/附录
1. **稳健性验证**: `loss_2_quantile_bands.png`
2. **详细分析**: `loss_3_individual_trajectories.png`
3. **稳定性分析**: `loss_4_coefficient_of_variation.png`

---

## 统计学解释

### 置信区间 (Confidence Intervals)
- **±1σ (68% CI)**: 假设正态分布，约68%的数据落在此区间
- **±2σ (95% CI)**: 约95%的数据落在此区间
- 区间越窄 → 训练越稳定

### 分位数 (Quantiles)
- **中位数 (50%)**: 不受极端值影响的中心趋势
- **IQR (25-75%)**: 中间50%的数据范围
- **10-90%**: 涵盖80%的数据

### 变异系数 (Coefficient of Variation)
- **定义**: CV = σ/μ (标准差 / 均值)
- **意义**: 相对变异性，消除量级影响
- **判断标准**:
  - CV < 0.1: 低变异 (优秀)
  - CV 0.1-0.3: 中等变异 (可接受)
  - CV > 0.5: 高变异 (需改进)

---

## 常见问题

### Q1: 为什么不直接画100条线？
A: 100条线会重叠在一起，无法看清。统计方法更清晰。

### Q2: 什么时候用均值±std vs 分位数？
A: 
- 均值±std: 数据近似正态分布，直观易懂
- 分位数: 有异常值，或需要更稳健的结果

### Q3: 如何选择smooth参数？
A: 
- smooth=1: 无平滑，保留原始波动
- smooth=5-10: 轻度平滑，保留主要趋势
- smooth>20: 强平滑，仅看大趋势（可能过度）

### Q4: 置信区间很宽怎么办？
A: 
- 说明训练不稳定，变异性大
- 可能需要调整超参数
- 在论文中诚实报告，讨论原因

### Q5: 不同buffer size的曲线长度不同？
A: 
- 脚本会自动对齐到中位数长度
- 确保公平比较
- 终点标注显示有效样本数

---

## 修改记录

### 主要改动
1. **删除了原有的简单loss绘图** (只有均值线)
2. **添加了4个增强版loss可视化函数**
3. **专门针对100次重复实验优化**

### 其他metric保持不变
- success_rate
- td_mean  
- td_std
- 其他所有图表

仍使用原来的median-aligned方法绘制。

---

## 技术细节

### 数据对齐策略
- 不同run可能有不同的episode数
- 使用median长度截断，确保公平
- 每个episode点计算统计量时忽略缺失值

### 平滑处理
- 使用移动平均 (moving average)
- 保留主要趋势，减少噪声
- 不改变数据的整体特征

### 颜色方案
- 一致的颜色映射 (COLORS列表)
- 同一buffer size在所有图中颜色相同
- 便于跨图对比

---

## 示例输出说明

运行脚本后，会看到如下输出：

```
[PHASE 1.5] Plotting ENHANCED LOSS visualizations (100 runs per size)...
  这些图表专门设计用于展示100次重复实验的loss走势:
  [1/4] 均值 + 多层置信区间 (推荐用于论文主图)...
    ✓ /path/to/plots/loss_1_mean_multilevel_ci.png
  [2/4] 分位数带演化 (稳健性分析)...
    ✓ /path/to/plots/loss_2_quantile_bands.png
  [3/4] 个体轨迹采样 (直观展示变异性)...
    ✓ /path/to/plots/loss_3_individual_trajectories.png
  [4/4] 变异系数演化 (相对稳定性)...
    ✓ /path/to/plots/loss_4_coefficient_of_variation.png

======================================================================
LOSS可视化推荐 (100次重复实验)
======================================================================

推荐使用顺序:
  1. loss_1_mean_multilevel_ci.png
     → 论文主图，展示均值趋势 + 68%/95%置信区间
     ...
```

---

## 联系与反馈

如有问题或需要其他定制化可视化，请修改 `plot_replay_sweep.py` 中的相应函数。

所有增强函数以 `plot_loss_enhanced_` 开头，易于识别和修改。
