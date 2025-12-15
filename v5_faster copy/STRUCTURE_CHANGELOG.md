# Structure.txt 优化变更日志

## 修改日期：2025-11-20

## 主要变更：简化研究框架，聚焦核心问题

### 🗑️ 删除内容

#### 1. **删除原第三维度（SRQ3）：数据重用动态与样本相关性**
   - **原因**：过于细节化，需要大量额外代码实现
   - **包含内容**：
     - Data Reuse Rate详细分析
     - Buffer Turnover Time计算
     - Temporal Span测量
     - 样本覆写统计
     - 三区间假设（过度重用/平衡/欠利用）
   - **处理方式**：部分内容降级为SRQ3（可选机制验证）

#### 2. **简化原第四维度（SRQ4）：计算效率与资源权衡**
   - **原因**：不作为主要研究方向，仅需描述性统计
   - **删除内容**：
     - 详细的帕累托前沿分析
     - 场景化推荐的加权评分系统
     - 边际收益递减分析
     - 多目标优化算法
   - **保留内容**：
     - 基础指标：episodes_per_min, time_to_target, memory_peak
     - 简单散点图：时间vs样本效率

#### 3. **降级原第五维度（SRQ5）：理论机制解释**
   - **原因**：验证难度大，非论文核心贡献
   - **新定位**：SRQ3（可选深度分析）
   - **简化内容**：
     - 保留数据重用率基础分析
     - 保留Q值健康度监控
     - 删除复杂的因果链实验设计

---

## 📊 新的研究框架（简化版）

### **核心研究问题（2个）**

#### 🔴 SRQ1: 样本效率的联合优化分析
- Buffer容量主效应（U型曲线）
- Batch size主效应（倒U型曲线）
- **交互效应验证**（Two-way ANOVA）
- 2D热力图：Buffer × Batch → steps_to_target

#### 🔴 SRQ2: 训练稳定性问题诊断与缓解
- 高方差问题根因分析（CoV > 50%）
- 4个假设：
  - H2a: 奖励尺度不匹配
  - H2b: Epsilon衰减耦合
  - H2c: 多稳态策略空间
  - H2d: 目标网络更新不足
- **消融实验**：验证缓解策略效果

### **扩展分析（1个，可选）**

#### 🟢 SRQ3: 理论机制验证（可选深度内容）
- 数据重用率计算与分析
- Q值漂移测量（需额外代码）
- 样本相关性的理论推导

### **补充观察（非主要研究）**

#### 📊 计算效率特征（描述性统计）
- 记录基础性能指标（随实验自动收集）
- 时间-样本效率权衡散点图
- 不进行深入分析

---

## 🎯 实验任务简化

### Phase 1: 紧急修复（1周）
- ✅ 消融实验：5组×5seeds = 25 runs
- ✅ 目标：CoV从>50%降至<30%

### Phase 2: 核心分析（2-3周）
- ✅ 2D grid sweep：15 cells × 3 reps = 45 runs
- ✅ 核心可视化：热力图、箱线图、学习曲线
- ✅ 统计检验：Two-way ANOVA

### Phase 3: 论文撰写（2-3周）
- ✅ Results：12-15页（SRQ1+SRQ2为主）
- ✅ Discussion：5-6页
- ⭕ SRQ3机制验证：时间允许再做

---

## 📈 图表数量变化

### 原计划（复杂版）
- Figures: 9-12个
- Tables: 4个
- 包含：帕累托前沿、3D散点图、重用率热力图、失败模式分析等

### 新计划（简化版）
- **必做Figures**: 6-7个
  - Figure 4.1: 学习曲线（by buffer/batch）
  - Figure 4.2: 2D热力图（steps_to_target）
  - Figure 4.3: 交互作用图（验证H1c）
  - Figure 4.4: Boxplots（success rate分布）
  - Figure 4.5: Q值健康度演化
  - Figure 4.6: DTW聚类（可选）
  - Figure 4.7: 时间-样本权衡散点图
- **Tables**: 3个
  - Table 3.1: 配置空间
  - Table 4.1: Top-5高效配置
  - Table 4.2: 消融实验结果
  - Table 4.3: 性能统计摘要

---

## ✅ 假设检验清单变化

### 原计划：18个假设（H1a-H5综合）
### 新计划：7个假设（聚焦核心）

| 假设 | 优先级 | 验证方法 |
|------|--------|----------|
| H1a: Buffer单轴U型 | 🔴高 | 单变量拟合 |
| H1b: Batch单轴倒U型 | 🔴高 | 单变量拟合 |
| H1c: 交互效应显著 | 🔴高 | Two-way ANOVA |
| H2a: 奖励尺度不匹配 | 🔴紧急 | 消融实验 |
| H2b: Epsilon耦合 | 🔴紧急 | 消融实验 |
| H2c: 多稳态策略 | 🟡中 | DTW聚类 |
| H2d: 目标网络不足 | 🟡中 | 对比实验 |

删除的假设：
- ~~H3a/b/c: 重用率三区间~~（降级为可选）
- ~~H4a/b/c: 帕累托优化~~（改为描述性）
- ~~H5-链1/2/3/4: 因果机制~~（降级为可选）

---

## 📝 指标记录清单变化

### 必须记录（🔴）：
- total_samples_to_target, episodes, success_rate
- td_loss, epsilon, buffer_capacity
- time_min, episodes_per_min

### 重要补充（🟡）：
- q_value_mean/max/std
- samples_at_epsilon_min
- termination_reason

### 可选指标（🟢）：
- data_reuse_rate（后处理计算）
- avg_q_drift（需额外代码）
- gradient_norm（理论验证）

---

## 🎓 论文写作影响

### Results章节结构调整

**原计划（5个子节）：**
- 4.1 Sample Efficiency (SRQ1)
- 4.2 Training Stability (SRQ2)
- 4.3 Data Reuse Dynamics (SRQ3)
- 4.4 Computational Efficiency (SRQ4)
- 4.5 Mechanistic Validation (SRQ5)

**新计划（4个子节）：**
- 4.1 Sample Efficiency Analysis (SRQ1) **【核心】**
- 4.2 Training Stability Analysis (SRQ2) **【核心】**
- 4.3 Computational Cost Observations **【描述性】**
- 4.4 Mechanistic Analysis (SRQ3) **【可选】**

预计页数：从15-18页缩减至12-15页

---

## 💡 优化效果

### ✅ 优点
1. **聚焦核心**：2个主要SRQ，目标明确
2. **可行性高**：实验量减少约30%
3. **时间可控**：Phase 1+2可在3-4周完成
4. **风险降低**：删除了实现困难的指标

### ⚠️ 注意事项
1. **保留灵活性**：SRQ3标记为"可选"，时间充裕时可做
2. **不影响核心贡献**：交互效应验证+方差缓解仍是亮点
3. **描述性统计足够**：计算成本不做深入分析，报告基础数据即可

---

## 🚀 下一步行动

立即执行（本周）：
1. [ ] 按新框架修改DQN.py（添加reward_scale等参数）
2. [ ] 运行Phase 1消融实验（5组×5seeds）
3. [ ] 验证CoV是否降至<0.30
4. [ ] 准备Phase 2的2D grid sweep

后续安排：
- Week 2-3: 完成2D grid + 核心可视化
- Week 4-5: 撰写Results（聚焦SRQ1+SRQ2）
- Week 6-7: Discussion + 全文打磨

---

## 📞 联系与反馈

如需恢复任何被删除的内容，或对简化方案有疑问，请及时沟通调整。

**核心原则**：优先保证SRQ1和SRQ2的质量，SRQ3作为"锦上添花"的深度内容。
