# RQ2 批量大小敏感性分析 - 快速参考指南

## 📋 研究问题

**RQ2**: 批量大小如何调节缓冲区大小对 DQN 性能的影响？

### SRQ2.1: 性能趋势一致性
- **问题**: Chapter 3 中观察到的缓冲区大小性能趋势在不同批量大小下是否保持一致？
- **指标**: 跨批量大小的缓冲区排名 Spearman 秩相关系数 (ρ)
- **假设 H2.1**: 缓冲区排名在不同批量下保持一致 (平均 ρ > 0.7)

### SRQ2.2: 交互效应强度
- **问题**: 批量大小是否与缓冲区大小显著交互，还是效应主要是加性的？
- **指标**: 双因素 ANOVA 交互项效应大小 (η²)
- **假设 H2.2**: 批量与缓冲区仅弱交互 (η² < 0.1)

---

## 🔧 已实施的代码修复

### 修复 1: 统一列名
**问题**: `sweep_worker` 返回 `'size'`，但分析函数期望 `'buffer_size'`

**修复**:
```python
row = {
    'buffer_size': task['size'],  # 原来是 'size'
    ...
}
```

### 修复 2: 添加收敛指标
**问题**: 缺少 `episodes_to_convergence` 和 `samples_to_convergence`

**修复**:
```python
episodes_to_convergence = res['first_success_episode'] if res['first_success_episode'] is not None else res['episodes']
samples_to_convergence = res['samples_at_first_success'] if res['samples_at_first_success'] is not None else res['total_samples_collected']

row = {
    ...
    'episodes_to_convergence': episodes_to_convergence,
    'samples_to_convergence': samples_to_convergence,
}
```

### 修复 3: 修正损失方差列名
**问题**: 分析期望 `loss_variance_last100`，但收集的是 `update_loss_variance_last100`

**修复**:
```python
row = {
    ...
    'loss_variance_last100': res.get('update_loss_variance_last100'),
}
```

### 修复 4: 更新 CSV 字段
在 `sweep_replay_buffer_sizes` 和 `sweep_replay_grid` 中:
- 将 `'size'` 改为 `'buffer_size'`
- 添加 `'loss_variance_last100'`, `'episodes_to_convergence'`, `'samples_to_convergence'`
- 添加 `extrasaction='ignore'` 以容错

---

## 📊 需要收集的数据

### 实验设计参数

| 参数 | 配置 | 说明 |
|------|------|------|
| 缓冲区大小 | `[100000, 50000, 10000, 5000, 1000]` | 5 个水平 |
| 批量大小 | `[512, 128, 64, 32, 4]` | 5 个水平 |
| 重复次数 | `3` | 建议至少 3 次 |
| **总运行次数** | **75** | 5 × 5 × 3 |

### 关键性能指标

| 指标 | CSV 列名 | 说明 | 用途 |
|------|----------|------|------|
| 成功率 | `success_rate` | 成功回合/总回合 | 主要性能指标 |
| 收敛速度 | `episodes_to_convergence` | 首次成功的回合数 | SRQ2.1 |
| 样本效率 | `samples_to_convergence` | 首次成功的样本数 | SRQ2.1 |
| 平均得分 | `avg_score` | 所有回合的平均奖励 | SRQ2.1, 2.2 |
| 训练稳定性 | `loss_variance_last100` | 后期损失方差 | SRQ2.2 |

---

## 🚀 运行实验

### 步骤 1: 配置参数（可选）

快速测试配置（编辑 `DQN_CAR_v5_Chapter4.py` 第 7-36 行）:
```python
BUFFER_SIZES = [10000, 5000]  # 2 个水平
BATCH_SIZES = [128, 32]       # 2 个水平
REPEATS = 2                    # 2 次重复
TAG = 'rq2_quick_test'
# 总运行次数: 2 × 2 × 2 = 8 (约 2-4 小时)
```

### 步骤 2: 提交任务

```bash
cd /nesi/project/uoa04575/SimpleSelf-drivingCar-master/v5_faster
sbatch run_DQN_CAR_v5_Chapter4.job
```

### 步骤 3: 监控进度

```bash
# 查看任务状态
squeue -u $USER

# 实时查看日志
tail -f logs/dqn_car_v5_ch4_*.out
```

---

## 📈 分析流程

### SRQ2.1 分析（由代码自动执行）

1. **计算性能矩阵**: 每个 (buffer_size, batch_size) 组合的平均性能
2. **排名计算**: 为每个批量大小，对缓冲区按性能排名
3. **秩相关**: 计算不同批量大小间的 Spearman ρ
4. **结果判断**:
   - **ρ > 0.7**: H2.1 支持 — 排名高度一致
   - **0.5 < ρ ≤ 0.7**: H2.1 部分支持 — 排名基本一致
   - **ρ ≤ 0.5**: H2.1 不支持 — 排名变化显著

### SRQ2.2 分析（由代码自动执行）

1. **双因素 ANOVA**: `performance ~ buffer + batch + buffer×batch`
2. **计算效应大小**: η² = SS_interaction / SS_total
3. **结果判断**:
   - **η² < 0.1**: H2.2 支持 — 弱交互，效应独立
   - **0.1 ≤ η² < 0.14**: H2.2 部分支持 — 中等交互
   - **η² ≥ 0.14**: H2.2 不支持 — 强交互，效应耦合

---

## 📁 输出文件

### 1. 原始数据 CSV
```
v5_faster/v5_exp_data/replay_grid/rb133k/grid_summary.csv
```
包含所有运行的完整数据，每行 = 一次运行的所有指标

### 2. RQ2 分析结果
```
v5_faster/v5_exp_data/replay_grid/rb133k/rq2_analysis/rq2_analysis_results.csv
```
包含:
- 每个指标的平均 Spearman ρ
- ANOVA 统计量 (η², p 值)
- H2.1 和 H2.2 支持状态

### 3. 详细日志
```
v5_faster/v5_exp_data/replay_grid/rb133k/size_{X}/batch_{Y}/rep_{Z}/
├── train_log.txt     # 训练过程日志
├── summary.txt       # 单次运行汇总
└── curve_data.txt    # 逐回合曲线数据
```

---

## 🎯 预期结果解释

### 情景 1: H2.1 和 H2.2 都支持（理想情况）
**数据特征**:
- 平均 Spearman ρ > 0.7
- 交互 η² < 0.1

**解释**:
> "缓冲区大小的性能排序在不同批量大小下保持稳定（平均 ρ = 0.85），交互效应微弱（η² = 0.03），表明这两个超参数的效应是加性且独立的。**实践建议**: 可以分别调优这两个参数。"

### 情景 2: H2.1 支持但 H2.2 不支持
**数据特征**:
- 平均 Spearman ρ > 0.7
- 交互 η² ≥ 0.1

**解释**:
> "虽然缓冲区排名保持一致（ρ = 0.78），但批量大小显著调节了效应强度（交互 η² = 0.12）。**实践建议**: 排名稳定，但大批量下缓冲区效应更强，需注意效应强度的变化。"

### 情景 3: H2.1 和 H2.2 都不支持
**数据特征**:
- 平均 Spearman ρ < 0.5
- 交互 η² > 0.14

**解释**:
> "缓冲区和批量大小存在强交互（η² = 0.18），最优缓冲区大小依赖于批量大小的选择。**实践建议**: 需要联合调优这两个参数，例如小批量（32）需要大缓冲区（100k），大批量（512）可用小缓冲区（10k）。"

---

## ⚠️ 注意事项

### 1. 计算资源
- **完整实验**: 75 次运行，约 60-125 小时（使用 32 核并行）
- **快速测试**: 8 次运行，约 2-4 小时
- **建议**: 先运行快速测试验证代码

### 2. 数据完整性检查
实验完成后运行:
```bash
python validate_rq2_data.py v5_faster/v5_exp_data/replay_grid/rb133k/grid_summary.csv
```

检查项目:
- ✅ 数据完整性（是否所有组合都运行）
- ✅ 缺失值
- ✅ 异常值
- ✅ 分组平衡性

### 3. 常见问题

**Q: 如何减少实验时间？**
A: 
- 减少 `REPEATS` 到 2（最低要求）
- 减少 `max_success` 到 50
- 使用更少的缓冲区/批量大小组合

**Q: 分析报错找不到列？**
A: 确保运行的是修复后的代码版本，检查:
```bash
grep "buffer_size" DQN_CAR_v5_Chapter4.py | grep "task\['size'\]"
# 应该看到: 'buffer_size': task['size']
```

---

## 📚 参考

### 统计方法
- **Spearman ρ**: 非参数秩相关，适用于排序数据
- **双因素 ANOVA**: 分解方差来源（主效应 + 交互效应）
- **效应大小 η²**: 解释的方差比例（< 0.06 小，0.06-0.14 中，≥ 0.14 大）

### 代码位置
- 主文件: `DQN_CAR_v5_Chapter4.py`
- 任务脚本: `run_DQN_CAR_v5_Chapter4.job`
- 分析函数: `analyze_rq2_robustness()` (自动调用)

---

**文档版本**: 1.0  
**最后更新**: 2024-12-02
