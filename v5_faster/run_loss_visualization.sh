#!/bin/bash
# Loss可视化快速使用示例

# 基本用法
python plot_replay_sweep.py \
    --base_dir ./v5_faster/v5_exp_data/replay_sweep \
    --tag rb122k \
    --max_ep 1000 \
    --smooth 5

# 说明：
# --base_dir: 实验数据根目录
# --tag: 实验标签 (例如 rb122k, experiment_20241130 等)
# --max_ep: 最大episode数 (可选，用于截断X轴)
# --smooth: 平滑窗口 (1=无平滑, 5-10=轻度平滑, >20=强平滑)

# 生成的Loss图表 (保存在 <base_dir>/<tag>/plots/ 目录):
# 1. loss_1_mean_multilevel_ci.png       - 均值+置信区间 (主图)
# 2. loss_2_quantile_bands.png           - 分位数带 (稳健性)
# 3. loss_3_individual_trajectories.png  - 个体轨迹 (变异性)
# 4. loss_4_coefficient_of_variation.png - 变异系数 (相对稳定性)

echo "Loss可视化图表生成完成！"
echo "请查看输出目录中的 loss_*.png 文件"
