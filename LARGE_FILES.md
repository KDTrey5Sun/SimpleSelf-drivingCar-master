# 大文件与数据说明

本仓库仅包含代码和必要的轻量资源，为保证可推送至 GitHub：

- 已忽略以下大文件和目录：
  - 所有模型权重与检查点：`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`
  - 实验数据与日志目录：`**/exp_data/`, `**/v5_exp_data/`, `**/checkpoints/`
  - 本地虚拟环境：`venv/`, `.venv/`, `.conda/`

## 如果你需要权重或训练日志

- 推荐把权重文件上传到云盘或 GitHub Releases，然后在 README 中附上下载链接。
- 或者使用 Git LFS 存储少量必要的权重文件（注意 GitHub LFS 免费额度约 1GB/每月）。

## 本地复现实验

1. 安装依赖：
   - 见 `requirements.txt`
2. 训练得到权重：
   - 运行相应版本目录下的训练脚本（如 `v5/DQN_CAR_v5.py`），会在本地生成 `exp_data/` / `checkpoints/` 等。

## 想把历史中的大文件也彻底清理掉（可选，需谨慎）

当前本地 `.git` 目录仍可能很大（包含历史中的大文件对象），如需彻底瘦身，可使用 git-filter-repo 重写历史：

- 安装：`brew install git-filter-repo` 或 `python3 -m pip install git-filter-repo`
- 清理命令示例（会重写所有提交，务必先确认）：
  - 删除权重与实验文件：`git filter-repo --force --path-glob "*.pt" --invert-paths`
  - 也可用 `--path-glob "**/exp_data/**" --path-glob "**/checkpoints/**" --invert-paths`
- 重写后需要强推：`git push -f origin main`

注意：历史重写是破坏性的操作，请先确认无人基于旧历史开发。
