# 局部伪影实验协议

本目录保存第一阶段实验代码，不修改 MBRS 原始 `train.py` / `test.py`。

## 文件存储

数据集、训练输出、日志和可视化文件统一放在 `/mnt/wmcontent/GLX/icassp/MBRS/`。
项目目录提供 `datasets`、`experiments/runs`、`experiments/logs` 和 `visualizations`
软链接，便于从代码目录查看，不在本地保存大文件。

## 固定协议

- 输入：128×128，64 bit。
- 数据：DIV2K train 800 张、validation 50 张、test 50 张。
- 训练噪声：`RandomCrop(0.3, 1.0)`，保留面积范围为 30%–100%。
- 图像损失：不使用判别器；消息 MSE 权重为 10。
- `global`：全局 MSE 权重 1。
- `patch_mean`：非重叠 32×32 patch MSE 平均，权重 1。
- `worst`：全局 MSE 和 top-25% 最差 patch MSE 各占 0.5。
- patch 指标：全局 MSE、patch 平均 MSE、最差 patch MSE 及其 PSNR。
- patch 划分：右侧/下侧不足一个 patch 时补零，同时用有效像素 mask 计算 patch MSE；因此不遗漏边界像素，也不把 padding 计入损失。
- 每次 `image_loss_components` 调用都会断言独立计算的 patch mean 与 global MSE 的差小于 `1e-6`。
- 裁剪评测：固定 30%、50%、70%、100% 保留面积，随机位置；每个 checkpoint 使用相同测试 split。

## Smoke test

当前 Codex 环境检测不到 CUDA，因此先使用 CPU 做链路验证。输出放在 `/tmp`，不与正式结果混淆：

```bash
cd /root/workspace/GLX/icassp/MBRS
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/workspace/GLX/icassp/.conda-envs/mbrs/bin/python \
  experiments/train_local_patch.py \
  --config experiments/config_global_128_m64.json \
  --output-dir /tmp/mbrs-smoke-global-20260903 \
  --device cpu --epochs 1 --max-train-batches 2 --max-val-batches 1
```

正式训练节点可将 `--output-dir` 指向 `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/<name>`，并去掉 batch 限制。三个 config 使用同一 seed 和裁剪协议，逐个运行。

CUDA 模式默认优先使用 `/root/miniforge/bin/python`（PyTorch 2.7/CUDA 12.6），以兼容 A800；CPU 模式仍使用项目的 PyTorch 1.5 环境。也可以通过 `PYTHON_BIN` 显式覆盖。

正式节点可直接使用以下命令在 tmux 后台按顺序运行三组训练。默认使用 CUDA；没有 GPU 时显式设置 `DEVICE=cpu`，脚本会按 CPU 方案运行：

```bash
cd /root/workspace/GLX/icassp/MBRS
DEVICE=cpu RESULTS_ROOT=/root/workspace/GLX/icassp/MBRS/experiments/runs \
  experiments/run_matrix_background.sh
tmux attach -t icassp-mbrs-matrix
```

CPU 模式默认使用 8 个 OpenMP/MKL 线程。若 CPU 负载过高，可设置 `OMP_NUM_THREADS` 和 `MKL_NUM_THREADS` 为更小值。

修正 patch 划分/归约逻辑后，所有依赖 patch loss 的实验可用以下命令重新运行。旧结果不会覆盖，输出目录带有 `fixed_` 前缀；global 基线不包含 patch loss，不在此矩阵中重复训练：

```bash
cd /root/workspace/GLX/icassp/MBRS
experiments/run_patch_related_fixed_background.sh
tail -f /mnt/wmcontent/GLX/icassp/MBRS/logs/patch-related-fixed.log
```

如果需要在这 6 组 patch 相关实验全部完成后自动重训 global，可提前启动 watcher。它会检查所有 `checkpoint_0100.pth` 存在后才开始：

```bash
experiments/run_global_after_patch_background.sh
tail -f /mnt/wmcontent/GLX/icassp/MBRS/logs/global-after-patch-fixed.log
```

## 评测

```bash
/root/workspace/GLX/icassp/.conda-envs/mbrs/bin/python \
  experiments/evaluate_crop.py \
  --config experiments/config_worst_128_m64.json \
  --checkpoint <run>/checkpoint_0100.pth \
  --output <run>/crop_metrics.json \
  --device cuda --repeats 5
```

LPIPS 暂不加入第一版环境，避免引入新依赖；待全局/局部 MSE 结论稳定后再作为补充指标。

## 消融实验

主矩阵完成后，以 `worst` 方案为基线，固定 seed=17，只改变一个因素：

- `patch16`、`patch64`：patch size 从 32 改为 16、64。
- `topk10`、`topk50`：top-k ratio 从 25% 改为 10%、50%。

消融实验仍使用 100 epoch、相同数据划分和相同 crop 评测协议。

评测生成的 `examples.pt` 可渲染为 cover、水印图、归一化残差和不同裁剪结果的拼图：

```bash
/root/workspace/GLX/icassp/.conda-envs/mbrs/bin/python \
  experiments/render_examples.py \
  --examples <run>/examples.pt --output <run>/examples.png
```

所有 fixed checkpoint 的统一评测可在后台运行：

```bash
experiments/run_fixed_evaluation_background.sh
tail -f /mnt/wmcontent/GLX/icassp/MBRS/logs/fixed-evaluation.log
```

多 seed 和 worst patch 权重扫描可用以下命令后台运行。它会顺序执行 9 组多 seed 实验及 2 组权重实验，结果写入 `optimization_*` 目录：

```bash
experiments/run_optimization_suite_background.sh
tail -f /mnt/wmcontent/GLX/icassp/MBRS/logs/optimization-suite.log
```
