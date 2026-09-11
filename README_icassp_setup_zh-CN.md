# ICASSP 水印实验环境

本目录以论文思路文件 [`../idea.txt`](../idea.txt) 为实验要求，使用官方
[MBRS](https://github.com/jzyustc/MBRS) 代码作为基础，不将旁边的
`VideoX-Fun` 项目依赖混入本实验。

## 已落实的实验基线

- 输入尺寸：`128×128`
- 消息长度：`64 bit`
- 数据集：完整下载公开的 DIV2K HR 图像（800 张训练图 + 100 张公开验证图）
- MBRS 所需目录：`train/validation/test`
- 数据位置：`/mnt/wmcontent/GLX/icassp/MBRS/datasets`
- `train_settings.json` 和 `test_settings.json` 已指向上述绝对路径。

官方 DIV2K 没有公开 test HR 图像。为满足 MBRS 原始代码的三目录约定，下载的
100 张公开 validation HR 图像按编号确定性拆成：`0801–0850` 为 validation，
`0851–0900` 为 test。没有复制图像，三个目录中的文件是指向 `raw/` 的符号链接。

## 环境

已经在当前工作区建立环境前缀：

```text
/root/workspace/GLX/icassp/.conda-envs/mbrs
```

关键版本为 Python 3.8.20、PyTorch 1.5.0、torchvision 0.6.0、kornia 0.3.0、
NumPy 1.19.5、Pillow 8.4.0、SciPy 1.5.4。重建环境可执行：

```bash
./setup_env.sh
conda activate /root/workspace/GLX/icassp/.conda-envs/mbrs
```

## 下载 DIV2K

```bash
./scripts/download_div2k.sh
```

脚本默认下载到 `/mnt/wmcontent`，支持断点续传，并在解压后检查 zip 完整性和
三个 split 的图像数量。也可以用环境变量改变挂载盘根路径，例如：

```bash
WMCONTENT_ROOT=/mnt/wmcontent ./scripts/download_div2k.sh
```

当前 Codex 沙箱看到的 `/mnt/wmcontent` 是只读 GPFS 挂载，因此本次无法直接写入
数据。挂载恢复可写后，在本目录执行上面的下载命令即可；脚本会自动创建缺少的
`/mnt/wmcontent` 目录。

## 运行前检查

```bash
cd /root/workspace/GLX/icassp/MBRS
/root/workspace/GLX/icassp/.conda-envs/mbrs/bin/python -c \
  'import torch, torchvision, kornia; print(torch.__version__, torchvision.__version__, kornia.__version__)'
```

原始 `train.py` / `test.py` 以及局部最差 patch 损失尚未改写；本目录目前是可复现
的 MBRS 基线环境和数据准备阶段。

训练输出默认放在代码目录的 `results/`。在挂载盘可写时，建议显式指定：

```bash
MBRS_RESULTS_ROOT=/mnt/wmcontent/GLX/icassp/MBRS/results \
  /root/workspace/GLX/icassp/.conda-envs/mbrs/bin/python train.py
```
