# MBRS 当前研究入口

主实验固定seed17，最佳像素尾部配置为 **Patch16 / stride8 / Top10原始MSE选块 / 整图与局部权重0.5/0.5**。同源模型续训20轮，相对整图MSE对照：整图PSNR +0.179753 dB，Top25局部PSNR +0.247056 dB。当前不启动新的训练。

## 结果与证据

| 内容 | 当前权威入口 |
|---|---|
| 10组正式controlled结果、绝对/差值与工件哈希 | [master table](reports/current_controlled_master_table.md) / [CSV](reports/current_controlled_master_table.csv) |
| 65个实际run的配置与来源 | [谱系v3](reports/full_experiment_lineage_v3.md) / [CSV](reports/full_experiment_lineage_v3.csv) |
| 真正的Spearman、分网格逐图配对 | [证据审计](reports/perceptual_evidence_audit.md) |
| 实际RGB整图与局部观察 | [看图入口](reports/artifact_observation_guide.md) |
| 本轮修正、剩余局限及下一步 | [执行计划](reports/project_readiness_and_next_steps.md) |

旧full_experiment_lineage.csv与v2保留历史，不能作为配置数据源。旧感知summary里相关类型/网格的错误已修正并指向新的审计。原master table在覆盖前按内容哈希归档到reports/audit_history。

## 复现和检查（均不训练）

在项目根目录执行：

```bash
# 只读校验结果、配置、checkpoint、训练轮数与公开CSV
/root/miniforge/bin/python experiments/rebuild_research_tables.py --check
# 从原始数据重建正式表格（旧表自动按内容哈希归档）
/root/miniforge/bin/python experiments/rebuild_research_tables.py
# 复核两种主方法的逐图浓度与感知尾部，缓存放在挂载数据目录
CUDA_VISIBLE_DEVICES=0 /root/miniforge/bin/python experiments/audit_perceptual_evidence.py
# 科学表格与指标定义回归检查
/root/miniforge/bin/python -m unittest experiments.test_research_evidence
```

## 当前论文表述

受控局部监督降低了像素定义的残差集中度；32×32 Gini在98%图像中下降。native32 Top10 LPIPS总体均值几乎不变，直接观看两个主模型差异也较小。因此人眼可见性仍是待验证问题。下一步优先冻结公平的匿名人眼对比方案，而非自动继续调ratio或alpha。
