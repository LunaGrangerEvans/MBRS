# 感知证据复核：逐图数据、统计口径与纠错

仅重算既有epoch20 checkpoint与固定50张test image，未训练。所有逐图数据和浮点输出持久化，报告生成可直接复查。

## 已确认的旧报告问题

- 旧matched分析脚本使用 `np.corrcoef`，结果是Pearson；后续summary却称为Spearman。
- 旧summary的“Top10/Mean vs Top10 LPIPS ≈0.028”实际引用了Top10/Mean与LPIPS TailGap的Pearson相关，既换了变量又换了相关类型。
- 旧paired图按32×32计算，报告中100% Gini改善/median值来自16×16；现在分网格列出。
- Top10在32×32网格上是ceil(16×0.1)=2块（实际12.5%）；16×16是7/64块。Top10 energy share与Top10/Mean只差常数k/N，不是两份独立机制证据。

## 正确的逐图配对统计

| 网格 | 指标 | 整图MSE续训 | Hard Top10 | 数值更低的图像比例 | 配对变化中位数 |
|---|---|---:|---:|---:|---:|
| 16 | gini | 0.131393502 | 0.120987325 | 100% | -0.00974242781 |
| 16 | cv | 0.237871738 | 0.218945595 | 98% | -0.017276894 |
| 16 | top10_over_mean | 1.44067459 | 1.40186939 | 96% | -0.0324145605 |
| 16 | top10_energy_share | 0.157573783 | 0.153329465 | 96% | -0.00354534255 |
| 32 | gini | 0.0940044007 | 0.085523585 | 98% | -0.00771855796 |
| 32 | cv | 0.172153071 | 0.156886757 | 98% | -0.0144280947 |
| 32 | top10_over_mean | 1.29347525 | 1.26740732 | 92% | -0.0208058898 |
| 32 | top10_energy_share | 0.161684406 | 0.158425915 | 92% | -0.00260073623 |
| 32 | top10_lpips | 0.00135634619 | 0.00135624524 | 64% | -4.10757784e-05 |
| 32 | lpips_tail_gap | 0.000846302984 | 0.000869936356 | 64% | -1.24215876e-05 |
| 32 | bottom10_ssim | 0.917602355 | 0.920069772 | 40% | +0.000666499138 |
| 32 | ssim_tail_gap | 0.0349468458 | 0.0339690208 | 52% | -0.000483453274 |

Bottom10 SSIM越高越好；该行“更低比例”表示退化比例。其余指标越低越好。

## 真正的Spearman（每点一张图，50张）

| 方法 | x | y | Spearman rho | 描述性p值 |
|---|---|---|---:|---:|
| 整图MSE续训（权重1，无局部项） | gini | lpips_tail_gap | -0.165522 | 0.250655 |
| 整图MSE续训（权重1，无局部项） | top10_over_mean | top10_lpips | -0.070828 | 0.624998 |
| 整图MSE续训（权重1，无局部项） | top10_energy_share | lpips_tail_gap | -0.143818 | 0.319049 |
| 整图MSE续训（权重1，无局部项） | cv | ssim_tail_gap | -0.049124 | 0.734779 |
| Patch16/stride8/Top10原始MSE（整图/局部权重0.5/0.5） | gini | lpips_tail_gap | -0.221513 | 0.122107 |
| Patch16/stride8/Top10原始MSE（整图/局部权重0.5/0.5） | top10_over_mean | top10_lpips | -0.094454 | 0.514106 |
| Patch16/stride8/Top10原始MSE（整图/局部权重0.5/0.5） | top10_energy_share | lpips_tail_gap | -0.213349 | 0.136846 |
| Patch16/stride8/Top10原始MSE（整图/局部权重0.5/0.5） | cv | ssim_tail_gap | -0.088788 | 0.539773 |

不把多种相关检验的p值用作项目成功目标；相关性不是因果关系，也不是人类可见性实验。

## 数据与复现

- 逐图CSV、paired CSV、rho CSV、浮点缓存、provenance：`/mnt/wmcontent/GLX/icassp/MBRS/reports/perceptual_evidence_audit`。
- 复现命令：`CUDA_VISIBLE_DEVICES=0 /root/miniforge/bin/python experiments/audit_perceptual_evidence.py`。
- 正式PSNR仍按数据集平均MSE换算；这里不引入新的候选筛选或训练。
