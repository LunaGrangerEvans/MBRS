"""Recompute image-paired evidence from fixed checkpoints; never optimize a model."""

import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.analyze_patch_distortion import extract_patches, load_encoded, ssim_per_sample
from experiments.rebuild_research_tables import digest, publish, csv_text

ROOT = Path('/mnt/wmcontent/GLX/icassp/MBRS')
OUT = ROOT / 'reports/perceptual_evidence_audit'
METHODS = {
    '整图MSE续训（权重1，无局部项）': 'controlled_seed17_global_continuation',
    'Patch16/stride8/Top10原始MSE（整图/局部权重0.5/0.5）': 'controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50',
}


def energy_metrics(scores):
    x = np.asarray(scores, dtype=np.float64)
    n = x.shape[1]
    k = math.ceil(n * .1)
    sorted_x = np.sort(x, axis=1)
    total = x.sum(1)
    mean = x.mean(1)
    gini = ((2*np.arange(1,n+1)-n-1) * sorted_x).sum(1) / (n * total)
    return dict(gini=gini, cv=x.std(1)/mean,
                top10_over_mean=sorted_x[:,-k:].mean(1)/mean,
                top10_energy_share=sorted_x[:,-k:].sum(1)/total)


def paired_summary(base, candidate):
    delta = np.asarray(candidate) - np.asarray(base)
    return dict(percent_lower=float(100*np.mean(delta<0)),
                median_change=float(np.median(delta)), mean_change=float(np.mean(delta)))


def main():
    import lpips
    torch.set_num_threads(4)
    torch.manual_seed(17)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    manifest_path = ROOT / 'reports/uniform_eval_manifest.pt'
    manifest = torch.load(manifest_path, map_location='cpu', weights_only=False)
    assert digest(manifest_path) == '36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339'
    images, messages = manifest['images'].float(), manifest['messages'].float()
    assert len(images) == 50
    metric = lpips.LPIPS(net='alex', version='0.1', verbose=False).to(device).eval()
    rows, tensors = [], {}
    for label, run in METHODS.items():
        ck = ROOT / 'experiments/runs' / run / 'checkpoint_0020.pth'
        encoded = load_encoded(ck, images, messages, device, 16)
        tensors[label] = dict(encoded=encoded, checkpoint=str(ck), checkpoint_sha256=digest(ck))
        for size in (16,32):
            p, q = extract_patches(encoded,size), extract_patches(images,size)
            count = (128//size)**2
            mse = (p-q).square().mean((1,2,3)).reshape(50,count).numpy()
            measures = energy_metrics(mse)
            measures['global_mse'] = mse.mean(1)
            if size == 32:
                lp, ss = [], []
                with torch.no_grad():
                    for start in range(0,len(p),256):
                        a,b = p[start:start+256].to(device),q[start:start+256].to(device)
                        lp.append(metric(a,b).flatten().cpu())
                        ss.append(ssim_per_sample(a,b).cpu())
                l = torch.cat(lp).reshape(50,count).numpy()
                s = torch.cat(ss).reshape(50,count).numpy()
                k = math.ceil(count*.1)
                lt = np.sort(l,axis=1)[:,-k:].mean(1)
                st = np.sort(s,axis=1)[:,:k].mean(1)
                measures.update(top10_lpips=lt, lpips_tail_gap=lt-np.median(l,axis=1),
                                bottom10_ssim=st, ssim_tail_gap=np.median(s,axis=1)-st)
                tensors[label].update(lpips32=l, ssim32=s, mse32=mse)
            for i in range(50):
                # Same columns for both grids; no unsupported native16 LPIPS.
                rows.append(dict(method=label,index=i,patch_size=size,patch_stride=size,
                                 top10_k=math.ceil(count*.1),
                                 **{key:float(measures[key][i]) if key in measures else '' for key in
                                    ['global_mse','gini','cv','top10_over_mean','top10_energy_share',
                                     'top10_lpips','lpips_tail_gap','bottom10_ssim','ssim_tail_gap']}))
    OUT.mkdir(parents=True,exist_ok=True)
    torch.save(tensors, OUT/'final_outputs_and_patch_metrics.pt')
    (OUT/'per_image.csv').write_text(csv_text(rows))
    names = list(METHODS)
    summary, correlations = [], []
    for size in (16,32):
        a = [r for r in rows if r['method']==names[0] and r['patch_size']==size]
        b = [r for r in rows if r['method']==names[1] and r['patch_size']==size]
        for key in ['gini','cv','top10_over_mean','top10_energy_share'] + (['top10_lpips','lpips_tail_gap','bottom10_ssim','ssim_tail_gap'] if size==32 else []):
            av,bv = [r[key] for r in a],[r[key] for r in b]
            summary.append(dict(grid=size,metric=key,base_mean=float(np.mean(av)),candidate_mean=float(np.mean(bv)),**paired_summary(av,bv)))
    for name in names:
        records=[r for r in rows if r['method']==name and r['patch_size']==32]
        for x,y in [('gini','lpips_tail_gap'),('top10_over_mean','top10_lpips'),
                    ('top10_energy_share','lpips_tail_gap'),('cv','ssim_tail_gap')]:
            rho,p = spearmanr([r[x] for r in records],[r[y] for r in records])
            correlations.append(dict(method=name,x=x,y=y,spearman_rho=float(rho),p_value=float(p),images=50))
    (OUT/'paired_summary.csv').write_text(csv_text(summary))
    (OUT/'spearman_correlations.csv').write_text(csv_text(correlations))
    (OUT/'provenance.json').write_text(json.dumps(dict(manifest=str(manifest_path),
        manifest_sha256=digest(manifest_path),script_sha256=digest(Path(__file__)),
        checkpoint_provenance={k:{f:v for f,v in t.items() if f in {'checkpoint','checkpoint_sha256'}} for k,t in tensors.items()},
        concentration_grids=[16,32],perceptual_grid=32,epoch=20,
        display_clipping=False,lpips='AlexNet v0.1 native32',
        quantile='per-image',unit='image; not patch or repeated crop'),indent=2))
    lines=['# 感知证据复核：逐图数据、统计口径与纠错', '',
           '仅重算既有epoch20 checkpoint与固定50张test image，未训练。所有逐图数据和浮点输出持久化，报告生成可直接复查。', '',
           '## 已确认的旧报告问题', '',
           '- 旧matched分析脚本使用 `np.corrcoef`，结果是Pearson；后续summary却称为Spearman。',
           '- 旧summary的“Top10/Mean vs Top10 LPIPS ≈0.028”实际引用了Top10/Mean与LPIPS TailGap的Pearson相关，既换了变量又换了相关类型。',
           '- 旧paired图按32×32计算，报告中100% Gini改善/median值来自16×16；现在分网格列出。',
           '- Top10在32×32网格上是ceil(16×0.1)=2块（实际12.5%）；16×16是7/64块。Top10 energy share与Top10/Mean只差常数k/N，不是两份独立机制证据。', '',
           '## 正确的逐图配对统计', '',
           '| 网格 | 指标 | 整图MSE续训 | Hard Top10 | 数值更低的图像比例 | 配对变化中位数 |',
           '|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["grid"]} | {r["metric"]} | {r["base_mean"]:.9g} | {r["candidate_mean"]:.9g} | {r["percent_lower"]:g}% | {r["median_change"]:+.9g} |')
    lines += ['', 'Bottom10 SSIM越高越好；该行“更低比例”表示退化比例。其余指标越低越好。', '',
              '## 真正的Spearman（每点一张图，50张）', '',
              '| 方法 | x | y | Spearman rho | 描述性p值 |', '|---|---|---|---:|---:|']
    for r in correlations:
        lines.append(f'| {r["method"]} | {r["x"]} | {r["y"]} | {r["spearman_rho"]:+.6f} | {r["p_value"]:.6g} |')
    lines += ['', '不把多种相关检验的p值用作项目成功目标；相关性不是因果关系，也不是人类可见性实验。', '',
              '## 数据与复现', '', f'- 逐图CSV、paired CSV、rho CSV、浮点缓存、provenance：`{OUT}`。',
              '- 复现命令：`CUDA_VISIBLE_DEVICES=0 /root/miniforge/bin/python experiments/audit_perceptual_evidence.py`。',
              '- 正式PSNR仍按数据集平均MSE换算；这里不引入新的候选筛选或训练。']
    publish('perceptual_evidence_audit.md','\n'.join(lines)+'\n')
    print(json.dumps(dict(paired=summary,correlations=correlations),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
