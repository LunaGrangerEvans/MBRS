"""Publish the completed global OKLab experiment without further optimization."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import rgb2lab, deltaE_ciede2000

PROJECT=Path(__file__).resolve().parents[1]
ROOT=Path('/mnt/wmcontent/GLX/icassp/MBRS')
STUDY=ROOT/'reports/crop_global_oklab'
FIG=ROOT/'visualizations/crop_global_oklab'


def read(path):
    return list(csv.DictReader(path.open()))


def csv_write(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--split',choices=['test','validation'],default='test')
    args=parser.parse_args()
    folder=STUDY/f'{args.split}_{args.run}'
    result=json.loads((folder/'result.json').read_text())
    summary=result['summary']
    per_image=read(folder/'per_image.csv')
    config=json.loads((ROOT/'experiments/runs'/args.run/'config.resolved.json').read_text())
    reports=PROJECT/'reports'
    csv_write(reports/'crop_global_oklab_main_table.csv',summary)
    csv_write(reports/'crop_global_oklab_per_image.csv',per_image)
    base,cand=summary[1:]
    gate=result['gate']
    passed=gate['preservation_pass'] and gate['color_pass']
    status='MEETS THE PREDECLARED DESCRIPTIVE GATE' if passed else 'QUALITY AND COLOR IMPROVE; THE COMPLETE PRESERVATION GATE IS NOT MET'
    lines=['# Crop Hard Top10 + global OKLab result','',status,'',
        'This run starts from the same seed17 crop-trained Global epoch100 source as the incumbent, restores Adam/BN state, and continues20 epochs at LR1e-4, batch16, RandomCrop(0.3,1.0).',
        '',f'Loss: `10 message MSE + 0.5 global RGB MSE + 0.5 Hard patch16/stride8/Top10 MSE + {config["global_oklab_weight"]:.9g} global OKLab distance`.',
        '', 'The added term is a mean of per-pixel standard OKLab Euclidean distances. It does not replace the global RGB or local pixel-tail term. Chroma-only and signed biases are diagnostics. CIEDE2000 is evaluation-only.',
        '', f'## Fixed {args.split} metrics', '',
        '| Method | PSNR ↑ | SSIM ↑ | 3-scale MS-SSIM ↑ | Full LPIPS ↓ | Top25 local PSNR ↑ | P95 native32 MSE ↓ | Gini ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f'| {row["method"]} | {row["global_psnr"]:.6f} | {row["global_ssim"]:.6f} | {row["global_ms_ssim"]:.6f} | {row["full_lpips"]:.9f} | {row["top25_local_psnr"]:.6f} | {row["patch_mse_p95"]:.9f} | {row["gini"]:.6f} |')
    lines+=['','| Method | CIEDE2000 global ↓ | 5×5 patch Top10 ↓ | Patch P95 ↓ | Max patch ↓ | Global OKLab ↓ | Chroma-only ↓ | Signed CIELAB da/db |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in summary:
        lines.append(f'| {row["method"]} | {row["ciede2000_global"]:.6f} | {row["ciede2000_top10"]:.6f} | {row["ciede2000_p95"]:.6f} | {row["ciede2000_max"]:.6f} | {row["global_oklab"]:.7f} | {row["global_chroma"]:.7f} | {row["signed_cielab_da"]:+.6f} / {row["signed_cielab_db"]:+.6f} |')
    lines+=['','| Method | BER100 | BER70 | BER50 | BER40 | BER30 |','|---|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append('| '+row['method']+' | '+' | '.join(f'{row[f"ber{r}"]:.7f}' for r in (100,70,50,40,30))+' |')
    lines+=['','## Compared with incumbent Hard Top10','']
    for key in ('global_psnr','top25_local_psnr','global_ssim','patch_mse_p95','gini','ciede2000_global','ciede2000_top10','ciede2000_p95','ber30'):
        delta=cand[key]-base[key]
        relative=100*delta/base[key] if base[key]!=0 else None
        lines.append(f'- {key}: {delta:+.9f}'+(f' ({relative:+.2f}%)' if key.startswith('ciede') else ''))
    lines+=['',f'Per-image fractions with lower values (all50 {args.split} images):','']
    for key,value in result['per_image_improvements'].items():
        lines.append(f'- {key}: {value*100:.0f}%.')
    lines+=['','## Validation and debugging record','']
    validation_candidates={}
    for path in sorted(STUDY.glob('validation_*/result.json')):
        record=json.loads(path.read_text())
        if 'gate' not in record:
            continue
        gate_info=record['gate']
        for row in record['summary']:
            rc=json.loads((ROOT/'experiments/runs'/row['run']/'config.resolved.json').read_text())
            validation_candidates[row['run']]={**row,'global_oklab_weight':rc.get('global_oklab_weight',0.0)}
        lines.append(f'- `{path.parent.name}`: preservation={gate_info["preservation_pass"]}, color={gate_info["color_pass"]}; improved-color images={gate_info["color_improved_fraction"]:.0%}.')
    if validation_candidates:
        csv_write(reports/'crop_global_oklab_validation_candidates.csv',list(validation_candidates.values()))
        lines+=['','| Validation configuration | lambda | PSNR | Top25 local PSNR | Gini | Global CIEDE2000 | Top10 CIEDE2000 | BER30 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in validation_candidates.values():
            lines.append(f'| {row["method"]} | {row["global_oklab_weight"]:.9g} | {row["global_psnr"]:.6f} | {row["top25_local_psnr"]:.6f} | {row["gini"]:.6f} | {row["ciede2000_global"]:.6f} | {row["ciede2000_top10"]:.6f} | {row["ber30"]:.7f} |')
    evaluation_note = (
        'Selection used validation only. The selected epoch20 was then evaluated on the existing formal test manifest. This re-used test set was previously examined and is not a pristine holdout. No training decisions follow its outcome.'
        if args.split=='test' else
        'These are validation-only endpoint results. No candidate passed the full validation gate, so no candidate formal-test evaluation was opened and the incumbent paper evidence remains unchanged.'
    )
    lines+=['',evaluation_note,
        '', '## Interpretation', '',
        ('The candidate satisfies the predeclared preservation and color-benefit tolerances on this split. It is a promising extension, not evidence of universal superiority or statistical equivalence. Keep the existing paper draft as its historical freeze until this new evidence is reviewed.' if passed else 'The candidate does not meet every preservation/color-benefit tolerance. Retain the incumbent paper method and report the measured trade-off. No extra test-driven parameter scans are performed.'),
        '', 'The acceptance tolerances are engineering criteria, not p-values or a human perceptual study. Lower CIEDE2000 does not prove visible improvement for every observer. The 5×5 color grid differs from native32 MSE/concentration/perceptual evaluation.',
        '', 'This study adds an image regularizer; it has no matched-strength extra-RGB-regularizer control. Gains cannot all be attributed exclusively to the geometry of OKLab. In training logs, `global_oklab`/`global_chroma` are the actual new diagnostics; the inherited `local_oklab` field in topk mode is a legacy alias of raw local MSE and must not be interpreted as a color metric.',
        '', '## Why this is a concentration trade-off', '',
        'Gini describes relative inequality, not absolute error. The candidate reduces per-image P95 pixel error and color error on all50 validation images, while its Gini can increase. Compared with Global continuation, even the half-weight candidate retains a lower mean Gini (0.097303 versus0.100938), but less of the reduction achieved by incumbent Hard Top10 (0.094569). Thus the original concentration benefit is partly reduced, not evidence that absolute errors increased.',
        '', 'A reference-content-only diagnostic groups native32 patches by original brightness/gradient quartiles. At the larger weight, dark/bright group mean MSE decreases9.46%/8.52% and low/high-gradient group MSE decreases9.27%/8.63%. At half weight the respective reductions are5.65%/5.08% and5.58%/5.12%. All grouped mean errors decline; the differences are modest and do not establish artifact relocation as a sole cause. Unequal regional reductions are consistent with the changed normalized distribution. See [region diagnostics](crop_global_oklab_region_diagnostics.csv).',
        '', 'The legacy JPEG+OKLab implementation used an incorrect XYZ matrix on linear RGB. The new module follows the [author’s linear-sRGB implementation](https://bottosson.github.io/posts/oklab/) and has independent primary-color/neutral/gradient tests. Old JPEG outcomes are not evidence against standard OKLab.',
        '', '## Files and metric contract','',
        '- [protocol](crop_global_oklab_protocol.md)',
        '- [main CSV](crop_global_oklab_main_table.csv)',
        '- [per-image CSV](crop_global_oklab_per_image.csv)',
        f'- Complete validation/calibration/provenance: `{STUDY}`.',
        f'- Fixed {args.split} qualitative examples and color distribution: `{FIG}`.',
        '- Quality: clipped RGB[0,1], current frozen evaluator; PSNR from dataset mean MSE, localPSNR mean per-image dB; CIEDE2000 from CIELAB D65, color5×5 stride1, Top10 ceil1538/15376. BER: the existing normalized float output, five fixed masks per ratio, no resize-back. No metric implementation tuned on these results.']
    (reports/'crop_global_oklab_summary.md').write_text('\n'.join(lines)+'\n')

    FIG.mkdir(parents=True,exist_ok=True)
    manifest_path=ROOT/('reports/uniform_eval_manifest.pt' if args.split=='test' else 'reports/content_selector/validation_manifest.pt')
    manifest=torch.load(manifest_path,map_location='cpu',weights_only=False)
    originals=manifest['images']
    values=[]
    for row in (base,cand):
        encoded=torch.load(folder/f'{row["run"]}_outputs.pt',map_location='cpu',weights_only=False)['encoded']
        values.append(((encoded+1)/2).clamp(0,1).permute(0,2,3,1).numpy())
    refs=((originals+1)/2).clamp(0,1).permute(0,2,3,1).numpy()
    maps=[np.stack([deltaE_ciede2000(rgb2lab(a),rgb2lab(b)) for a,b in zip(refs,v)]) for v in values]
    scale=float(np.max(np.stack(maps)))
    fig,axes=plt.subplots(3,5,figsize=(15,9),layout='constrained')
    example_indices=(7,20,42) if args.split=='test' else (0,10,20)
    for row,index in enumerate(example_indices):
        for col,data in enumerate((refs[index],values[0][index],values[1][index],maps[0][index],maps[1][index])):
            ax=axes[row,col]
            im=ax.imshow(data,cmap='magma',vmin=0,vmax=scale) if col>2 else ax.imshow(data)
            ax.set_xticks([])
            ax.set_yticks([])
            if col==0:
                ax.set_ylabel(f'{args.split} image {index}')
            if row==0:
                ax.set_title(['Original','Incumbent Hard Top10',f'+ OKLab, λ={config["global_oklab_weight"]:.5f}','Incumbent CIEDE2000','+ OKLab CIEDE2000'][col],fontsize=10)
    fig.colorbar(im,ax=axes[:,3:].ravel().tolist(),shrink=.6,label='CIEDE2000, shared [0, maximum] scale')
    fig.savefig(FIG/'fixed_color_comparison.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
    for ax,key in zip(axes,('ciede2000_global','ciede2000_top10')):
        for row,label,color in zip((base,cand),('Incumbent',f'+ OKLab, λ={config["global_oklab_weight"]:.5f}'),('#326a9f','#bc5c31')):
            x=np.sort([float(r[key]) for r in per_image if r['method']==row['method']])
            ax.step(x,np.arange(1,51)/50,where='post',label=label,color=color)
        xlabel='Global CIEDE2000' if key=='ciede2000_global' else 'Top10 patch CIEDE2000 (5×5)'
        ax.set(xlabel=xlabel+' (lower better)',ylabel='Fraction of validation images' if args.split=='validation' else 'Fraction of test images')
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(FIG/'color_error_cdf.png',dpi=180)
    plt.close(fig)
    print(status)
    print(reports/'crop_global_oklab_summary.md')


if __name__=='__main__':
    main()
