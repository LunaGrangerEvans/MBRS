#!/usr/bin/env python3
"""Summarize the one local-chroma-tail validation run against two validation controls."""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import deltaE_ciede2000, rgb2lab

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0,str(PROJECT))
from experiments.oklab_global import rgb_to_oklab  # noqa: E402

ROOT=Path('/mnt/wmcontent/GLX/icassp/MBRS')
REPORTS=ROOT/'reports'
VAL=ROOT/'reports/content_selector/validation_manifest.pt'
FOLDERS={
    'Hard Top10 incumbent': ROOT/'reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5',
    'Hard Top10 + global OKLab': ROOT/'reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5',
    'Hard Top10 + local chroma-tail': ROOT/'reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_local_chroma',
}
FILES={
    'Hard Top10 incumbent':'controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_outputs.pt',
    'Hard Top10 + global OKLab':'seed17_crop_hard16_stride8_top10_global_oklab_g12p5_outputs.pt',
    'Hard Top10 + local chroma-tail':'seed17_crop_hard16_stride8_top10_local_chroma_outputs.pt',
}


def rgb_lab(x):
    return ((x+1)/2).clamp(0,1)


def local_color(encoded, original):
    encoded_rgb=rgb_lab(encoded)
    original_rgb=rgb_lab(original)
    rows=[]
    for i in range(len(original)):
        ref=original_rgb[i].permute(1,2,0).numpy()
        out=encoded_rgb[i].permute(1,2,0).numpy()
        lab_ref=rgb2lab(ref).astype(np.float32)
        lab_out=rgb2lab(out).astype(np.float32)
        ciede=deltaE_ciede2000(lab_ref,lab_out).astype(np.float32)
        patch_ciede=F.avg_pool2d(torch.from_numpy(ciede)[None,None],5,stride=1).flatten().numpy()
        delta=rgb_to_oklab(encoded_rgb[i:i+1])-rgb_to_oklab(original_rgb[i:i+1])
        chroma=(delta[:,1:3].square().sum(1,keepdim=True)+1e-12).sqrt()
        patch_chroma=F.avg_pool2d(chroma,5,stride=1).flatten().numpy()
        k=math.ceil(len(patch_chroma)*.1)
        rows.append(dict(image_index=i,global_ciede2000=float(ciede.mean()),ciede2000_top10=float(np.sort(patch_ciede)[-k:].mean()),
                         ciede2000_p95=float(np.percentile(patch_ciede,95)),global_chroma=float((chroma-1e-6).mean()),
                         chroma_top10=float(np.sort(patch_chroma)[-k:].mean()),chroma_p95=float(np.percentile(patch_chroma,95))))
    return rows


def main():
    original=torch.load(VAL,map_location='cpu',weights_only=False)['images'].float()
    all_rows=[]
    for method,folder in FOLDERS.items():
        path=folder/FILES[method]
        encoded=torch.load(path,map_location='cpu',weights_only=False)['encoded'].float()
        color=local_color(encoded,original)
        raw_rows=list(csv.DictReader((folder/'per_image.csv').open()))
        expected='Hard16 stride8 Top10 (incumbent)' if method=='Hard Top10 incumbent' else 'Hard16 stride8 Top10 + global OKLab'
        if method=='Hard Top10 + local chroma-tail':
            # The generic validation runner labels the third endpoint with its
            # old global-OKLab display label; use the final 50 records in the
            # candidate folder, and relabel only in this comparison output.
            base_rows=raw_rows[-50:]
        else:
            base_rows=[r for r in raw_rows if r['method']==expected]
        if len(base_rows)!=50:
            raise ValueError((method,len(base_rows)))
        for c,q in zip(color,base_rows):
            all_rows.append(dict(method=method,**c,**{k:float(q[k]) for k in ('global_mse','global_psnr','global_ssim','global_ms_ssim','full_lpips','top25_local_psnr','patch_mse_p95','gini','cv','top10_over_mean','top10_energy_share','ber50','ber40','ber30')}))
    out=REPORTS/'local_chroma_tail_comparison.csv'
    with out.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary=[]
    for method in FOLDERS:
        rows=[r for r in all_rows if r['method']==method]
        item={'method':method,**{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('method','image_index')}}
        # Match the authoritative evaluator: global PSNR comes from the
        # dataset mean MSE, while local PSNR remains mean per-image dB.
        item['global_psnr']=float(-10*np.log10(item['global_mse']))
        summary.append(item)
    with (REPORTS/'local_chroma_tail_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    base=summary[0]
    lines=['# Local chroma-tail comparison (validation only)','',
           'One seed17 continuation was run from the common epoch100 source. The loss keeps the existing Hard RGB tail and adds one conservative Top10 chroma-tail term. These results use the fixed validation manifest and are not formal-test main results.','',
           '| Method | PSNR | Top25 local PSNR | Global CIEDE2000 | CIEDE Top10 | CIEDE P95 | Global chroma | Chroma Top10 | P95 MSE | Gini | BER30 | BER40 | BER50 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('| '+' | '.join([r['method']]+[f'{r[k]:.9f}' for k in ('global_psnr','top25_local_psnr','global_ciede2000','ciede2000_top10','ciede2000_p95','global_chroma','chroma_top10','patch_mse_p95','gini')]+[f'{r[k]:.7f}' for k in ('ber30','ber40','ber50')])+' |')
    lines+=['','## Delta versus incumbent Hard Top10','']
    for r in summary[1:]:
        lines.append(f'### {r["method"]}')
        for k in ('global_psnr','top25_local_psnr','global_ciede2000','ciede2000_top10','global_chroma','chroma_top10','patch_mse_p95','gini','ber30','ber40','ber50'):
            lines.append(f'- {k}: {r[k]-base[k]:+.9f}')
        candidate=[x for x in all_rows if x['method']==r['method']]
        incumbent=[x for x in all_rows if x['method']==base['method']]
        for k in ('ciede2000_top10','chroma_top10','patch_mse_p95','gini'):
            lines.append(f'- images with lower {k}: {100*np.mean([a[k]<b[k] for a,b in zip(candidate,incumbent)]):.1f}%')
    lines+=['','## Interpretation','',
            'The local chroma-tail model is the direct target of this experiment: it selects the largest 5×5 chroma distances inside the same Patch16/stride8/Top10-style local grid and optimizes only that chroma term in addition to the inherited RGB objective. A lower CIEDE2000 or chroma-only value is the desired color-tail direction; Gini and P95 MSE are checked for collateral effects.',
            '', 'No model selection was performed using the formal test manifest. The global-OKLab row is the previously completed validation extension at lambda0.029574882. The comparison is therefore a validation comparison among three frozen endpoints, not a new formal ranking.',
            '', 'Sources: [validation candidates](crop_global_oklab_validation_candidates.csv), [per-image comparison](local_chroma_tail_comparison.csv), [calibration](crop_global_oklab/calibration.json), and the local-chroma checkpoint under `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_crop_hard16_stride8_top10_local_chroma/`.']
    (REPORTS/'local_chroma_tail_comparison.md').write_text('\n'.join(lines)+'\n')
    print((REPORTS/'local_chroma_tail_comparison.md'))


if __name__=='__main__':
    main()
