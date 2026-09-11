"""Read-only content-group diagnostics from saved validation outputs."""
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path('/mnt/wmcontent/GLX/icassp/MBRS')
OUT=ROOT/'reports/crop_global_oklab'


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def patches(x):
    return F.unfold(x,32,stride=32).transpose(1,2).reshape(50,16,x.shape[1],32,32)


def main():
    torch.set_num_threads(4)
    original=((load(ROOT/'reports/content_selector/validation_manifest.pt')['images']+1)/2).clamp(0,1)
    rgb=patches(original)
    gray=(rgb*torch.tensor([.299,.587,.114]).view(1,1,3,1,1)).sum(2)
    brightness=gray.mean((-2,-1))
    activity=((gray[:,:,:-1,1:]-gray[:,:,:-1,:-1]).square()+(gray[:,:,1:,:-1]-gray[:,:,:-1,:-1]).square()).sqrt().mean((-2,-1))
    groups={'darkest_quartile':brightness.argsort(1)[:,:4],
            'brightest_quartile':brightness.argsort(1)[:,-4:],
            'lowest_gradient_quartile':activity.argsort(1)[:,:4],
            'highest_gradient_quartile':activity.argsort(1)[:,-4:]}
    rows=[]
    for suffix in ('g25','g12p5'):
        run=f'seed17_crop_hard16_stride8_top10_global_oklab_{suffix}'
        folder=OUT/f'validation_{run}'
        candidate=folder/f'{run}_outputs.pt'
        if not candidate.exists():
            continue
        reference=folder/'controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_outputs.pt'
        base=((load(reference)['encoded']+1)/2).clamp(0,1)
        proposed=((load(candidate)['encoded']+1)/2).clamp(0,1)
        a=(patches(base)-rgb).square().mean((2,3,4))
        b=(patches(proposed)-rgb).square().mean((2,3,4))
        for group,indices in groups.items():
            before=float(a.gather(1,indices).mean())
            after=float(b.gather(1,indices).mean())
            rows.append(dict(run=run,split='validation',group=group,incumbent_mse=before,
                             candidate_mse=after,percent_change=100*(after/before-1)))
    path=Path(__file__).resolve().parents[1]/'reports/crop_global_oklab_region_diagnostics.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row['run'],row['group'],f'{row["percent_change"]:+.3f}%')


if __name__=='__main__':
    main()
