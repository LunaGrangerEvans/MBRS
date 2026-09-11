"""Read-only verification of the two completed crop/global-OKLab trials."""
import csv
import json
import math
import sys
from pathlib import Path

import torch

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0,str(PROJECT))
from experiments.crop_global_oklab_study import SOURCE, OUT, ROOT, sha  # noqa: E402


def main():
    source=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True)
    calibration=json.loads((OUT/'calibration.json').read_text())
    assert sha(SOURCE)==calibration['source_sha256']
    weight=calibration['global_oklab_weight']
    critical=dict(seed=17,epochs=20,batch_size=16,lr=.0001,message_loss_weight=10,
        global_loss_weight=.5,local_loss_weight=.5,local_loss_mode='topk',local_patch_size=16,
        local_patch_stride=8,local_topk_ratio=.1,noise_layers=['RandomCrop(0.3, 1.0)'],
        require_single_gpu=True,deterministic=True,num_workers=0)
    for suffix,multiplier in (('g25',1),('g12p5',.5)):
        run=f'seed17_crop_hard16_stride8_top10_global_oklab_{suffix}'
        folder=ROOT/'experiments/runs'/run
        cfg=json.loads((folder/'config.resolved.json').read_text())
        assert all(cfg[k]==v for k,v in critical.items())
        assert cfg['init_checkpoint']==str(SOURCE) and cfg['resume'] is None
        assert abs(cfg['global_oklab_weight']-weight*multiplier)<1e-12
        for filename in ('train.jsonl','val.jsonl'):
            logs=[json.loads(line) for line in (folder/filename).read_text().splitlines()]
            assert [r['epoch'] for r in logs]==list(range(1,21))
            assert all(r['samples']==(800 if filename=='train.jsonl' else 50) for r in logs)
            for row in logs:
                assert all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
                expected=sum(row[k] for k in ('weighted_message_loss','weighted_global_loss','weighted_local_loss','weighted_global_oklab'))
                assert abs(row['loss']-expected)<1e-7
        cp=torch.load(folder/'checkpoint_0020.pth',map_location='cpu',weights_only=False,mmap=True)
        assert cp['epoch']==20
        a,b=source['optimizer']['state'],cp['optimizer']['state']
        assert a.keys()==b.keys()
        for key in a:
            assert float(b[key]['step'])-float(a[key]['step'])==1000
        result_folder=OUT/f'validation_{run}'
        result=json.loads((result_folder/'result.json').read_text())
        records=list(csv.DictReader((result_folder/'per_image.csv').open()))
        assert len(records)==150 and len(result['summary'])==3
        for row in result['summary']:
            assert abs(row['global_psnr']+10*math.log10(row['global_mse']))<1e-10
            for ratio in (100,70,50,40,30):
                errors=row[f'ber{ratio}']*16000
                assert abs(errors-round(errors))<1e-8
        print('PASS:',run,'20 epochs; source/Adam counters; loss decomposition; validation records.')
    print('No training/metric rerun performed by this verifier.')


if __name__=='__main__':
    main()
