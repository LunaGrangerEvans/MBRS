"""Bounded global-color study: prepare on training data; evaluate fixed endpoints."""
import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import deltaE_ciede2000, rgb2lab
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from experiments.evaluate_extended_image_quality import evaluate, encode  # noqa: E402
from experiments.losses import image_loss_components  # noqa: E402
from experiments.oklab_global import global_oklab_components  # noqa: E402
from experiments.train_local_patch import seed_everything, configure_determinism  # noqa: E402
from network.Encoder_MP_Decoder import EncoderDecoder  # noqa: E402
from utils.Dataloader import MBRSDataset  # noqa: E402

ROOT = Path('/mnt/wmcontent/GLX/icassp/MBRS')
OUT = ROOT/'reports/crop_global_oklab'
SOURCE = ROOT/'experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth'
BASE_CONFIG = PROJECT/'experiments/config_controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_128_m64.json'
RUN = 'seed17_crop_hard16_stride8_top10_global_oklab_g25'
CONTROLS = {
    'Global continuation': 'controlled_seed17_global_continuation',
    'Hard16 stride8 Top10 (incumbent)': 'controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def model_at(path,device):
    ck = load(path)
    model = EncoderDecoder(128,128,64,['Identity()']).to(device)
    model.load_state_dict(ck['model'])
    return model.eval()


def csv_write(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(device):
    OUT.mkdir(parents=True,exist_ok=True)
    config_path=PROJECT/f'experiments/config_{RUN}.json'
    if config_path.exists() or (OUT/'calibration.json').exists():
        raise FileExistsError('Calibration already frozen; do not overwrite.')
    config=json.loads(BASE_CONFIG.read_text())
    configure_determinism(config)
    seed_everything(17)
    dataset=MBRSDataset(str(ROOT/'datasets/train'),128,128,sort_files=True)
    loader=DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0,
                      generator=torch.Generator().manual_seed(170912))
    model=model_at(SOURCE,device)
    diagnostics=[]
    iterator=iter(loader)
    for batch in range(2):
        images=next(iterator).to(device)
        messages=torch.randint(0,2,(16,64),device=device).float()
        with torch.no_grad():
            encoded=model.encoder(images,messages)
        leaf=encoded.detach().requires_grad_()
        comp=image_loss_components(leaf,images,mode='topk',patch_size=16,patch_stride=8,topk_ratio=.1)
        image_objective=.5*comp['global_mse']+.5*comp['local_mse']
        color=global_oklab_components(leaf,images)['global_oklab']
        grad_image=torch.autograd.grad(image_objective,leaf,retain_graph=True)[0]
        grad_color=torch.autograd.grad(color,leaf)[0]
        diagnostics.append(dict(batch=batch,first_train_index=batch*16,samples=16,
            images_sha256=hashlib.sha256(images.cpu().numpy().tobytes()).hexdigest(),
            image_loss=float(image_objective.detach()),color_loss=float(color.detach()),
            image_gradient_norm=float(grad_image.norm()),color_gradient_norm=float(grad_color.norm()),
            gradient_cosine=float(F.cosine_similarity(grad_image.flatten(),grad_color.flatten(),dim=0))))
    weight=.25*sum(d['image_gradient_norm'] for d in diagnostics)/sum(d['color_gradient_norm'] for d in diagnostics)
    weight=float(f'{weight:.8g}')
    config.update(name=RUN,global_oklab_weight=weight,global_oklab_version='linear_srgb_author_matrix_v1')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    metadata=dict(source=str(SOURCE),source_sha256=sha(SOURCE),training_only_calibration=True,
                  gradient_ratio=.25,global_oklab_weight=weight,batches=diagnostics,
                  original_config=json.loads(BASE_CONFIG.read_text()),config_path=str(config_path),
                  formula='10 message + .5 RGB global + .5 raw-MSE Top10 + lambda global mean OKLab distance')
    (OUT/'calibration.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)


def color_rows(encoded,images):
    rows=[]
    for i in range(len(images)):
        ref=((images[i]+1)/2).clamp(0,1).permute(1,2,0).numpy()
        wm=((encoded[i]+1)/2).clamp(0,1).permute(1,2,0).numpy()
        a,b=rgb2lab(ref),rgb2lab(wm)
        distance=deltaE_ciede2000(a,b).astype(np.float32)
        scores=F.avg_pool2d(torch.from_numpy(distance)[None,None],5,stride=1).flatten().numpy()
        top=np.sort(scores)[-math.ceil(len(scores)*.1):]
        diag=global_oklab_components(encoded[i:i+1],images[i:i+1])
        rows.append(dict(ciede2000_global=float(distance.mean()),ciede2000_patch_mean=float(scores.mean()),
             ciede2000_top10=float(top.mean()),ciede2000_p95=float(np.percentile(scores,95)),
             ciede2000_max=float(scores.max()),signed_cielab_da=float((b-a)[...,1].mean()),
             signed_cielab_db=float((b-a)[...,2].mean()),
             **{k:float(v) for k,v in diag.items()}))
    return rows


def get_manifest(split):
    if split=='test':
        path=ROOT/'reports/uniform_eval_manifest.pt'
        manifest=load(path)
        extension=load(ROOT/'reports/controlled_crop35_40_manifest.pt')
        return manifest,{**manifest['attack_masks'],**extension['attack_masks']},path
    path=ROOT/'reports/content_selector/validation_manifest.pt'
    manifest=load(path)
    assert manifest['split']=='validation'
    mask_path=OUT/'validation_crop_masks.pt'
    if not mask_path.exists():
        rng=np.random.RandomState(170912)
        masks={}
        for ratio in (100,70,50,40,30):
            size=int(128*math.sqrt(ratio/100))
            masks[f'crop_{ratio}']=[]
            for _ in range(5):
                repeat=[]
                for _ in range(4):
                    y,x=(int(rng.randint(0,129-size)),int(rng.randint(0,129-size)))
                    mask=torch.zeros(1,1,128,128)
                    mask[:,:,y:y+size,x:x+size]=1
                    repeat.append(mask)
                masks[f'crop_{ratio}'].append(repeat)
        torch.save(masks,mask_path)
    return manifest,load(mask_path),path


def gate(base,candidate,base_rows,candidate_rows):
    checks={
        'global_psnr':candidate['global_psnr']>=base['global_psnr']-.10,
        'local_psnr':candidate['top25_local_psnr']>=base['top25_local_psnr']-.10,
        'ssim':candidate['global_ssim']>=base['global_ssim']-.001,
        'p95_mse':candidate['patch_mse_p95']<=base['patch_mse_p95']*1.02,
        'gini':candidate['gini']<=base['gini']*1.02,
    }
    for ratio,tol in ((100,.0003125),(70,.0003125),(50,.0005),(40,.001),(30,.001)):
        checks[f'ber{ratio}']=candidate[f'ber{ratio}']<=base[f'ber{ratio}']+tol+1e-12
    fraction=np.mean([r['ciede2000_top10']<b['ciede2000_top10'] for r,b in zip(candidate_rows,base_rows)])
    color_checks=dict(global_color=candidate['ciede2000_global']<=.95*base['ciede2000_global'],
                      top10_color=candidate['ciede2000_top10']<=.95*base['ciede2000_top10'],
                      majority_images=bool(fraction>.5))
    return dict(preservation_checks=checks,color_checks=color_checks,
                preservation_pass=bool(all(checks.values())),color_pass=bool(all(color_checks.values())),
                color_improved_fraction=float(fraction))


def evaluate_stage(split,run,device,controls_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    folder=OUT/f'{split}_{run}'
    folder.mkdir(parents=True,exist_ok=True)
    manifest,masks,path=get_manifest(split)
    images,messages=manifest['images'].float(),manifest['messages'].float()
    methods=copy.copy(CONTROLS)
    if not controls_only:
        methods['Hard16 stride8 Top10 + global OKLab']=run
    summaries,records,provenance=[],[],[]
    for label,name in methods.items():
        cp=ROOT/'experiments/runs'/name/'checkpoint_0020.pth'
        print('Evaluating',split,label,flush=True)
        model=model_at(cp,device)
        encoded=encode(model,images,messages,device,16)
        qr=evaluate(encoded,images,device,16)
        cr=color_rows(encoded,images)
        for i,(a,b) in enumerate(zip(qr,cr)):
            a.pop('global_dists')
            a.pop('global_gmsd')
            a.update(b)
            a.update(method=label,image_index=i,split=split)
        for ratio in (100,70,50,40,30):
            errors=torch.zeros(50,dtype=torch.int64)
            with torch.no_grad():
                for repeat in range(5):
                    for batch,start in enumerate(range(0,50,16)):
                        attacked=encoded[start:start+16].to(device)*masks[f'crop_{ratio}'][repeat][batch].to(device)
                        pred=model.decoder(attacked).cpu()>.5
                        errors[start:start+16]+=(pred!=(messages[start:start+16]>.5)).sum(1)
            for i,r in enumerate(qr):
                r[f'ber{ratio}']=int(errors[i])/(5*64)
        summary={k:float(np.mean([r[k] for r in qr])) for k in qr[0] if k not in ('method','split','image_index')}
        summary['global_psnr']=float(-10*np.log10(summary['global_mse']))
        summary.update(method=label,run=name,split=split)
        records+=qr
        summaries.append(summary)
        torch.save(dict(encoded=encoded,checkpoint_sha256=sha(cp)),folder/f'{name}_outputs.pt')
        provenance.append(dict(run=name,checkpoint=str(cp),checkpoint_sha256=sha(cp)))
        del model
    result=dict(summary=summaries)
    if not controls_only:
        base,cand=summaries[1:]
        br=[r for r in records if r['method']==base['method']]
        pr=[r for r in records if r['method']==cand['method']]
        result['gate']=gate(base,cand,br,pr)
        result['delta_vs_hard']={k:cand[k]-base[k] for k in base if isinstance(base[k],float)}
        result['per_image_improvements']={k:float(np.mean([r[k]<b[k] for r,b in zip(pr,br)])) for k in ('ciede2000_global','ciede2000_top10','ciede2000_p95','patch_mse_p95','gini','full_lpips')}
    csv_write(folder/'per_image.csv',records)
    csv_write(folder/'summary.csv',summaries)
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (folder/'provenance.json').write_text(json.dumps(dict(manifest=str(path),manifest_sha256=sha(path),
          checkpoints=provenance,script_sha256=sha(Path(__file__)),color_module_sha256=sha(PROJECT/'experiments/oklab_global.py')),indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','validation','test'])
    parser.add_argument('--run',default=RUN)
    parser.add_argument('--controls-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.stage=='prepare':
        prepare(device)
    else:
        evaluate_stage(args.stage,args.run,device,args.controls_only)


if __name__=='__main__':
    main()
