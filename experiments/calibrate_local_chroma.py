#!/usr/bin/env python3
"""Training-only gradient-scale calibration for one local chroma-tail run."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0,str(PROJECT))
from experiments.losses import image_loss_components
from experiments.oklab_global import chroma_tail_components
from experiments.train_local_patch import configure_determinism, seed_everything
from network.Encoder_MP_Decoder import EncoderDecoder
from utils.Dataloader import MBRSDataset

ROOT=Path('/mnt/wmcontent/GLX/icassp/MBRS')
SOURCE=ROOT/'experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth'
OUT=ROOT/'reports/local_chroma_tail'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    torch.set_num_threads(4)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_everything(17)
    config=json.loads((PROJECT/'experiments/config_controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_128_m64.json').read_text())
    configure_determinism(config)
    dataset=MBRSDataset(str(ROOT/'datasets/train'),128,128,sort_files=True)
    loader=DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(170913))
    checkpoint=torch.load(SOURCE,map_location=device,weights_only=False)
    from network.Encoder_MP_Decoder import EncoderDecoder
    model=EncoderDecoder(128,128,64,['Identity()']).to(device)
    model.load_state_dict(checkpoint['model']); model.eval()
    rows=[]
    iterator=iter(loader)
    for batch in range(2):
        images=next(iterator).to(device)
        messages=torch.randint(0,2,(16,64),device=device).float()
        with torch.no_grad(): encoded=model.encoder(images,messages)
        leaf=encoded.detach().requires_grad_()
        rgb=image_loss_components(leaf,images,mode='topk',patch_size=16,patch_stride=8,topk_ratio=.1)
        chroma=chroma_tail_components(leaf,images,patch_size=16,patch_stride=8,topk_ratio=.1)
        image_loss=.5*rgb['global_mse']+.5*rgb['local_mse']
        grad_image=torch.autograd.grad(image_loss,leaf,retain_graph=True)[0]
        grad_chroma=torch.autograd.grad(chroma['local_chroma'],leaf)[0]
        rows.append(dict(batch=batch,image_gradient_norm=float(grad_image.norm()),chroma_gradient_norm=float(grad_chroma.norm()),
                         cosine=float(torch.nn.functional.cosine_similarity(grad_image.flatten(),grad_chroma.flatten(),dim=0)),
                         image_loss=float(image_loss),chroma_loss=float(chroma['local_chroma'])))
    weight=float(f'{0.15*sum(r["image_gradient_norm"] for r in rows)/sum(r["chroma_gradient_norm"] for r in rows):.8g}')
    OUT.mkdir(parents=True,exist_ok=True)
    data=dict(source=str(SOURCE),source_sha256=sha(SOURCE),target_gradient_ratio=.15,weight=weight,batches=rows,
              fixed={'patch_size':16,'stride':8,'top_ratio':.1,'seed':17,'epochs':20,'lr':.0001})
    (OUT/'calibration.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':
    main()
