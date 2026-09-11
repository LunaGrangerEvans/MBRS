#!/usr/bin/env python3
"""Render native-resolution fixed validation comparisons with shared zoom insets."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.color import deltaE_ciede2000, rgb2lab


ROOT=Path('/root/workspace/GLX/icassp/MBRS')
MOUNT=Path('/mnt/wmcontent/GLX/icassp/MBRS')
MANIFEST=MOUNT/'reports/content_selector/validation_manifest.pt'
BASE=MOUNT/'reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5'
OUT=MOUNT/'visualizations/native_color_tail_4method'
SAMPLES=[0,10,20,30,40]
METHODS={
    'Original': None,
    'Hard Top10': BASE/'controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_outputs.pt',
    'Hard Top10 + Global OKLab': BASE/'seed17_crop_hard16_stride8_top10_global_oklab_g12p5_outputs.pt',
    'Hard Top10 + Local Chroma-tail': MOUNT/'reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_local_chroma/seed17_crop_hard16_stride8_top10_local_chroma_outputs.pt',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def load_rgb(path):
    if path is None:
        manifest=torch.load(MANIFEST,map_location='cpu',weights_only=False)
        return [((x+1)/2).clamp(0,1).permute(1,2,0).numpy() for x in manifest['images']]
    data=torch.load(path,map_location='cpu',weights_only=False)
    return [((x+1)/2).clamp(0,1).permute(1,2,0).numpy() for x in data['encoded']]


def psnr(ref,out):
    return float(10*np.log10(1/max(float(np.square(ref-out).mean()),1e-12)))


def roi_for(index, ref, hard):
    color=deltaE_ciede2000(rgb2lab(ref[index]),rgb2lab(hard[index]))
    y,x=np.unravel_index(int(np.argmax(color)),color.shape)
    return int((x//32)*32),int((y//32)*32),32,32


def render(indices, images, rois, path, title):
    fig,axes=plt.subplots(len(indices),4,figsize=(14,3.35*len(indices)),squeeze=False,dpi=180)
    for row,index in enumerate(indices):
        x,y,w,h=rois[index]
        for col,(method,values) in enumerate(images.items()):
            ax=axes[row,col]
            ax.imshow(values[index],interpolation='nearest',resample=False)
            if method=='Original':
                label=f'Original\nvalidation index {index:02d}'
            else:
                label=f'{method}\nPSNR {psnr(images["Original"][index],values[index]):.2f} dB'
            ax.set_title(label,fontsize=10)
            ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,edgecolor='#ff3b30',linewidth=1.5))
            inset=inset_axes(ax,width='34%',height='34%',loc='upper right',borderpad=.8)
            inset.imshow(values[index][y:y+h,x:x+w],interpolation='nearest',resample=False)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_edgecolor('#ff3b30')
                spine.set_linewidth(1.5)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title,fontsize=15,y=.995)
    fig.text(.5,.004,'Native 128×128 RGB source elements; nearest-neighbor inset only; red ROI is selected once from incumbent Hard Top10 CIEDE2000 tail and shared by all methods.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.02,1,.98))
    fig.savefig(path,dpi=180,bbox_inches='tight',pad_inches=.05)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    images={method:load_rgb(path) for method,path in METHODS.items()}
    rois={i:roi_for(i,images['Original'],images['Hard Top10']) for i in SAMPLES}
    render(SAMPLES,images,rois,OUT/'qualitative_4method_native_resolution.png',
           'Fixed validation samples: Original / Hard Top10 / Global OKLab / Local Chroma-tail')
    for index in SAMPLES:
        render([index],images,rois,OUT/f'validation_index_{index:02d}_native_zoom.png',
               f'Fixed validation index {index:02d} — shared local color-tail ROI')
    rows=[]
    for index in SAMPLES:
        x,y,w,h=rois[index]
        for method,values in images.items():
            rows.append(dict(validation_index=index,method=method,psnr_db='' if method=='Original' else psnr(images['Original'][index],values[index]),roi_x=x,roi_y=y,roi_size=32))
    with (OUT/'per_sample_psnr_and_roi.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT/'provenance.json').write_text(json.dumps({
        'manifest':str(MANIFEST),'manifest_sha256':sha(MANIFEST),'validation_indices':SAMPLES,
        'method_sources':{method:('manifest' if path is None else {'path':str(path),'sha256':sha(path)}) for method,path in METHODS.items()},
        'roi_rule':'native32 block containing the maximum per-pixel CIEDE2000 location of incumbent Hard Top10; shared across all four columns',
        'display':'source RGB arrays retained at native 128x128; matplotlib nearest/no-resample; PNG lossless',
        'note':'OKLab and local chroma-tail outputs are validation-only; no formal-test claim.',
    },indent=2)+'\n')
    (OUT/'README.md').write_text('# Native-resolution four-method color-tail comparison\n\n'
        'Five fixed validation indices: 0, 10, 20, 30, 40. Columns: Original, Hard Top10, Hard Top10 + Global OKLab, Hard Top10 + Local Chroma-tail.\n\n'
        'All methods use the same source image and message. Red boxes are a shared native32 ROI chosen from the incumbent Hard Top10 CIEDE2000 tail; each cell includes the same ROI as a nearest-neighbor inset. Source arrays are not downsampled or JPEG-compressed.\n')
    print(OUT/'qualitative_4method_native_resolution.png')


if __name__=='__main__':
    main()
