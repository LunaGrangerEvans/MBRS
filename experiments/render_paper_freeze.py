"""Refresh only the selected draft figures from frozen tables and outputs."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from experiments.losses import patch_mse_per_sample  # noqa: E402

PAPER = PROJECT/'paper'
ROOT = Path('/mnt/wmcontent/GLX/icassp/MBRS')
EVIDENCE = ROOT/'reports/paper_freeze_v1'
OUT = ROOT/'visualizations/paper_freeze_v1'
COLORS = ['#326a9f', '#bc5c31']
LABELS = ['Global continuation', 'Hard P16 / S8 / Top10%']


def rows(path):
    return list(csv.DictReader(path.open()))


def save(fig, name):
    for suffix in ('png','pdf'):
        fig.savefig(OUT/f'{name}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    table = rows(PAPER/'main_table.csv')
    pair = [table[0],table[3]]
    individual = rows(EVIDENCE/'per_image.csv')
    groups = [[r for r in individual if r['method']==s['method']] for s in pair]
    fig, ax = plt.subplots(figsize=(13,5))
    ax.set(xlim=(0,13),ylim=(0,5))
    ax.axis('off')
    def box(x,y,text,color='#eef3f7'):
        ax.text(x,y,text,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',facecolor=color,edgecolor='#536477'),fontsize=10)
    def arrow(a,b):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#536477',lw=1.5))
    box(1,3.8,'Host x\n128×128 RGB\n+ 64-bit message')
    box(3.5,3.8,'MBRS\nencoder')
    box(5.7,3.8,'Clean output\ny = E(x,m)')
    box(8.3,3.8,'Rectangle mask\nz = M × y')
    box(11,3.8,'Decoder scores\nMessage MSE')
    for a,b in [(1.9,2.9),(4.1,4.9),(6.5,7.3),(9.3,10.1)]:
        arrow((a,3.8),(b,3.8))
    box(3,1.9,'Compare clean y with x\nGlobal RGB MSE', '#e6eff9')
    box(7.5,1.9,'Patch16, stride8 → 225 scores: MSE(y_i, x_i)\nSelect highest 23 scores (Top10%)\nLocal loss = mean selected raw MSE', '#f9eade')
    arrow((5.7,3.3),(3,2.5))
    arrow((5.7,3.3),(7.5,2.6))
    box(6.5,.45,'Main: 10 L_message + 0.5 L_global + 0.5 L_tail\nControl: 10 L_message + L_global', '#eef0ec')
    arrow((3,1.4),(5,.9))
    arrow((7.5,1.25),(7,.9))
    ax.text(11,1.8,'Local branch:\ntraining only\n\nEvaluation grid:\n32×32, stride32\n16 patches/image',ha='center',va='center',fontsize=10)
    ax.set_title('Fine-grained overlapping hard-tail supervision',fontsize=14)
    save(fig,'fig01_method')

    fig, axes = plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,key,title in zip(axes[:2],['global_psnr','top25_local_psnr'],['Clean global PSNR','Clean Top25 local PSNR']):
        for i,r in enumerate(pair):
            ax.scatter(i,float(r[key]),s=70,color=COLORS[i])
            ax.annotate(f'{float(r[key]):.3f}',(i,float(r[key])),xytext=(0,9),textcoords='offset points',ha='center')
        ax.set_xticks([0,1],['Global','Hard Top10'])
        ax.set(xlim=(-.5,1.5),ylabel='dB (higher is better)',title=title)
        ax.margins(y=.35)
        ax.grid(alpha=.2)
    ratios=[30,40,50,70,100]
    for r,label,c in zip(pair,LABELS,COLORS):
        axes[2].plot(ratios,[float(r[f'ber{a}']) for a in ratios],marker='o',label=label,color=c,alpha=.8)
    axes[2].set(xlabel='Nominal retained area (%)',ylabel='Raw 64-bit BER',title='Fixed rectangle-mask protocol')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=.2)
    save(fig,'fig03_quality_robustness')

    manifest= torch.load(ROOT/'reports/uniform_eval_manifest.pt',weights_only=False,map_location='cpu')
    original=((manifest['images']+1)/2).clamp(0,1)
    fig, axes = plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
    for idx,label,c in zip((0,3),LABELS,COLORS):
        encoded=torch.load(EVIDENCE/f'{idx:02d}_outputs.pt',weights_only=False,map_location='cpu')['encoded']
        scores=patch_mse_per_sample(((encoded+1)/2).clamp(0,1),original,32).numpy()
        ordered=np.sort(scores,axis=1)
        lorenz=np.c_[np.zeros(50),np.cumsum(ordered,axis=1)/ordered.sum(1,keepdims=True)]
        axes[0].plot(np.linspace(0,1,17),lorenz.mean(0),color=c,label=label)
    axes[0].plot([0,1],[0,1],'--',color='gray',label='Equal patch energy')
    axes[0].set(xlabel='Cumulative fraction of patches',ylabel='Cumulative fraction of MSE',title='Mean per-image Lorenz curve')
    axes[0].legend(fontsize=8)
    av=np.array([float(r['gini']) for r in groups[0]])
    bv=np.array([float(r['gini']) for r in groups[1]])
    lim=max(av.max(),bv.max())*1.05
    axes[1].scatter(av,bv,s=23,color=COLORS[1])
    axes[1].plot([0,lim],[0,lim],'--',color='gray')
    axes[1].set(xlabel='Global Gini',ylabel='Hard Top10 Gini',title=f'Lower Gini on {(bv<av).mean():.0%} of images')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Clipped RGB; native32 non-overlapping evaluation patches')
    save(fig,'fig04_concentration')

    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for ax,key,title in zip(axes,['patch_mse_p95','patch_lpips_top10_mean'],['Pixel tail: per-image P95 MSE','Perceptual tail: native32 Top10 LPIPS']):
        delta=np.array([float(b[key])-float(a[key]) for a,b in zip(*groups)])
        ax.scatter(np.arange(50),delta,c=np.where(delta<0,COLORS[0],COLORS[1]),s=21)
        ax.axhline(0,color='gray',linestyle='--')
        ax.set(xlabel='Fixed image index (all 50)',ylabel='Hard minus Global (lower is better)',
               title=title+f'\nLower: {(delta<0).mean():.0%}; mean change: {delta.mean():+.2e}')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax.grid(alpha=.2)
    save(fig,'fig05_pixel_perceptual')

    refs=rows(PAPER/'external_reference_table.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    crop_colors={70:'#336ea1',50:'#e19333',40:'#388c6c',30:'#aa4773'}
    markers=['o','s','^','D']
    names=['MBRS Global','MBRS Hard Top10','TrustMark Q (REF)','TrustMark P (REF)']
    for ax,key,title in zip(axes,['global_psnr','full_lpips'],['PSNR ↑','Full LPIPS ↓']):
        for method,(r,marker,name) in enumerate(zip([*pair,*refs],markers,names)):
            for crop,c in crop_colors.items():
                acc=1-float(r[f'ber{crop}']) if method<2 else float(r[f'raw_bit_accuracy_{crop}'])
                ax.scatter(float(r[key]),acc,marker=marker,color=c,s=55)
        ax.set(xlabel=title,ylabel='Pre-ECC bit accuracy ↑',ylim=(.70,1.01))
        ax.grid(alpha=.2)
    handles=[plt.Line2D([],[],ls='',marker=m,color='#555',label=n) for m,n in zip(markers,names)]
    handles += [plt.Line2D([],[],ls='',marker='o',color=c,label=f'{a}% retained') for a,c in crop_colors.items()]
    fig.legend(handles=handles,loc='outside lower center',ncols=4,fontsize=8)
    fig.suptitle('REFERENCE only: 64 uncoded MBRS bits vs 100 transmitted TrustMark packet bits')
    save(fig,'figS1_external_reference')
    files=[PAPER/'main_table.csv',PAPER/'external_reference_table.csv',EVIDENCE/'per_image.csv',Path(__file__)]
    files += list(OUT.glob('*.png')) + list(OUT.glob('*.pdf'))
    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'provenance.json').write_text(json.dumps({str(p):digest(p) for p in files},indent=2)+'\n')
    print('Saved five selected draft figures (PNG/PDF); Fig2 reuses existing fixed RGB panels.')


if __name__ == '__main__':
    main()
