#!/usr/bin/env python3
"""Render fixed RGB baseline/ours comparisons and residual-strength sweeps."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


ROOT = Path('/root/workspace/GLX/icassp/MBRS')
MOUNT = Path('/mnt/wmcontent/GLX/icassp/MBRS')
MANIFEST = MOUNT/'reports/uniform_eval_manifest.pt'
SAMPLES = [0, 10, 20, 30, 40]
OUT = MOUNT/'visualizations/visual_comparison_strength'
OUTPUTS = MOUNT/'reports/paper_freeze_v1'
METHODS = {
    'MBRS Global': OUTPUTS/'00_outputs.pt',
    'MBRS Hard Top25': OUTPUTS/'02_outputs.pt',
    'TrustMark Q (REF)': MOUNT/'external_baselines/outputs/trustmark_Q',
    'TrustMark P (REF)': MOUNT/'external_baselines/outputs/trustmark_P',
    'Ours: MBRS Hard Top10': OUTPUTS/'03_outputs.pt',
}
STRENGTHS = [0.25, 0.50, 0.75, 1.00]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tensor_rgb(tensor):
    return ((tensor.detach().cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()


def psnr(reference, output):
    mse = float(np.square(reference - output).mean())
    return 10 * np.log10(1 / max(mse, 1e-12))


def load_method(path):
    if path.is_dir():
        return [np.asarray(Image.open(path/f'image_{i:03d}.png').convert('RGB'), dtype=np.float32)/255 for i in range(50)]
    data = torch.load(path, map_location='cpu', weights_only=False)
    return [tensor_rgb(x) for x in data['encoded']]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = torch.load(MANIFEST, map_location='cpu', weights_only=False)
    originals = [tensor_rgb(x) for x in manifest['images']]
    loaded = {name: load_method(path) for name, path in METHODS.items()}

    fig, axes = plt.subplots(len(SAMPLES), 6, figsize=(18, 15), squeeze=False)
    for row, index in enumerate(SAMPLES):
        axes[row, 0].imshow(originals[index])
        axes[row, 0].set_title(f'Original\nformal index {index:02d}')
        for col, (name, values) in enumerate(loaded.items(), start=1):
            axes[row, col].imshow(values[index])
            score = psnr(originals[index], values[index])
            axes[row, col].set_title(f'{name}\nPSNR {score:.2f} dB')
        for ax in axes[row]:
            ax.axis('off')
    fig.suptitle('Fixed RGB watermark comparison — five predetermined formal-test images', fontsize=16)
    fig.text(0.5, 0.01, 'TrustMark Q/P are reference outputs with different payload/ECC and preprocessing; visual comparison is qualitative.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(OUT/'five_sample_baseline_ours_comparison.png', dpi=180)
    fig.savefig(OUT/'five_sample_baseline_ours_comparison.pdf', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(SAMPLES), 9, figsize=(23, 14), squeeze=False)
    global_rgb = loaded['MBRS Global']
    ours_rgb = loaded['Ours: MBRS Hard Top10']
    for row, index in enumerate(SAMPLES):
        axes[row, 0].imshow(originals[index])
        axes[row, 0].set_title(f'Original\nindex {index:02d}')
        col = 1
        for name, values in [('Global', global_rgb), ('Ours', ours_rgb)]:
            for alpha in STRENGTHS:
                image = np.clip(originals[index] + alpha * (values[index] - originals[index]), 0, 1)
                axes[row, col].imshow(image)
                axes[row, col].set_title(f'{name} α={alpha:g}\n{psnr(originals[index], image):.2f} dB')
                col += 1
        for ax in axes[row]:
            ax.axis('off')
    fig.suptitle('Watermark-strength sweep: residual scaling changes PSNR without retraining', fontsize=16)
    fig.text(0.5, 0.01, 'Display-only sweep: output = clip(original + α × (encoded − original)); α=1 is the saved checkpoint output.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(OUT/'global_vs_ours_strength_sweep.png', dpi=180)
    fig.savefig(OUT/'global_vs_ours_strength_sweep.pdf', dpi=180)
    plt.close(fig)

    strength_rows = []
    for method, values in [('MBRS Global', global_rgb), ('Ours: MBRS Hard Top10', ours_rgb)]:
        for index in SAMPLES:
            for alpha in [0.0, *STRENGTHS]:
                image = originals[index] if alpha == 0 else np.clip(originals[index] + alpha * (values[index] - originals[index]), 0, 1)
                strength_rows.append({'method': method, 'formal_index': index, 'alpha': alpha, 'psnr_db': psnr(originals[index], image)})
    import csv
    with (OUT/'strength_sweep_psnr.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(strength_rows[0]))
        writer.writeheader()
        writer.writerows(strength_rows)
    provenance = {
        'manifest': str(MANIFEST), 'manifest_sha256': digest(MANIFEST),
        'sample_indices': SAMPLES, 'strengths': STRENGTHS,
        'method_sources': {name: {'path': str(path), 'sha256': digest(path) if path.is_file() else 'directory_outputs'} for name, path in METHODS.items()},
        'display_transform': 'clip(original_rgb + alpha * (encoded_rgb - original_rgb), 0, 1)',
        'notes': 'Five indices fixed before rendering: 0,10,20,30,40. TrustMark is qualitative/reference only.',
    }
    (OUT/'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(OUT/'five_sample_baseline_ours_comparison.png')
    print(OUT/'global_vs_ours_strength_sweep.png')


if __name__ == '__main__':
    main()
