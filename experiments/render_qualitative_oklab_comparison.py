#!/usr/bin/env python3
"""Render fixed validation RGB/zoom and matched-PSNR OKLab comparisons."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab


ROOT = Path('/root/workspace/GLX/icassp/MBRS')
MOUNT = Path('/mnt/wmcontent/GLX/icassp/MBRS')
MANIFEST = MOUNT/'reports/content_selector/validation_manifest.pt'
BASE_DIR = MOUNT/'reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5'
OUT = MOUNT/'visualizations/oklab_qualitative_comparison'
SAMPLES = [0, 10, 20, 30, 40]
METHOD_FILES = {
    'Global baseline': BASE_DIR/'controlled_seed17_global_continuation_outputs.pt',
    'Hard Top10': BASE_DIR/'controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_outputs.pt',
    'Hard Top10 + OKLab (Ours)': BASE_DIR/'seed17_crop_hard16_stride8_top10_global_oklab_g12p5_outputs.pt',
}
TARGET_PSNR = [37.0, 40.0]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rgb(tensor):
    return ((tensor.detach().cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()


def psnr(ref, out):
    return float(10 * np.log10(1 / max(float(np.square(ref - out).mean()), 1e-12)))


def load_encoded(path):
    return torch.load(path, map_location='cpu', weights_only=False)['encoded']


def ciede_map(ref, out):
    return deltaE_ciede2000(rgb2lab(ref), rgb2lab(out)).astype(np.float32)


def fixed_zoom_rois(originals, incumbent):
    rois = {}
    for index in SAMPLES:
        color = ciede_map(originals[index], incumbent[index])
        y, x = np.unravel_index(int(np.argmax(color)), color.shape)
        # Select the shared native32 block containing the incumbent's worst 5x5 location.
        rois[index] = (int((y // 32) * 32), int((x // 32) * 32), 32, 32)
    return rois


def matched_output(ref, encoded, target):
    residual = encoded - ref
    low, high = 0.0, 2.0
    for _ in range(45):
        mid = (low + high) / 2
        candidate = np.clip(ref + mid * residual, 0, 1)
        # Increasing strength lowers PSNR.
        if psnr(ref, candidate) > target:
            low = mid
        else:
            high = mid
    alpha = (low + high) / 2
    return np.clip(ref + alpha * residual, 0, 1), alpha


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = torch.load(MANIFEST, map_location='cpu', weights_only=False)
    originals = [rgb(x) for x in manifest['images']]
    encoded = {name: [rgb(x) for x in load_encoded(path)] for name, path in METHOD_FILES.items()}
    rois = fixed_zoom_rois(originals, encoded['Hard Top10'])

    fig, axes = plt.subplots(len(SAMPLES), 4, figsize=(13, 15), squeeze=False)
    for row, index in enumerate(SAMPLES):
        axes[row, 0].imshow(originals[index])
        axes[row, 0].set_title(f'Original\nvalidation index {index:02d}')
        for col, method in enumerate(METHOD_FILES, start=1):
            axes[row, col].imshow(encoded[method][index])
            axes[row, col].set_title(f'{method}\nPSNR {psnr(originals[index], encoded[method][index]):.2f} dB')
        for ax in axes[row]:
            ax.axis('off')
    fig.suptitle('Validation RGB comparison: fixed image and message per row', fontsize=16)
    fig.text(0.5, 0.01, 'OKLab extension: validation-only, λ=0.029574882; no cherry-picking. All methods use the same validation image/message.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(OUT/'qualitative_5x4_main.png', dpi=200)
    fig.savefig(OUT/'qualitative_5x4_main.pdf', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(len(SAMPLES), 4, figsize=(13, 15), squeeze=False)
    for row, index in enumerate(SAMPLES):
        y, x, h, w = rois[index]
        axes[row, 0].imshow(originals[index][y:y+h, x:x+w], interpolation='nearest')
        axes[row, 0].set_title(f'Original zoom\nindex {index:02d}, ROI ({x},{y})')
        for col, method in enumerate(METHOD_FILES, start=1):
            axes[row, col].imshow(encoded[method][index][y:y+h, x:x+w], interpolation='nearest')
            axes[row, col].set_title(f'{method}\nPSNR {psnr(originals[index], encoded[method][index]):.2f} dB')
        for ax in axes[row]:
            ax.axis('off')
    fig.suptitle('Shared local 4× zoom: ROI fixed by incumbent Hard Top10 CIEDE2000 tail', fontsize=15)
    fig.text(0.5, 0.01, 'Each ROI is a shared 32×32 native crop, enlarged with nearest-neighbor display; the selection rule is fixed before comparing methods.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(OUT/'qualitative_5x4_zoom4x.png', dpi=220)
    fig.savefig(OUT/'qualitative_5x4_zoom4x.pdf', dpi=220)
    plt.close(fig)

    rows = []
    for index in SAMPLES:
        ref = originals[index]
        for target in TARGET_PSNR:
            outputs = {}
            for method in ('Hard Top10', 'Hard Top10 + OKLab (Ours)'):
                out, alpha = matched_output(ref, encoded[method][index], target)
                outputs[method] = out
                rows.append({'validation_index': index, 'target_psnr': target, 'method': method,
                             'alpha': alpha, 'actual_psnr': psnr(ref, out)})
    fig, axes = plt.subplots(len(SAMPLES) * len(TARGET_PSNR), 3, figsize=(11, 23), squeeze=False)
    row = 0
    for index in SAMPLES:
        ref = originals[index]
        for target in TARGET_PSNR:
            axes[row, 0].imshow(ref)
            axes[row, 0].set_title(f'Original\nindex {index:02d}, target {target:.0f} dB')
            for col, method in enumerate(('Hard Top10', 'Hard Top10 + OKLab (Ours)'), start=1):
                out, alpha = matched_output(ref, encoded[method][index], target)
                axes[row, col].imshow(out)
                axes[row, col].set_title(f'{method}\nα={alpha:.3f}, PSNR {psnr(ref,out):.2f} dB')
            for ax in axes[row]:
                ax.axis('off')
            row += 1
    fig.suptitle('Matched-PSNR comparison: display-only residual scaling', fontsize=16)
    fig.text(0.5, 0.01, 'For each image and target, α is solved separately for each saved encoder output; no model is retrained. This controls approximate global PSNR before visual comparison.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(OUT/'matched_psnr_hard_vs_oklab.png', dpi=200)
    fig.savefig(OUT/'matched_psnr_hard_vs_oklab.pdf', dpi=200)
    plt.close(fig)

    with (OUT/'matched_psnr.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT/'provenance.json').write_text(json.dumps({
        'manifest': str(MANIFEST), 'manifest_sha256': sha(MANIFEST),
        'sample_indices': SAMPLES, 'target_psnr': TARGET_PSNR,
        'method_outputs': {name: {'path': str(path), 'sha256': sha(path)} for name, path in METHOD_FILES.items()},
        'oklab_run': 'seed17_crop_hard16_stride8_top10_global_oklab_g12p5; validation-only; lambda=0.029574882',
        'roi_rule': 'For each fixed validation image, choose the native32 block containing the maximum 5x5 CIEDE2000 score of incumbent Hard Top10; share that ROI across all methods.',
        'strength_rule': 'clip(original_rgb + alpha * (encoded_rgb - original_rgb), 0, 1); alpha solved by per-image binary search for target PSNR.',
    }, indent=2) + '\n')
    (OUT/'README.md').write_text(
        '# Qualitative Hard Top10 / OKLab comparison\n\n'
        'All rows use the same validation image and fixed message. The OKLab output is validation-only.\n\n'
        '- `qualitative_5x4_main.png/pdf`: Original, Global baseline, Hard Top10, Hard Top10 + OKLab.\n'
        '- `qualitative_5x4_zoom4x.png/pdf`: same shared 32×32 ROIs with nearest-neighbor 4× display.\n'
        '- `matched_psnr_hard_vs_oklab.png/pdf`: Hard Top10 and OKLab matched to 37 and 40 dB by display-only residual scaling.\n'
        '- `matched_psnr.csv`: per-image target, solved alpha and actual PSNR.\n\n'
        'The OKLab model is not a formal-test main method; TrustMark is not included in these four-column rows because this figure is intended to compare MBRS outputs under the same image/message and same RGB representation.\n'
    )
    print(OUT/'qualitative_5x4_main.png')
    print(OUT/'qualitative_5x4_zoom4x.png')
    print(OUT/'matched_psnr_hard_vs_oklab.png')


if __name__ == '__main__':
    main()
