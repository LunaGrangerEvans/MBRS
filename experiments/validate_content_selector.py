#!/usr/bin/env python3
"""Frozen, validation-only content selector screening; never trains or opens test tensors."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import rankdata, spearmanr
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.analyze_patch_distortion import load_encoded, ssim_per_sample
from experiments.content_selector import content_activity, patches, selected_indices, selector_scores

ROOT = Path('/mnt/wmcontent/GLX/icassp/MBRS')
PROJECT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/content_selector'
CHECKPOINT = ROOT / 'experiments/runs/controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50/checkpoint_0020.pth'
CANDIDATES = [('mse', 0.0)] + [(f, a) for f in ('gradient', 'gradient_hf') for a in (0.5, 1.0, 2.0)]
EXAMPLES = [0, 10, 20, 30, 40]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_manifest():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    split_hashes = {}
    split_names = {}
    for split, count in [('train', 800), ('validation', 50), ('test', 50)]:
        files = sorted((ROOT / 'datasets' / split).glob('*.png'))
        if len(files) != count:
            raise ValueError(f'unexpected {split} count: {len(files)}')
        split_names[split] = [p.name for p in files]
        split_hashes[split] = set()
        for path in files:
            sha = digest(path)
            split_hashes[split].add(sha)
            rows.append(dict(split=split, filename=path.name, path=str(path), sha256=sha))
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if split_hashes[left] & split_hashes[right] or set(split_names[left]) & set(split_names[right]):
            raise ValueError(f'data leakage: {left}/{right}')
    if split_names['validation'] != [f'{i:04d}.png' for i in range(801, 851)]:
        raise ValueError('validation identities differ from frozen protocol')
    write_csv(PROJECT / 'reports/content_selector_split_manifest.csv', rows)
    validation_rows = [r for r in rows if r['split'] == 'validation']
    path = OUT / 'validation_manifest.pt'
    if path.exists():
        manifest = torch.load(path, map_location='cpu', weights_only=False)
        if manifest['sources'] != validation_rows or manifest['seed'] != 170901:
            raise ValueError('existing validation manifest mismatch')
    else:
        images = []
        for row in validation_rows:
            with Image.open(row['path']) as source:
                image = source.convert('RGB').resize((140, 140), Image.Resampling.BILINEAR).crop((6, 6, 134, 134))
                array = np.array(image, dtype=np.float32) / 127.5 - 1
                images.append(torch.from_numpy(array).permute(2, 0, 1))
        generator = torch.Generator().manual_seed(170901)
        manifest = dict(split='validation', sources=validation_rows, seed=170901,
                        images=torch.stack(images), messages=torch.randint(0, 2, (50, 64), generator=generator).float(),
                        examples=EXAMPLES, transform='PIL bilinear resize140 center128 RGB [-1,1]')
        torch.save(manifest, path)
    info = dict(manifest_path=str(path), manifest_sha256=digest(path), checkpoint=str(CHECKPOINT),
                checkpoint_sha256=digest(CHECKPOINT), examples=[split_names['validation'][i] for i in EXAMPLES],
                split_counts={k: len(v) for k, v in split_names.items()},
                protocol_sha256=digest(PROJECT / 'reports/content_selector_preregistered_protocol.md'))
    (OUT / 'provenance.json').write_text(json.dumps(info, indent=2) + '\n')
    return manifest, info


def measure(manifest, device):
    import lpips
    images, messages = manifest['images'], manifest['messages']
    encoded = load_encoded(CHECKPOINT, images, messages, device, 8)
    original_patches = patches(images).flatten(0, 1)
    encoded_patches = patches(encoded).flatten(0, 1)
    mse = (patches(encoded) - patches(images)).square().mean((2, 3, 4))
    activity = content_activity(images)
    metric = lpips.LPIPS(net='alex', version='0.1', spatial=False, verbose=False).to(device).eval()
    local_lpips, local_ssim = [], []
    with torch.no_grad():
        for start in range(0, len(original_patches), 128):
            left = encoded_patches[start:start+128].to(device)
            right = original_patches[start:start+128].to(device)
            local_ssim.append(1 - ssim_per_sample(left, right).cpu())
            local_lpips.append(metric(F.interpolate(left, size=32, mode='bilinear', align_corners=False),
                                       F.interpolate(right, size=32, mode='bilinear', align_corners=False)).flatten().cpu())
        metric.spatial = True
        spatial = []
        for start in range(0, len(images), 8):
            maps = metric(encoded[start:start+8].to(device), images[start:start+8].to(device))
            spatial.append(F.avg_pool2d(maps, 16, stride=8).flatten(1).cpu())
    result = dict(images=images, encoded=encoded, mse=mse, activity=activity,
                  lpips=torch.cat(local_lpips).reshape(50, 225),
                  ssim=torch.cat(local_ssim).reshape(50, 225), spatial_lpips=torch.cat(spatial))
    torch.save(result, OUT / 'validation_patch_tensors.pt')
    return result


def jaccard(left, right, ratio):
    a = selected_indices(left, ratio).numpy()
    b = selected_indices(right, ratio).numpy()
    return np.array([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)])


def correlation(left, right):
    return np.array([float(spearmanr(a, b).statistic) if np.ptp(a) > 0 and np.ptp(b) > 0 else 0.0
                     for a, b in zip(left.numpy(), right.numpy())])


def bootstrap_delta(values, baseline):
    delta = values - baseline
    indices = np.random.default_rng(170902).integers(0, len(delta), (10000, len(delta)))
    low, high = np.quantile(delta[indices].mean(1), [0.025, 0.975])
    return dict(mean=float(delta.mean()), ci95=[float(low), float(high)])


def screen(data):
    arrays, summaries, per_image = {}, [], []
    for feature, alpha in CANDIDATES:
        name = 'mse' if feature == 'mse' else f'{feature}_alpha{alpha:g}'
        scores = selector_scores(data['mse'], data['activity'], feature, alpha)
        indices = selected_indices(scores)
        values = {}
        for target in ('lpips', 'ssim', 'spatial_lpips'):
            values[f'{target}_spearman'] = correlation(scores, data[target])
            for ratio, label in [(0.1, 'top10'), (0.25, 'top25')]:
                values[f'{target}_{label}_jaccard'] = jaccard(scores, data[target], ratio)
        for metric in ('gradient', 'hf', 'variance', 'edge_density'):
            original = data['activity'][metric]
            values[f'selected_{metric}'] = original.gather(1, indices).mean(1).numpy()
            values[f'score_{metric}_spearman'] = correlation(scores, original)
        activity = data['activity']['gradient_normalized']
        if feature == 'gradient_hf':
            activity = (activity + data['activity']['hf_normalized']) / 2
        percentiles = torch.tensor(np.stack([rankdata(a) / len(a) for a in activity.numpy()]))
        values['selected_activity_percentile'] = percentiles.gather(1, indices).mean(1).numpy()
        threshold = torch.quantile(activity, 0.20, dim=1, keepdim=True)
        values['selected_lowest_quintile_fraction'] = (activity.gather(1, indices) <= threshold).float().mean(1).numpy()
        summary = dict(selector=name, feature=feature, alpha=alpha, **{k: float(v.mean()) for k, v in values.items()})
        for i in range(50):
            per_image.append(dict(selector=name, image=f'{801+i:04d}.png', **{k: float(v[i]) for k, v in values.items()}))
        arrays[name] = values
        summaries.append(summary)
    deltas = {}
    for row in summaries[1:]:
        name = row['selector']
        delta = {k: bootstrap_delta(v, arrays['mse'][k]) for k, v in arrays[name].items()}
        deltas[name] = delta
        target_pass = []
        for target in ('lpips', 'ssim'):
            rho, jac = delta[f'{target}_spearman'], delta[f'{target}_top10_jaccard']
            target_pass.append(rho['mean'] >= .03 and rho['ci95'][0] > 0 and jac['mean'] >= .02 and jac['ci95'][0] > 0)
        no_regression = all(delta[f'{t}_spearman']['mean'] >= -.03 and delta[f'{t}_top10_jaccard']['mean'] >= -.02 for t in ('lpips', 'ssim'))
        noncollapse = row['selected_lowest_quintile_fraction'] <= .60 and row['selected_activity_percentile'] >= .25
        bias_reduced = delta['selected_gradient']['mean'] < 0
        row.update(gate_pass=bool(any(target_pass) and no_regression and noncollapse and bias_reduced),
                   gate_lpips=bool(target_pass[0]), gate_ssim=bool(target_pass[1]),
                   no_regression=no_regression, noncollapse=noncollapse, bias_reduced=bias_reduced)
    summaries[0].update(gate_pass=False, gate_lpips=False, gate_ssim=False, no_regression=True, noncollapse=True, bias_reduced=False)
    candidates = [r for r in summaries[1:] if r['gate_pass']] or summaries[1:]
    def both(r):
        return all(deltas[r['selector']][f'{t}_spearman']['mean'] > 0 and deltas[r['selector']][f'{t}_top10_jaccard']['mean'] > 0 for t in ('lpips', 'ssim'))
    best_both = max(both(r) for r in candidates)
    candidates = [r for r in candidates if both(r) == best_both]
    def improvement(r):
        return sum(deltas[r['selector']][f'{t}_spearman']['mean'] for t in ('lpips', 'ssim'))
    best = max(improvement(r) for r in candidates)
    comparable = [r for r in candidates if improvement(r) >= best - .01]
    selected = min(comparable, key=lambda r: (r['feature'] != 'gradient', r['alpha']))
    write_csv(PROJECT / 'reports/content_selector_validation_summary.csv', summaries)
    write_csv(OUT / 'validation_per_image.csv', per_image)
    return dict(summaries=summaries, deltas=deltas, selected=selected, training_justified=selected['gate_pass'])


def mask(scores):
    values = torch.zeros(1, 225)
    values.scatter_(1, selected_indices(scores[None]), 1)
    union = F.fold(values[:, None].expand(-1, 256, -1), (128, 128), 16, stride=8)
    return (union[0, 0] > 0).numpy()


def visualize(data, decision):
    folder = ROOT / 'visualizations/content_selector_validation'
    folder.mkdir(parents=True, exist_ok=True)
    alpha = decision['selected']['alpha']
    gradient = selector_scores(data['mse'], data['activity'], 'gradient', alpha)
    combined = selector_scores(data['mse'], data['activity'], 'gradient_hf', alpha)
    residual = (data['encoded'] - data['images']).abs().mean(1) / 2
    scale = float(residual.max())
    fig, axes = plt.subplots(5, 8, figsize=(20, 12))
    names = ['Original', 'Watermarked', 'Absolute residual', 'MSE Top10', f'Gradient a={alpha:g}', f'Gradient+HF a={alpha:g}', 'LPIPS-proxy Top10', 'SSIM-worst Top10']
    for row, i in enumerate(EXAMPLES):
        original = ((data['images'][i] + 1) / 2).permute(1, 2, 0).clamp(0, 1).numpy()
        encoded = ((data['encoded'][i] + 1) / 2).permute(1, 2, 0).clamp(0, 1).numpy()
        axes[row, 0].imshow(original)
        axes[row, 1].imshow(encoded)
        axes[row, 2].imshow(residual[i], cmap='magma', vmin=0, vmax=scale)
        for col, scores in enumerate([data['mse'], gradient, combined, data['lpips'], data['ssim']], 3):
            axes[row, col].imshow(original)
            overlay = np.zeros((128, 128, 4))
            overlay[:, :, 0] = 1
            overlay[:, :, 3] = mask(scores[i]) * .42
            axes[row, col].imshow(overlay)
        axes[row, 0].set_ylabel(f'{801+i:04d}.png')
        for col in range(8):
            if row == 0:
                axes[row, col].set_title(names[col], fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(f'Predetermined validation images; Patch16/stride8, 23/225 patches; shared residual scale [0,{scale:.4f}]\nRed overlays show patch unions; Jaccard uses patch identities, not union pixels', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    path = folder / 'predetermined_examples.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def reports(decision, info, figure):
    row = decision['selected']
    base = decision['summaries'][0]
    delta = decision['deltas'][row['selector']]
    def format_delta(key):
        return f"{delta[key]['mean']:+.6f}, paired 95% CI [{delta[key]['ci95'][0]:+.6f}, {delta[key]['ci95'][1]:+.6f}]"
    lines = ['# Content-aware selector validation', '', 'Analysis-only; all candidate comparisons use local DIV2K validation IDs 0801–0850. No formal test metrics were read by this script.', '',
             'Frozen definitions and screening thresholds: [preregistered protocol](content_selector_preregistered_protocol.md).', '',
             f"- Manifest SHA-256: `{info['manifest_sha256']}`",
             f"- Checkpoint SHA-256: `{info['checkpoint_sha256']}`",
             '- 50 images, 225 patches/image; image-level paired bootstrap, 10,000 resamples. CIs are descriptive and unadjusted for candidate screening.',
             '- LPIPS here means the 16→32 bilinear patch proxy; native SSIM uses 16×16. Supplemental full-image spatial LPIPS is also recorded.', '',
             '## BASELINE MSE ALIGNMENT', '', f"- LPIPS Spearman {base['lpips_spearman']:.6f}; SSIM Spearman {base['ssim_spearman']:.6f}.",
             f"- Top10 Jaccard: LPIPS {base['lpips_top10_jaccard']:.6f}; SSIM {base['ssim_top10_jaccard']:.6f}.", '',
             '## BEST SELECTOR', '', f"- {row['selector']} ({'selected for one controlled training' if row['gate_pass'] else 'descriptive best only; no training candidate qualifies'}).", '',
             '## BEST ALPHA', '', f"- {row['alpha']:g}; validation-only choice from 0.5, 1, 2.", '',
             '## LPIPS ALIGNMENT CHANGE', '', '- ' + format_delta('lpips_spearman'), '',
             '## SSIM ALIGNMENT CHANGE', '', '- ' + format_delta('ssim_spearman'), '',
             '## TOP10 JACCARD CHANGE', '', '- LPIPS: ' + format_delta('lpips_top10_jaccard'), '- SSIM: ' + format_delta('ssim_top10_jaccard'), '',
             '## CONTENT BIAS CHANGE', '', '- Selected gradient: ' + format_delta('selected_gradient'), '- Selected HF: ' + format_delta('selected_hf'),
             f"- Selected activity percentile {row['selected_activity_percentile']:.3f}; lowest-quintile fraction {row['selected_lowest_quintile_fraction']:.3f}; no-collapse gate {row['noncollapse']}.", '',
             '## QUALITATIVE OBSERVATION', '', f'- Fixed images: {", ".join(info["examples"])}.', f'- Figure: `{figure}`.',
             '- Visual interpretation is recorded separately after inspecting this predetermined panel; candidate selection uses only the frozen numerical rule.', '',
             '## IS TRAINING JUSTIFIED?', '', '- ' + ('YES' if row['gate_pass'] else 'NO'),
             f"- LPIPS gate {row['gate_lpips']}; SSIM gate {row['gate_ssim']}; no material secondary-target regression {row['no_regression']}; reduced gradient bias {row['bias_reduced']}.", '',
             '## All seven candidates', '', '| Selector | rho LPIPS | rho SSIM | J10 LPIPS | J10 SSIM | J25 LPIPS | J25 SSIM | Train gate |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in decision['summaries']:
        lines.append(f"| {r['selector']} | {r['lpips_spearman']:.4f} | {r['ssim_spearman']:.4f} | {r['lpips_top10_jaccard']:.4f} | {r['ssim_top10_jaccard']:.4f} | {r['lpips_top25_jaccard']:.4f} | {r['ssim_top25_jaccard']:.4f} | {r['gate_pass']} |")
    lines += ['', 'Full metrics including selected variance/edge density and contextual LPIPS: [summary CSV](content_selector_validation_summary.csv).',
              f'Per-image values, tensors, bootstrap differences and provenance: `{OUT}`.', '',
              '## Evidence boundary', '', '- One fixed view and one message per validation image; no multi-seed replication or human annotation. Upsampled-patch LPIPS is a diagnostic proxy, not native-16 LPIPS.',
              '- Reduced texture bias alone does not establish better perceptual alignment. Do not claim a content-aware paper method without the conditional controlled training and final test.',
              '- If the gate fails: content normalization does not solve metric alignment under this bounded protocol. Stop normalization tuning; retain the incumbent pixel-tail method.', '']
    (PROJECT / 'reports/content_aware_selector_validation.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    manifest, info = prepare_manifest()
    print('validation manifest frozen', json.dumps(info), flush=True)
    data = measure(manifest, torch.device(args.device))
    decision = screen(data)
    decision['provenance'] = info
    decision['training_launched'] = False
    figure = visualize(data, decision)
    decision['figure'] = figure
    (OUT / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    reports(decision, info, figure)
    print(json.dumps({'best': decision['selected'], 'training_justified': decision['training_justified'], 'figure': figure}, indent=2), flush=True)


if __name__ == '__main__':
    main()
