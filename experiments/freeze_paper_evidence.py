"""Freeze paper tables from existing checkpoints. No optimizer or training calls."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from experiments.evaluate_extended_image_quality import encode, evaluate, load_file, load_model  # noqa: E402
from experiments.rebuild_research_tables import SOURCE, master  # noqa: E402

ROOT = Path('/mnt/wmcontent/GLX/icassp/MBRS')
OUT = ROOT / 'reports/paper_freeze_v1'
PAPER = PROJECT / 'paper'
IDS = [0, 1, 2, 3, 4, 7, 8, 9]
NAMES = ['Global continuation (RGB image weight 1)',
         'Hard MSE, patch16 / stride16 / Top25%',
         'Hard MSE, patch16 / stride8 / Top25%',
         'Hard MSE, patch16 / stride8 / Top10% (main)',
         'Soft normalized weighting, patch16 / stride16 / T=0.25',
         'Multi-scale hard Top25%, patch16/stride16 (0.7) + patch32/stride32 (0.3)',
         'Excess, patch16 / stride16 / threshold1 / scale3',
         'Gradient-aware MSE, patch16 / stride8 / Top10% / alpha2']
METRICS = ['global_psnr', 'global_ssim', 'global_ms_ssim', 'full_lpips',
           'top25_local_psnr', 'patch_mse_p95', 'gini', 'cv',
           'top10_over_mean', 'top10_energy_share', 'patch_ssim_bottom10_mean',
           'patch_lpips_top10_mean', 'ssim_tail_gap']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    result = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
    result['global_psnr'] = float(-10 * np.log10(np.mean([r['global_mse'] for r in rows])))
    return result


def main():
    torch.set_num_threads(4)
    torch.manual_seed(17)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(exist_ok=True)
    manifest_path = ROOT / 'reports/uniform_eval_manifest.pt'
    ext_path = ROOT / 'reports/controlled_crop35_40_manifest.pt'
    manifest, extension = load_file(manifest_path, 'cpu'), load_file(ext_path, 'cpu')
    assert sha(manifest_path) == '36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339'
    assert sha(ext_path) == 'ffa19caecb5d97e717208d3aad7755e67d4b027d79c7c405e83b82bc6a995dde'
    images, messages = manifest['images'].float(), manifest['messages'].float()
    assert tuple(images.shape) == (50, 3, 128, 128)
    registry = master()
    all_rows, table, provenance = [], [], []
    masks = {**manifest['attack_masks'], **extension['attack_masks']}
    for idx, label in zip(IDS, NAMES):
        info = registry[idx]
        print('Frozen inference:', label, flush=True)
        model = load_model(info['method'], device)
        encoded = encode(model, images, messages, device, 16)
        records = evaluate(encoded, images, device, 16)
        for record in records:
            # Remove unavailable optional metrics rather than publish NaN as evidence.
            record.pop('global_dists')
            record.pop('global_gmsd')
            record['method'] = label
        stats = summarize(records)
        for ratio in (100, 70, 50, 40, 30):
            errors = torch.zeros(50, dtype=torch.int64)
            with torch.no_grad():
                for repeat in range(5):
                    for batch, start in enumerate(range(0, 50, 16)):
                        mask = masks[f'crop_{ratio}'][repeat][batch].to(device)
                        prediction = model.decoder(encoded[start:start+16].to(device) * mask).cpu() > .5
                        errors[start:start+16] += (prediction != (messages[start:start+16] > .5)).sum(1)
            stats[f'ber{ratio}'] = int(errors.sum()) / (50 * 64 * 5)
            # Integer-count replay verifies published raw tensor robustness.
            assert abs(stats[f'ber{ratio}'] - float(info[f'ber{ratio}'])) < 1e-7
            for i, record in enumerate(records):
                record[f'ber{ratio}'] = int(errors[i]) / (64 * 5)
        stats['method'] = label
        stats['run'] = info['method']
        table.append(stats)
        all_rows.extend(records)
        torch.save({'encoded': encoded, 'checkpoint': info['checkpoint'],
                    'checkpoint_sha256': info['checkpoint_sha256']}, OUT / f'{idx:02d}_outputs.pt')
        provenance.append({k: info[k] for k in ('method', 'description', 'checkpoint',
                           'checkpoint_sha256', 'evaluation_json', 'evaluation_sha256', 'source_checkpoint')})
        del model
    csv_write(OUT / 'per_image.csv', all_rows)
    csv_write(PAPER / 'main_table.csv', table)
    base = table[0]
    columns = ['global_psnr', 'global_ssim', 'global_ms_ssim', 'full_lpips',
               'top25_local_psnr', 'patch_mse_p95', 'gini', 'top10_over_mean', 'ber50', 'ber30']
    headings = ['PSNR ↑', 'SSIM ↑', '3-scale MS-SSIM ↑', 'LPIPS ↓',
                'Top25 local PSNR ↑', 'P95 MSE ↓', 'Gini ↓', 'Top10/Mean ↓', 'BER50 ↓', 'BER30 ↓']
    intro = ['# Frozen main table', '',
             'Eight existing seed17 epoch20 checkpoints, one shared epoch100 source. Image weights are global/local 0.5/0.5 except Global 1/0; message MSE weight 10; LR=1e-4; batch16. No new training.', '',
             'All quality values use the current clipped-RGB evaluator. PSNR is computed from dataset mean MSE. Top25 local PSNR is the **mean of per-image** PSNR of the four highest-MSE native32 patches. P95 is the mean per-image P95, not a pooled percentile. Gini and other concentration metrics use 16 native32 non-overlap patches. BER replays the frozen unquantized normalized-output mask protocol. Units and aggregation are defined in draft §5.', '',
             '| Method | ' + ' | '.join(headings) + ' |', '|---|' + '---:|' * len(columns)]
    for i, record in enumerate(table):
        name = f'**{record["method"]}**' if i in (0, 3) else record['method']
        intro.append('| ' + name + ' | ' + ' | '.join(f'{record[k]:.9f}' if k in ('patch_mse_p95','full_lpips') else f'{record[k]:.6f}' for k in columns) + ' |')
    intro += ['', 'Bold identifies the fixed primary comparison, not column winners. Soft T=0.25 is the historical best local-pixel-tail Soft setting, not a new selection. The CSV additionally contains CV, energy share, perceptual tails, all five BER values, and run identifiers.', '',
              'Source: [main_table.csv](main_table.csv), [per-image evidence](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv), [provenance](evidence_manifest.json).']
    (PAPER / 'main_table.md').write_text('\n'.join(intro)+'\n')
    ablation = ['# Frozen ablations', '', 'All deltas are candidate minus Global under the same frozen evaluator. These are existing configurations, not a new search.', '',
                '| Configuration | ΔPSNR | ΔTop25 local PSNR | Δlocal PSNR − Δglobal PSNR | ΔP95 MSE | ΔGini | ΔBER30 |', '|---|---:|---:|---:|---:|---:|---:|']
    for record in table[1:]:
        d = {k: record[k]-base[k] for k in columns}
        ablation.append(f'| {record["method"]} | {d["global_psnr"]:+.6f} | {d["top25_local_psnr"]:+.6f} | {d["top25_local_psnr"]-d["global_psnr"]:+.6f} | {d["patch_mse_p95"]:+.9f} | {d["gini"]:+.6f} | {d["ber30"]:+.7f} |')
    ablation += ['', 'The difference of the two PSNR deltas is descriptive: their aggregation differs, so it is not a matched-PSNR causal estimate. Training overlap and selection ratio are ablated through the listed controlled pairs.', '',
                 'Soft: detached softmax of per-image mean-normalized patch MSE / T. Multi-scale: 0.7×patch16 + 0.3×patch32 hard Top25 losses. Excess: 3×detached patch-mean MSE×mean(ReLU(patch MSE / detached mean − 1)²). Gradient-aware: original-image activity only changes ranking; selected raw patch MSE remains the optimized loss.']
    (PAPER / 'ablation_table.md').write_text('\n'.join(ablation)+'\n')
    paired = {}
    a = [r for r in all_rows if r['method']==NAMES[0]]
    b = [r for r in all_rows if r['method']==NAMES[3]]
    for k in METRICS:
        delta = np.array([rb[k]-ra[k] for ra, rb in zip(a,b)])
        paired[k] = dict(mean_delta=float(delta.mean()), median_delta=float(np.median(delta)),
                         percent_lower=float(100*(delta<0).mean()), percent_higher=float(100*(delta>0).mean()))
    (PAPER / 'paired_diagnostics.json').write_text(json.dumps(paired, indent=2)+'\n')
    tm_records = list(csv.DictReader((ROOT/'reports/external_baseline_metrics/per_image.csv').open()))
    tm_raw_path = PROJECT/'reports/trustmark_raw_bit_results.csv'
    tm_raw = list(csv.DictReader(tm_raw_path.open()))
    external = []
    for method in ('TrustMark Q','TrustMark P'):
        recs = [{k: float(v) for k,v in r.items() if k not in ('method',)} for r in tm_records if r['method']==method]
        row = {'method':method,'comparison':'REFERENCE','payload':'61 data + 35 parity + 4 schema bits',**summarize(recs)}
        for ratio in (100,70,50,40,30):
            r = next(r for r in tm_raw if r['method']==method and r['crop_ratio']==f'{ratio}%')
            for key in ('raw_ber','raw_bit_accuracy','exact_decode_success','detection_success'):
                row[f'{key}_{ratio}'] = float(r[key])
        external.append(row)
    csv_write(PAPER / 'external_reference_table.csv', external)
    lines = ['# External reference table', '',
             'TrustMark Q/P: official released models, strength unchanged, 100 transmitted bits including 61 data bits, BCH parity and schema. Every cross-family comparison is **REFERENCE**. PNG output and native decoder resizing differ from MBRS tensor inference. No strict ranking is made.', '',
             '| Method | PSNR ↑ | Top25 local PSNR ↑ | Gini ↓ | Full LPIPS ↓ |', '|---|---:|---:|---:|---:|']
    for r in [table[0],table[3],*external]:
        lines.append(f'| {r["method"]} | {r["global_psnr"]:.6f} | {r["top25_local_psnr"]:.6f} | {r["gini"]:.6f} | {r["full_lpips"]:.9f} |')
    lines += ['', '| Method | Retained area | Pre-ECC packet BER ↓ | Packet bit accuracy ↑ | ECC exact message ↑ | Decode flag ↑ |', '|---|---:|---:|---:|---:|---:|']
    for r in external:
        for ratio in (100,70,50,40,30):
            lines.append(f'| {r["method"]} | {ratio}% | {r[f"raw_ber_{ratio}"]:.6f} | {r[f"raw_bit_accuracy_{ratio}"]:.6f} | {r[f"exact_decode_success_{ratio}"]:.6f} | {r[f"detection_success_{ratio}"]:.6f} |')
    lines += ['', 'Quality recomputed by aggregation from saved per-image records: older TrustMark PSNR used mean image PSNR; here it uses dataset mean MSE exactly as MBRS. No encoder or strength change. Raw-bit/detection records are replayed from the verified official pre-ECC audit, not newly optimized. The decode flag is not evidence of correct payload or a calibrated detection rate on unwatermarked images.', '',
              'StegaStamp: BLOCKED (no verified checkpoint in the audited environment). HiDDeN: REQUIRES RETRAINING in the audited setup. Neither receives numeric performance claims.']
    (PAPER/'external_reference_table.md').write_text('\n'.join(lines)+'\n')
    sources = [manifest_path,ext_path,SOURCE,Path(__file__),PROJECT/'experiments/evaluate_extended_image_quality.py',
               PROJECT/'experiments/losses.py',ROOT/'reports/external_baseline_metrics/per_image.csv',tm_raw_path]
    manifest_data = {'protocol':'paper-freeze-v1; current clipped RGB evaluator', 'training':False,
                     'source_hashes':{str(p):sha(p) for p in sources}, 'checkpoints':provenance,
                     'artifacts': {str(p):sha(p) for p in [OUT/'per_image.csv',PAPER/'main_table.csv',PAPER/'external_reference_table.csv']}}
    (PAPER/'evidence_manifest.json').write_text(json.dumps(manifest_data,indent=2,ensure_ascii=False)+'\n')
    print('FROZEN:',len(table),'controlled rows;',len(all_rows),'image records; BER replay matches published integer counts.',flush=True)


if __name__ == '__main__':
    main()
