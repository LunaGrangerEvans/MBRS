"""Rebuild research tables from existing records; no training or model forward."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
ROOT = Path('/mnt/wmcontent/GLX/icassp/MBRS')
REPORTS = PROJECT / 'reports'
RUNS = ROOT / 'experiments/runs'
SOURCE = RUNS / 'optimization_global_seed17_128_m64_crop/checkpoint_0100.pth'
LATEST = 'controlled_seed17_gradientaware_alpha2_metrics.json'
SOFT = 'controlled_soft_temperature_seed17.json'
# These are artifact identities; configuration descriptions are recovered below.
REGISTRY = [
    (LATEST, 'Global continuation', 'controlled_seed17_global_continuation'),
    (LATEST, 'Hard P16-T25-L50', 'controlled_seed17_hard_p16_t25_l50'),
    (LATEST, 'Hard patch16 stride8 top25 global0.5 local0.5', 'controlled_seed17_hard_patch16_stride8_top25_weight50'),
    (LATEST, 'Hard patch16 stride8 top10 global0.5 local0.5', 'controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50'),
    (SOFT, 'Soft P16-T0.25-L50', 'controlled_seed17_soft_p16_t025_l50'),
    (SOFT, 'Soft P16-T0.5-L50', 'controlled_seed17_soft_p16_t05_l50'),
    (SOFT, 'Soft P16-T1.0-L50', 'controlled_seed17_soft_p16_t10_l50'),
    (LATEST, 'Multiscale patch16 top25 weight0.7 plus patch32 top25 weight0.3 global0.5 local0.5', 'controlled_seed17_multiscale_patch16_patch32_top25_weight70_weight30_global_weight50_local_weight50'),
    (LATEST, 'Excess patch16 stride16 threshold1 scale3 global0.5 local0.5', 'controlled_seed17_excess_patch16_stride16_global_weight50_local_weight50'),
    (LATEST, 'Gradient-aware patch16 stride8 top10 alpha2 global0.5 local0.5', 'seed17_contentaware_gradient_patch16_stride8_top10_alpha2_global0.5_local0.5'),
]
EXPECTED = dict(seed=17, H=128, W=128, message_length=64, epochs=20,
                lr=0.0001, batch_size=16, num_workers=0, message_loss_weight=10.0,
                deterministic=True, require_single_gpu=True)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError:
        return torch.load(path, map_location='cpu', weights_only=False)


def describe(c):
    mode = c['local_loss_mode']
    p, s = c['local_patch_size'], c.get('local_patch_stride', c['local_patch_size'])
    if mode == 'none':
        local = '仅整图MSE，无局部项'
    elif mode == 'mean':
        local = f'{p}×{p}/步长{s}，全部块平均MSE'
    elif mode == 'multiscale':
        local = ' + '.join(f'{w:g}×{p}×{p}/步长{s} Top{c["local_topk_ratio"]*100:g}%'
                           for p, s, w in zip(c['multi_scale_patch_sizes'], c['multi_scale_patch_strides'], c['multi_scale_weights']))
    elif mode == 'soft_tail':
        local = f'{p}×{p}/步长{s}，归一化softmax，温度{c["soft_tail_temperature"]:g}，detach={c.get("soft_tail_detach_weights", True)}'
    elif mode == 'excess':
        local = f'{p}×{p}/步长{s}，超均值平方惩罚，阈值{c.get("excess_threshold",1):g}，倍率{c.get("excess_loss_scale",1):g}'
    else:
        local = f'{p}×{p}/步长{s}，Top{c["local_topk_ratio"]*100:g}%原始MSE均值'
        if mode == 'contentaware_gradient':
            local += f'，按梯度分数选块，alpha={c.get("content_selector_alpha",1):g}'
    return f'{local}；整图/局部权重={c["global_loss_weight"]:g}/{c["local_loss_weight"]:g}'


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    result = stream.getvalue()
    assert all(None not in row for row in csv.DictReader(io.StringIO(result)))
    return result


def publish(name, text):
    path = REPORTS / name
    if path.exists() and path.read_text() != text:
        old = path.read_bytes()
        archive = REPORTS / 'audit_history' / (path.stem + '-' + hashlib.sha256(old).hexdigest()[:12] + path.suffix)
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():
            archive.write_bytes(old)
    path.write_text(text, encoding='utf-8')


def master():
    artifacts = {file: read_json(ROOT / 'reports' / file) for file in (LATEST, SOFT)}
    baseline = artifacts[LATEST]['metrics']['Global continuation']
    rows = []
    for file, key, run in REGISTRY:
        folder = RUNS / run
        resolved = read_json(folder / 'config.resolved.json')
        ck_path = folder / 'checkpoint_0020.pth'
        ck = checkpoint(ck_path)
        c = ck['config']
        assert ck['epoch'] == 20 and 'optimizer' in ck and 'model' in ck
        for field, expected in EXPECTED.items():
            assert c[field] == expected == resolved[field], (run, field)
        for field in ('local_loss_mode', 'local_patch_size', 'local_topk_ratio', 'global_loss_weight', 'local_loss_weight'):
            assert c[field] == resolved[field], (run, field)
        for field in ('local_patch_stride','content_selector_alpha','soft_tail_temperature',
                      'multi_scale_patch_sizes','multi_scale_patch_strides','multi_scale_weights',
                      'excess_loss_scale','excess_threshold'):
            assert c.get(field) == resolved.get(field), (run, field)
        assert Path(resolved['init_checkpoint']) == SOURCE and resolved['resume'] is None
        assert resolved['noise_layers'] == ['RandomCrop(0.3, 1.0)']
        # The historical Noise constructor mutates the config list to modules.
        layer = c['noise_layers'][0]
        assert layer.min_area == 0.3 and layer.max_area == 1.0
        for kind in ('train', 'val'):
            logs = [json.loads(x) for x in (folder / f'{kind}.jsonl').read_text().splitlines() if x]
            assert [r['epoch'] for r in logs] == list(range(1, 21)), (run, kind)
        m = artifacts[file]['metrics'][key]
        assert math.isclose(m['worst_psnr'], 10 * math.log10(4 / m['top25_patch_mse']), abs_tol=1e-7)
        dp, dw = m['psnr'] - baseline['psnr'], m['worst_psnr'] - baseline['worst_psnr']
        row = dict(method=run, description=describe(c), seed=c['seed'], source_checkpoint=str(SOURCE),
                   continuation_epochs=20, image_size='128x128', payload_bits=64,
                   patch_size=c['local_patch_size'], stride=c.get('local_patch_stride',c['local_patch_size']),
                   top_ratio=c['local_topk_ratio'] if c['local_loss_mode'] in {'topk','multiscale','contentaware_gradient'} else '',
                   global_weight=c['global_loss_weight'], local_weight=c['local_loss_weight'],
                   loss_mode=c['local_loss_mode'], temperature=c.get('soft_tail_temperature',''),
                   selector_alpha=c.get('content_selector_alpha',''), psnr=m['psnr'], delta_psnr=dp,
                   worst_psnr=m['worst_psnr'], delta_worst_psnr=dw, local_specific_gain=dw-dp)
        row.update({k: v for k, v in m.items() if not k.startswith(('final_', 'val_', 'soft_'))})
        row.update({'delta_' + k: v-baseline[k] for k, v in m.items()
                    if k in baseline and not k.startswith(('final_', 'val_', 'soft_'))})
        row['status'] = 'pixel_tail_best_fixed' if run == REGISTRY[3][2] else 'reference' if run == REGISTRY[0][2] else 'ablation'
        row.update(evaluation_json=str(ROOT / 'reports' / file), evaluation_key=key,
                   evaluation_sha256=digest(ROOT/'reports'/file), checkpoint=str(ck_path), checkpoint_sha256=digest(ck_path))
        rows.append(row)
    return rows


def lineage():
    rows = []
    for folder in sorted(RUNS.iterdir()):
        if not folder.is_dir():
            continue
        config_file = folder / 'config.resolved.json'
        files = sorted(folder.glob('checkpoint_*.pth'))
        if not config_file.exists():
            continue
        resolved = read_json(config_file)
        ck = checkpoint(files[-1]) if files else {}
        c = ck.get('config', resolved)
        source, resumed = resolved.get('init_checkpoint'), resolved.get('resume')
        mode = 'continuation_from_external_checkpoint' if source else 'resume_same_run_origin_unverified' if resumed else 'random_initialized_scratch'
        delta = [field for field in resolved if field not in {'noise_layers','device','resume','init_checkpoint'} and field in c and c[field] != resolved[field]]
        rows.append(dict(run=folder.name, description=describe(c), seed=c['seed'],
                         training_mode=mode, init_checkpoint=source or '', resume_checkpoint=resumed or '',
                         latest_checkpoint=str(files[-1]) if files else '', latest_epoch=ck.get('epoch',''),
                         configured_epochs=c['epochs'], optimizer_in_checkpoint='optimizer' in ck,
                         image_size=f'{c["H"]}x{c["W"]}', payload_bits=c['message_length'],
                         crop_protocol=json.dumps(resolved['noise_layers']), patch_size=c['local_patch_size'],
                         stride=c.get('local_patch_stride',c['local_patch_size']), ratio=c['local_topk_ratio'],
                         global_weight=c['global_loss_weight'], local_weight=c['local_loss_weight'],
                         loss_mode=c['local_loss_mode'], temperature=c.get('soft_tail_temperature',''),
                         alpha=c.get('content_selector_alpha',''), lr=c['lr'], batch=c['batch_size'],
                         deterministic_config=c.get('deterministic',False), single_gpu_required=c.get('require_single_gpu',False),
                         actual_device_count='not persisted in checkpoint',
                         evaluation_role='controlled registered' if folder.name in {x[2] for x in REGISTRY} else 'exploratory only / not a registered main result',
                         resolved_checkpoint_mismatches=json.dumps(delta), config_json=json.dumps(resolved,ensure_ascii=False)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='read-only check of published tables')
    args = parser.parse_args()
    rows, history = master(), lineage()
    if args.check:
        assert (REPORTS/'current_controlled_master_table.csv').read_text() == csv_text(rows)
        assert (REPORTS/'full_experiment_lineage_v3.csv').read_text() == csv_text(history)
        print(f'PASS: {len(rows)} source-backed results; {len(history)} run records; no writes.')
        return
    publish('current_controlled_master_table.csv', csv_text(rows))
    lines = ['# 正式 controlled seed17 结果表（从原始评测JSON重建）', '',
             '主方法固定为Patch16/stride8/Top10原始MSE选块；所有行同源续训20轮。原始CSV已按内容哈希归档到 audit_history。', '',
             'PSNR = 10log10(4 / 数据集平均MSE)。局部WorstPSNR沿用正式定义：32×32非重叠块中每图Top25% MSE均值的PSNR，并非单个最差块。', '',
             '| 完整配置（共用seed17、batch16、LR=1e-4、消息权重10） | Δ整图PSNR | Δ局部PSNR | 局部额外增益 | ΔTop25 MSE | ΔP95 MSE | Δ局部SSIM | Δ局部LPIPS | ΔBER30 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["description"]} | {r["delta_psnr"]:+.6f} | {r["delta_worst_psnr"]:+.6f} | {r["local_specific_gain"]:+.6f} | {r["delta_top25_patch_mse"]:+.9f} | {r["delta_patch_mse_p95"]:+.9f} | {r["delta_local_ssim_top25"]:+.6f} | {r["delta_local_lpips_top25"]:+.9f} | {r["delta_ber30"]:+.7f} |')
    lines += ['', 'CSV含绝对指标、所有BER、实际配置、评测键、JSON与checkpoint SHA-256。哈希是在本次审计时建立的绑定，不代表旧评测文件在生成时已经记录了该绑定。', '',
              '局部SSIM按SSIM自身排序取最低25%；LPIPS按LPIPS自身排序取最高25%，并非在MSE最差块上计算。两者必须分别解释。', '',
              'Excess的局部额外增益为正，但整体与局部PSNR都下降，不能标记为成功。Gradient alpha2是感知/像素尾部权衡项，当前仍保留MSE Top10作为像素尾部主方法。']
    publish('current_controlled_master_table.md', '\n'.join(lines)+'\n')
    publish('full_experiment_lineage_v3.csv', csv_text(history))
    publish('full_experiment_lineage_v3.md', '# 实验谱系 v3\n\n从实际run的config.resolved.json与最新checkpoint元数据重建，每个run独立一行。未从旧CSV截断/猜测错位字段。\n\n'
            f'共{len(history)}个有配置记录的run；旧版/v2保留历史，不用于机器分析。\n\n'
            'scratch：未指定外部init/resume；continuation：明确外部init checkpoint；resume：沿用指定断点，但若初始来源未保存则标记未知。\n\n'
            'GPU实际数量没有持久化时明确标记未知，require_single_gpu仅为配置证据。噪声层以resolved字符串记录，checkpoint里的实际模块属性已在主结果检查中核对。\n\n'
            '[逐run CSV](full_experiment_lineage_v3.csv)含完整resolved配置、全部关键训练字段及元数据不一致记录。没有重新训练或运行历史seed。\n')
    print(f'Validated {len(rows)} controlled results and rebuilt {len(history)} run records.')


if __name__ == '__main__':
    main()
