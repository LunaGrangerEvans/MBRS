"""Render unchanged RGB outputs for human inspection, without training."""

import base64
import hashlib
import io
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image
import torch

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.analyze_patch_distortion import load_encoded

ROOT = Path('/mnt/wmcontent/GLX/icassp/MBRS')
OUT = ROOT / 'visualizations/artifact_observation'
EXAMPLES = [7, 20, 42]
RUNS = {
    'Global continuation': 'controlled_seed17_global_continuation',
    'Hard Top10': 'controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50',
    'Hard Top25': 'controlled_seed17_hard_patch16_stride8_top25_weight50',
    'Gradient alpha2': 'seed17_contentaware_gradient_patch16_stride8_top10_alpha2_global0.5_local0.5',
}
TITLES = {
    'Original': 'Original',
    'Global continuation': 'Global continuation\nImage MSE weight 1.0',
    'Hard Top10': 'Hard patch16 / stride8 / Top10%\nImage / local weights 0.5 / 0.5',
    'Hard Top25': 'Hard patch16 / stride8 / Top25%\nImage / local weights 0.5 / 0.5',
    'Gradient alpha2': 'Gradient selector alpha2 / Top10%\nPatch16 / stride8; image / local 0.5 / 0.5',
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rgb8(tensor):
    return np.rint(((tensor.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1) * 255).astype(np.uint8)


def locations(original):
    # Only original luminance determines location; method differences never enter.
    luminance = original.astype(float) @ np.array([0.299, 0.587, 0.114]) / 255
    coords = [(x, y) for y in range(0, 128, 32) for x in range(0, 128, 32)]
    activity = [luminance[y:y+32, x:x+32].var() for x, y in coords]
    return {'Low texture': coords[int(np.argmin(activity))],
            'High texture': coords[int(np.argmax(activity))]}


def panel(data, index, crops, methods):
    fig, axes = plt.subplots(3, len(methods), figsize=(4 * len(methods), 10), squeeze=False)
    for col, name in enumerate(methods):
        a = data[name][index]
        axes[0, col].imshow(a, interpolation='nearest')
        axes[0, col].set_title(TITLES[name], fontsize=11)
        for row, ((label, (x, y)), color) in enumerate(zip(crops.items(), ['#ed9d00', '#187eab']), 1):
            axes[0, col].add_patch(Rectangle((x-0.5, y-0.5), 32, 32, fill=False, edgecolor=color, linewidth=1.5))
            axes[row, col].imshow(a[y:y+32, x:x+32], interpolation='nearest')
            axes[row, col].set_title(f'{label} | x={x}, y={y} | 32 x 32 crop', fontsize=10, color=color)
        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(f'Fixed test index {index:02d} | actual RGB images and same-location crops', fontsize=16)
    fig.text(.5, .015, '128 x 128 model input. Nearest-neighbor enlargement; unchanged RGB levels.\n'
             'No residual amplification or per-image contrast adjustment. Crops selected from original-image variance only.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .94))
    return fig


def image_url(a):
    stream = io.BytesIO()
    Image.fromarray(a).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


def browser(data, crop_coords):
    payload = {key: [image_url(a) for a in batch] for key, batch in data.items()}
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>MBRS 水印伪影直接观察</title><style>
body{font:16px system-ui,sans-serif;background:#ededed;color:#222;margin:24px}
header{max-width:1200px}button,select{font:inherit;padding:8px;margin:4px}button:focus-visible{outline:3px solid #1977af}
.cards{display:flex;gap:24px;flex-wrap:wrap}.card{background:white;padding:16px;border:1px solid #bbb}
canvas{display:block;image-rendering:pixelated;margin:12px 0;background:#808080}
.crops{display:flex;gap:16px}figcaption{font-size:13px}figure{margin:0}
.tip{max-width:1100px;color:#444}footer{margin-top:28px;font-size:14px}
</style><header><h1>MBRS：直接观察水印图像</h1>
<p>主比较：仅整图 MSE 续训 vs Patch16 / stride8 / 误差最高10%局部块监督。两者 seed17，续训20轮。
后者整图/局部损失权重均0.5。以原图作共同参考。</p>
<p class="tip">全部50张固定测试视图可浏览。默认样本7；汇报样本固定为7、20、42。
图像为实际128×128输入，导出为8-bit RGB PNG；放大不增加图像细节。未增强对比度、锐化或放大残差。
局部位置由原图16个非重叠32×32块中的最低/最高亮度方差确定。
切换用于定位差异，不代表主观质量评分。</p>
<label>测试图 <select id="sample"></select></label>
<label>显示倍率 <select id="zoom"><option>1</option><option>2</option><option selected>4</option></select></label>
<button id="prev">上一张</button><button id="next">下一张</button>
<button id="reference" aria-pressed="false">切换：全部显示原图</button>
<label><input type="checkbox" id="aux">显示两种辅助方法</label>
<p id="state" aria-live="polite"></p></header><main class="cards" id="cards"></main>
<footer>低纹理/高纹理是原图内容标签，不等于伪影强弱。相同位置跨方法比较；无方法专属取样。
空间裁剪只用于观察，不是新的攻击实验。</footer><script>
const data=PAYLOAD, crops=CROPS, titles=TITLES;
const sample=document.getElementById('sample'), zoom=document.getElementById('zoom');
const main=['Original','Global continuation','Hard Top10'];let showReference=false;
for(let i=0;i<50;i++){let o=new Option(String(i).padStart(2,'0'),i);sample.add(o)}sample.value=7;
function draw(){const i=Number(sample.value), z=Number(zoom.value), host=document.getElementById('cards');host.replaceChildren();
const methods=document.getElementById('aux').checked?Object.keys(data):main;
document.getElementById('state').textContent=`测试视图 ${i}；${showReference?'当前全部显示原图':'当前显示各方法实际输出'}；放大 ${z}×`;
for(const method of methods){const card=document.createElement('article');card.className='card';const h=document.createElement('h2');h.style.fontSize='17px';h.textContent=titles[method].replaceAll('\\n',' / ');card.append(h);
const full=document.createElement('canvas');full.width=128;full.height=128;full.style.width=128*z+'px';full.style.height=128*z+'px';card.append(full);
const strip=document.createElement('div');strip.className='crops';card.append(strip);
const img=new Image();img.onload=()=>{full.getContext('2d').drawImage(img,0,0);
for(const [label,xy] of Object.entries(crops[i])){const figure=document.createElement('figure'),crop=document.createElement('canvas');crop.width=32;crop.height=32;crop.style.width='128px';crop.style.height='128px';crop.getContext('2d').drawImage(img,xy[0],xy[1],32,32,0,0,32,32);figure.append(crop);const t=document.createElement('figcaption');t.textContent=`${label==='Low texture'?'低纹理':'高纹理'} x=${xy[0]} y=${xy[1]}`;figure.append(t);strip.append(figure)}};
img.src=data[showReference?'Original':method][i];host.append(card)}}
sample.onchange=zoom.onchange=document.getElementById('aux').onchange=draw;
document.getElementById('prev').onclick=()=>{sample.value=(Number(sample.value)+49)%50;draw()};
document.getElementById('next').onclick=()=>{sample.value=(Number(sample.value)+1)%50;draw()};
document.getElementById('reference').onclick=(e)=>{showReference=!showReference;e.target.setAttribute('aria-pressed',showReference);e.target.textContent=showReference?'切换：恢复方法输出':'切换：全部显示原图';draw()};draw();
</script></html>'''
    return document.replace('PAYLOAD', json.dumps(payload)).replace('CROPS', json.dumps(crop_coords)).replace('TITLES', json.dumps(TITLES))


def main():
    torch.set_num_threads(4)
    torch.manual_seed(17)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'reports/uniform_eval_manifest.pt'
    manifest = torch.load(manifest_path, map_location='cpu', weights_only=False)
    images, messages = manifest['images'].float(), manifest['messages'].float()
    assert images.shape == (50, 3, 128, 128) and messages.shape == (50, 64)
    data = {'Original': np.stack([rgb8(a) for a in images])}
    provenance = {'manifest': str(manifest_path), 'manifest_sha256': digest(manifest_path),
                  'examples': EXAMPLES, 'epoch': 20, 'samples': 50, 'checkpoints': {},
                  'display': 'clip((tensor+1)/2,0,1), round to uint8; nearest-neighbor enlargement',
                  'roi_rule': 'minimum/maximum original luminance variance among nonoverlap 32x32 blocks; row-major tie break',
                  'before_display_out_of_range_fraction': {}}
    for name, run in RUNS.items():
        checkpoint = ROOT / 'experiments/runs' / run / 'checkpoint_0020.pth'
        print('Rendering', name, flush=True)
        encoded = load_encoded(checkpoint, images, messages, 'cuda:0' if torch.cuda.is_available() else 'cpu', 16)
        provenance['checkpoints'][name] = {'path': str(checkpoint), 'sha256': digest(checkpoint)}
        provenance['before_display_out_of_range_fraction'][name] = float(((encoded < -1) | (encoded > 1)).float().mean())
        data[name] = np.stack([rgb8(a) for a in encoded])
    coords = [locations(a) for a in data['Original']]
    provenance['crop_coordinates_by_test_index'] = coords
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    main_methods = ['Original', 'Global continuation', 'Hard Top10']
    with PdfPages(OUT / 'artifact_observation_main.pdf') as pdf:
        for index in EXAMPLES:
            fig = panel(data, index, coords[index], main_methods)
            fig.savefig(OUT / f'actual_rgb_test_{index:02d}.png', dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
    fig, axes = plt.subplots(3, 5, figsize=(17, 11))
    for row, index in enumerate(EXAMPLES):
        for col, name in enumerate(data):
            x, y = coords[index]['Low texture']
            axes[row, col].imshow(data[name][index, y:y+32, x:x+32], interpolation='nearest')
            if row == 0:
                axes[row, col].set_title(TITLES[name], fontsize=9)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f'Test {index:02d}\nx={x}, y={y}\nlow-texture crop', fontsize=11)
    fig.suptitle('Supplement: same original-defined low-texture crops | actual RGB, no enhancement')
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(OUT / 'artifact_observation_supplement.png', dpi=160)
    plt.close(fig)
    raw = OUT / 'native_rgb'
    raw.mkdir(exist_ok=True)
    for name, batch in data.items():
        slug = name.lower().replace(' ', '_')
        for i, a in enumerate(batch):
            Image.fromarray(a).save(raw / f'{i:02d}_{slug}.png')
    (OUT / 'artifact_observation.html').write_text(browser(data, coords), encoding='utf-8')
    # Check exported pixel values, not just file existence.
    for i in EXAMPLES:
        for name in data:
            path = raw / f'{i:02d}_{name.lower().replace(" ", "_")}.png'
            assert np.array_equal(np.asarray(Image.open(path)), data[name][i])
    print('Verified native PNG pixels; outputs:', OUT, flush=True)


if __name__ == '__main__':
    main()
