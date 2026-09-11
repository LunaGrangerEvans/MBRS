"""Read-only validation of the frozen draft's data, provenance and structure."""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
PAPER = PROJECT/'paper'
ROOT = Path('/mnt/wmcontent/GLX/icassp/MBRS')


def read(path):
    with path.open(newline='') as stream:
        values=list(csv.DictReader(stream))
    assert all(None not in row and None not in row.values() for row in values), path
    return values


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    main_table=read(PAPER/'main_table.csv')
    individual=read(ROOT/'reports/paper_freeze_v1/per_image.csv')
    assert len(main_table)==8 and len(individual)==400
    paired=json.loads((PAPER/'paired_diagnostics.json').read_text())
    for row in main_table:
        group=[r for r in individual if r['method']==row['method']]
        assert [int(r['image_index']) for r in group]==list(range(50))
        for key,value in row.items():
            if key in ('method','run'):
                continue
            expected=-10*np.log10(np.mean([float(r['global_mse']) for r in group])) if key=='global_psnr' else np.mean([float(r[key]) for r in group])
            assert np.isfinite(float(value)) and abs(float(value)-expected)<1e-10,(row['method'],key)
        for r in group:
            assert abs(float(r['top10_energy_share'])-.125*float(r['top10_over_mean']))<1e-10
    a=[r for r in individual if r['method']==main_table[0]['method']]
    b=[r for r in individual if r['method']==main_table[3]['method']]
    for key,entry in paired.items():
        delta=np.array([float(rb[key])-float(ra[key]) for ra,rb in zip(a,b)])
        assert abs(float(entry['percent_lower'])-100*np.mean(delta<0))<1e-9
    provenance=json.loads((PAPER/'evidence_manifest.json').read_text())
    for group in ('source_hashes','artifacts'):
        for path,digest in provenance[group].items():
            assert sha(path)==digest,path
    for row in provenance['checkpoints']:
        assert sha(row['checkpoint'])==row['checkpoint_sha256']
        config=json.loads((Path(row['checkpoint']).parent/'config.resolved.json').read_text())
        assert config['seed']==17 and config['epochs']==20
        assert config['init_checkpoint']==row['source_checkpoint']
    figures=ROOT/'visualizations/paper_freeze_v1'
    for path,digest in json.loads((figures/'provenance.json').read_text()).items():
        assert sha(path)==digest,path
    external=read(PAPER/'external_reference_table.csv')
    ext_records=read(ROOT/'reports/external_baseline_metrics/per_image.csv')
    for row in external:
        items=[r for r in ext_records if r['method']==row['method']]
        psnr=-10*np.log10(np.mean([float(r['global_mse']) for r in items]))
        assert abs(float(row['global_psnr'])-psnr)<1e-10
        for ratio in (100,70,50,40,30):
            assert abs(float(row[f'raw_ber_{ratio}'])+float(row[f'raw_bit_accuracy_{ratio}'])-1)<1e-10
    text=(PAPER/'draft_v1.md').read_text()
    assert len(re.findall(r'^## \d+\.',text,re.M))==13
    assert '[CITATION NEEDED]' in text
    assert not re.search(r'36\.227406|36\.407159|0\.247056|42\.632394|48\.308034',text)
    for path in PAPER.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if link.startswith(('http','#')):
                continue
            target=Path(link) if link.startswith('/') else path.parent/link
            assert target.exists(),(path,link)
    print('PASS: 8 endpoints, 400 per-image rows, 40 replayed crop BER values, hashes, external aggregation, links, 13 draft sections.')


if __name__=='__main__':
    main()
