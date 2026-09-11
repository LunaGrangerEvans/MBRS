# External baseline reproduction status

This directory contains isolated source checkouts, an isolated TrustMark environment, and output locations. No MBRS model was trained in this phase.

## Checked sources

- `repos/stegastamp`: official checkout at `c984446048b826587ca1875027b0a1dc1885fb30`.
- `repos/trustmark`: official Adobe checkout at `2ecb73ad28d1a3f66ac9dc19e1b667711f14314f`.
- `repos/HiDDeN`: original author repository at `9baa4a79dffe293f7553068f92eb7527534cba26`.
- `repos/HiDDeN-pytorch`: author-linked community PyTorch port at `556f76dd0602e4351ed4c19bd2ee87d50ef3c0de`.

## Current blockers

- StegaStamp official checkout has no verified pretrained SavedModel checkpoint available in the current source.
- The original author repository has no released pretrained model (`Coming soon...`). The linked PyTorch port contains Git-tracked 30-bit experiment checkpoints; one was used only for a compatibility smoke test, not as an official baseline.
- A separate random-initialized 64-bit HiDDeN reimplementation was trained for one seed on the current DIV2K split and evaluated on the fixed manifest. It is explicitly labelled `HiDDeN reimplementation, 64-bit retrained`, not an official pretrained baseline.
- A separate Python 3.7/TF1 environment specification is recorded in `stegastamp_env.yml`. Creating the conda environment was attempted outside the MBRS environment, but the configured mirror failed with incomplete/timeout downloads for `icu`/`openssl`; no partially created environment is treated as valid and no StegaStamp inference was run.

HiDDeN details:

- [`hidden_compatibility_audit.md`](../reports/hidden_compatibility_audit.md)
- [`hidden_smoke_test.md`](../reports/hidden_smoke_test.md)
- [`hidden_64bit_retraining.md`](../reports/hidden_64bit_retraining.md)
- [`hidden_64bit_results.csv`](../reports/hidden_64bit_results.csv)
- `repos/HiDDeN-pytorch/experiments/no-noise adam-eps-1e-4/checkpoints/no-noise--epoch-200.pyt` (30-bit, Git-tracked; SHA-256 recorded in the audit)

## TrustMark Q/P completed reference run

Environment: `envs/trustmark` with `torch 2.7.0+cu126`, `pytorch_lightning 2.6.5`, and `torchmetrics 1.9.0`. The official `python/test-decode.py` example was run from the official working directory. Q and P use official CDN model files whose MD5 values are checked by the repository loader.

Run the fixed 50-image formal evaluation from the project root:

```bash
PYTHONPATH=$PWD/external_baselines/repos/trustmark/python \
  $PWD/external_baselines/envs/trustmark/bin/python \
  external_baselines/run_trustmark_formal.py --mode both
```

Then compute the frozen full-reference image metrics in the normal MBRS environment:

```bash
python experiments/evaluate_external_baseline_metrics.py --device cuda
```

Outputs:

- `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/trustmark_Q/`
- `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/trustmark_P/`
- `/mnt/wmcontent/GLX/icassp/MBRS/reports/external_baseline_metrics/`

TrustMark uses 100 internal bits with BCH-5 and 61 data bits. Its crop result is ECC exact-message success/detection, not MBRS raw 64-bit BER, so it remains `REFERENCE` only.
