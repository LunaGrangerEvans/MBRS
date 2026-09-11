# External baseline compatibility audit

Audit date: 2026-09-10. No MBRS training was started. The audit records repository identity, code availability, checkpoint provenance, and blockers before any external result is treated as a comparison.

## StegaStamp

- Repository: https://github.com/tancik/StegaStamp
- Head commit: `c984446048b826587ca1875027b0a1dc1885fb30`
- Officiality: official code release for StegaStamp.
- Framework: legacy TensorFlow 1.x, including `tensorflow.contrib`; `bchlib`, STN and OpenCV dependencies.
- README protocol: default model expects 400×400 images and a UTF-8 secret of at most seven characters. The 100-bit packet contains 56 payload bits plus BCH-5 redundancy.
- Checkpoint status: the current official checkout contains no `saved_models` checkpoint directory. The current README explicitly removed checkpoint details, and no verified official checkpoint URL was available in the checkout. No unverified download was attempted.
- Environment status: the independent Python 3.7/TF1 specification is recorded in `external_baselines/stegastamp_env.yml`; conda creation was attempted outside MBRS but the configured mirror failed on incomplete/timeout package downloads. No partial environment is treated as valid.
- Status: blocked for executable evaluation until both a working TF1 environment and a verified official checkpoint are obtained.
- Comparison: reference only if later executed; payload and ECC are not equivalent to MBRS 64 raw bits.

## TrustMark Q and P

- Repository: https://github.com/adobe/trustmark
- Head commit: `2ecb73ad28d1a3f66ac9dc19e1b667711f14314f`.
- Officiality: official Adobe open-source implementation.
- Framework: PyTorch, OmegaConf, Lightning, torchvision, einops.
- Official checkpoint source: model URLs and MD5 hashes are embedded in `python/trustmark/trustmark.py`; the Q and P encoder/decoder files were downloaded into the isolated repository model directory and their declared MD5 values are recorded in the CSV.
- Q: 100 internal bits; default BCH_5 exposes a protected payload of 61 bits; encoder resolution 256 and decoder resolution 245; official default strength 1.0.
- P: 100 internal bits; default BCH_5; encoder resolution 256 and decoder resolution 224; P forces center-square preprocessing; official default strength 1.0.
- Environment: isolated `external_baselines/envs/trustmark` with `torch 2.7.0+cu126`, `pytorch_lightning 2.6.5`, and `torchmetrics 1.9.0`; the MBRS environment was not modified.
- Official example: `python/test-decode.py` completed for the P model. A fixed 61-bit BCH-5 payload then decoded exactly in the direct Q/P smoke test.
- Formal inference: Q and P each encoded all 50 fixed formal images and saved 50 PNG outputs. Clean exact decode was 50/50 for Q and 48/50 for P. Fixed crop results are recorded in the external main table and under `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/trustmark_{Q,P}/`.
- Status: executable official inference completed. It remains `REFERENCE` only because payload/ECC, preprocessing, resolution and decoder semantics differ from MBRS.
- Comparison: reference only because payload/ECC, resolution and preprocessing are not equivalent to MBRS.

## HiDDeN

The current workspace has no verified HiDDeN official repository checkout or official pretrained checkpoint. The commonly cited repository identity is not sufficient to establish a checkpoint source, architecture version, preprocessing, or ECC. No third-party weight was downloaded and no retraining was started. Status: `REQUIRES RETRAINING` and not evaluated.

## Phase-B decision

TrustMark Q/P produced trustworthy official outputs and fixed-manifest reference metrics. StegaStamp remains blocked because the official checkout has no verified SavedModel checkpoint. HiDDeN is marked `REQUIRES RETRAINING` because no verified pretrained checkpoint is in scope and retraining is prohibited. This is a compatibility/reference audit, not a strict cross-method ranking.

The machine-readable record is [external_baseline_compatibility_audit.csv](external_baseline_compatibility_audit.csv). The isolated repositories are under `external_baselines/repos/`; no external dependency was installed into the MBRS environment.
