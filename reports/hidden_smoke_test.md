# HiDDeN smoke test

Date: 2026-09-11  
Status: **PASS — compatibility smoke only; not a 64-bit baseline**

## Command

```bash
PYTHONPATH="$PWD/external_baselines" \
  external_baselines/envs/hidden/bin/python \
  external_baselines/run_hidden_smoke.py --device cuda
```

The script uses the source-preserving adapter
[`external_baselines/hidden_adapter.py`](../external_baselines/hidden_adapter.py)
and the unmodified model files in the vendored PyTorch port
[`external_baselines/repos/HiDDeN-pytorch`](../external_baselines/repos/HiDDeN-pytorch).

## Inputs

- Source: MBRS fixed manifest at
  `/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_manifest.pt`
- Sample: manifest index 0
- Image tensor: `1 × 3 × 128 × 128`, normalized to `[-1, 1]`
- Message used: the first 30 bits of the manifest's 64-bit message
- Model checkpoint: `no-noise--epoch-200.pyt`, epoch 200
- Checkpoint SHA-256:
  `71118462194d647f6f5c08dd35f95d4f53db11a9bd2758f3bd336b46aafa5916`
- Device: NVIDIA A800, CUDA runtime 12.1 as reported by PyTorch

The message was truncated only because the checkpoint's decoder is structurally
30-bit. The test does not claim that this checkpoint handles the remaining 34
manifest bits.

## Observed output

| Check | Observed value |
|---|---:|
| Embed | PASS |
| Watermarked image generated | PASS |
| Decode | PASS |
| PSNR | 39.8090019 dB |
| Bit accuracy | 1.0000000 over 30 bits |
| Watermarked image normalized range before display clipping | [-1.0194685, 1.0333217] |
| Decoder output range before port rounding/clipping | [-0.0992404, 1.2041988] |

The image artifact is:

- [`watermarked.png`](/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_smoke/watermarked.png)

The complete machine-readable result is:

- [`result.json`](/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_smoke/result.json)

The reported PSNR is computed against the cover after converting both tensors
to RGB `[0,1]` and clipping the encoded output for display/metric evaluation.
Bit accuracy follows the port's documented `round().clip(0,1)` convention.

## Interpretation

This proves that the known PyTorch port and its repository-tracked 30-bit
checkpoint can perform the requested single-image embed → watermarked image →
decode path in the isolated CUDA environment. It is not an official HiDDeN
checkpoint result and is not a valid comparison to MBRS's 64-bit fixed-manifest
baseline.

The attempted 50-image unified evaluation was intentionally stopped by the
message-length guard before any PSNR, SSIM, LPIPS, Top25 local PSNR, or crop BER
was calculated.
