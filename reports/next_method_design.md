# Next-step method and experiment design

## Decision summary

Do not launch another broad grid. The only configuration passing the predefined paired-seed engineering screen is hard top-k with patch16/top25/global-local 0.5/0.5. Patch32 Weight25 does not replicate, so the condition for treating `patch16 + weight25` as a validated factorial combination is not met.

The next experiment should instead use patch16/weight25 as a controlled diagnostic continuation from each seed's Global checkpoint. Its purpose is to test whether a small, fine-grained local signal can improve the tail without re-solving message robustness from scratch.

An external full-from-scratch `patch16 + weight25` three-seed suite started while this audit was running. It is useful exploratory evidence, but it uses the historical multi-GPU/non-deterministic protocol and is not a substitute for the controlled continuation below.

## Evidence-based candidate ranking

| Candidate | Mean ΔPSNR | Mean ΔWorstPSNR | Mean ΔBER30 | Positive ΔWorst seeds | Decision |
|---|---:|---:|---:|---:|---|
| patch16/top25/weight50 | +0.701 | +0.890 | +0.00092 | 2/3 | passes internal screen; best current signal |
| patch16/top25/weight75 | -0.055 | +0.101 | -0.00125 | 2/3 | local coefficient too aggressive for the small effect |
| patch32/top25/weight25 | -0.750 | -0.586 | -0.00119 | 2/3 | fails due to seed29 collapse |
| patch32/top50/weight50 | -0.297 | -0.264 | -0.00085 | 1/3 | no local-quality gain |
| default patch32/top25/weight50 | -2.054 | -1.961 | +0.00346 | 0/3 | reject |

No candidate currently establishes statistical significance with three seeds. Patch16/weight50 has paired `p=0.262` for ΔWorstPSNR; it is a screening winner, not yet a paper claim.

## Priority 0: reproducibility gate

Before interpreting another method result:

1. Use exactly one visible GPU for every control and local run.
2. Keep batch size 16 on that GPU; do not mix with `DataParallel`, because BatchNorm changes when the batch is split per replica.
3. Use sorted dataset paths, explicit DataLoader generators, and deterministic Torch/cuDNN settings.
4. Record the exact source checkpoint and continuation config.
5. Keep Global control and local variant paired by source checkpoint and seed.

The implementation now supports these controls. Existing historical results should remain in the paper only as exploratory evidence; new confirmatory comparisons should follow this gate.

## Priority 1: Global continuation versus hard-local fine-tuning

Use each seed's epoch-100 crop-trained Global checkpoint as the common starting point.

| Arm | Continued epochs | LR | Global weight | Local weight | Patch / top ratio |
|---|---:|---:|---:|---:|---|
| Control | 20 | 1e-4 | 1.0 | 0.0 | none |
| Local | 20 | 1e-4 | 0.75 | 0.25 | patch16 / top25% |

Both arms load the same model and Adam state, reset the continuation schedule to epoch 1, and force the configured LR after loading optimizer state. This directly controls for “the model only improved because it trained longer.”

Prepared but not executed:

- `experiments/config_finetune_global_control_128_m64.json`
- `experiments/config_finetune_patch16_weight25_128_m64.json`
- `experiments/run_controlled_finetune_screen.sh`

Run the full six-arm screen only after a short seed17 preflight confirms deterministic repeatability.

## Priority 2: local-weight warm-up

If continuation helps but training from scratch is still required, use:

```text
epochs 1-40:   lambda_local = 0.00, lambda_global = 1.00
epochs 41-60:  lambda_local linearly 0.00 -> 0.25
               lambda_global linearly 1.00 -> 0.75
epochs 61-100: lambda_local = 0.25, lambda_global = 0.75
```

Keeping the image-loss coefficient sum equal to one avoids reducing regularization strength during warm-up. The schedule is fully config-driven in `config_warmup_patch16_weight25_128_m64.json`; no schedule constants are embedded in the training loop.

This is lower priority than continuation because it still spends 100 epochs learning robustness and image quality simultaneously.

## Priority 3: soft-tail loss if hard selection remains unstable

For per-image patch errors `e_i`, define

```text
scale = stop_gradient(mean_i(e_i))
z_i = e_i / scale
w_i = softmax(z_i / T)
L_soft-tail = sum_i(w_i * e_i)
```

Default settings:

- patch size: 16
- local weight: 0.25
- temperature `T=0.5`
- detach softmax weights: true

On current seed17 Crop-Global residuals, `T=0.5` gives an effective 52 of 64 patches, while the maximum patch receives about 3.29× the uniform weight. `T=0.25` is much sharper (effective 35 patches, maximum ≈7.86× mean) and risks recreating hard-tail instability; `T=1.0` is nearly uniform (effective 61 patches). Therefore 0.5 is the single justified screening value.

With detached weights, gradients are a smooth positive reweighting of patch MSE and all patches receive local gradient. Without detaching, gradients also flow through softmax; increasing a patch error changes its own weight and all competing weights, giving a more aggressive second pathway that can amplify instability. Start with detached weights.

Prepared but not executed: `experiments/config_finetune_soft_tail_patch16_weight25_128_m64.json`.

Screen soft-tail only if the controlled hard-local continuation fails repeatability. First run seed17 control/hard/soft; expand to seeds29/41 only if it clears the internal thresholds.

## Selection criteria

Continue using the internal screen:

- mean ΔWorstPSNR > +0.3 dB
- mean ΔGlobalPSNR > -0.2 dB
- mean ΔBER30 < +0.005
- positive ΔWorstPSNR in at least 2/3 seeds

Add two independent evaluation checks before promotion:

- top25 local SSIM must improve or remain within -0.002
- top25 local LPIPS must decrease or remain within +5%

These are engineering gates, not significance claims. For a paper claim, report paired seed deltas and confidence intervals and add more seeds if the effect remains near 0.3 dB.

## Motivation and paper claim

The data reject the strong concentration claim:

- No-crop Global mean top25/global MSE ratio: 1.374
- Crop-trained Global: 1.181
- Difference: -0.192, paired image-level `p=2.26e-15`

Crop training increases absolute global and local distortion, but the local tail does not become more concentrated relative to global distortion. The supported claim is:

> Crop-robust watermarking degrades visual quality, while global average objectives do not explicitly control the tail of local distortions.

For seed17, Patch16/weight50 improves the Crop-Global top25 MSE from 0.001670 to 0.001410, top25 local SSIM from 0.8949 to 0.9076, and top25 local LPIPS from 0.00262 to 0.00196. This is encouraging independent evidence, but cross-seed perceptual confirmation is still needed.

## Stop conditions

Do not escalate model complexity if controlled Patch16 continuation meets the gates. Stop the hard-top-k direction if deterministic continuation still shows inconsistent ΔWorst signs or BER30 degradation above +0.005. Only then screen the single soft-tail setting. GAN, frequency loss, and perceptual training loss remain out of scope until this minimal spatial-reweighting question is resolved.
