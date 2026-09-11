# Candidate experiments after system review

No experiment in this list is being started automatically.

## Candidate 1: Top-ratio refinement under overlap

- Scientific question: does the best overlap localization become more tail-specific when selected coverage is reduced?
- Configuration: seed17 controlled source; Patch16 stride8; one selected-ratio value, preferably Top15% because Top25 covers 38.9% unique pixels; global/local 0.5/0.5; 20 epochs.
- Controlled: all source, optimizer, RNG, crop, duration, and manifest settings unchanged.
- Expected: higher local-specific gain if current Top25 is too broad.
- Success: ΔWorstPSNR ≥ +0.30 dB, ΔPSNR ≥ -0.10 dB, BER30 change ≤ +0.002, and P95 improves.
- Failure interpretation: selected coverage is not the bottleneck; move to metric/task definition.
- Priority: highest next training candidate, but only after approval of the single ratio.

## Candidate 2: Content-normalized tail diagnostic, then training only if justified

- Scientific question: are MSE-worst regions simply high-texture/edge regions rather than watermark artifacts?
- Analysis first: correlate patch error with local image activity, gradient magnitude, and edge density.
- Candidate training form only if strong content bias is confirmed: patch MSE divided by detached local content activity, with the same Hard/overlap protocol.
- Success: MSE/LPIPS alignment improves and tail reduction remains positive without BER loss.
- Failure interpretation: content normalization adds complexity without making the target more perceptual.
- Priority: after Candidate 1 or instead of it if the paper pivots to perceptual tail.

## Candidate 3: Larger-region perceptual/structural tail definition

- Scientific question: is 32×32 the right artifact unit, given that 16×16 LPIPS is invalid and MSE/LPIPS rank alignment is weak?
- Analysis first: use 32×32 or larger regions with SSIM/LPIPS and test whether method rankings change.
- Training only if rankings are stable and the perceptual tail is clearly distinct from MSE.
- Success: a lightweight structural tail term improves local SSIM/LPIPS while preserving BER and global PSNR.
- Failure interpretation: the current MSE-defined problem is not perceptually meaningful enough for a stronger claim.
- Priority: highest paper-value diagnostic; not a blind LPIPS-loss addition.

## Not recommended now

- More soft temperatures.
- Multi-scale weight sweeps.
- Excess-loss threshold/scale sweeps.
- Patch8/Patch64 grids.
- GAN, frequency loss, backbone changes, or full-scratch matrices.
