# Training and Inference Overhead Benchmark Protocol

This benchmark measures practical cost for the frozen Global and Ours paths without retraining full models and without modifying any paper evidence.

## Hardware/software

- one visible NVIDIA A800-SXM4-80GB GPU;
- CUDA/cuDNN/PyTorch versions recorded in `environment.json`;
- FP32 precision, no autocast or mixed precision;
- batch size `16`, input shape `16×3×128×128`, payload length `64`;
- deterministic settings enabled where applicable; fixed validation tensors are reused for timing.

## Training-step measurement

For each path, load the same seed17 epoch-100 source model and Adam state in memory, restore the model, and run the actual training-step shape:

```text
Encoder → RandomCrop → Decoder → image/message losses → backward → Adam step
```

Global uses the frozen Global objective `10 L_msg + 1.0 L_RGB`. Ours uses the frozen objective `10 L_msg + 0.5 L_RGB + 0.5 L_tail + 0.059149764 L_OKLab`, with P16/S8 hard Top-10% local-tail computation. Each path receives 20 warm-up iterations followed by 100 measured iterations. CUDA is synchronized before and after each timed interval. Timing includes forward, loss computation, backward, and optimizer step.

## Inference measurement

For each frozen Global and Ours checkpoint, measure the same encoder-plus-decoder path with the crop/noise module omitted from inference:

```text
Encoder(cover, message) → Decoder(encoded image)
```

Use the same 20 warm-up and 100 measured iterations, fixed validation batch, FP32, and CUDA synchronization. Record parameter counts, state-dict architecture identity, latency, and peak allocated/reserved memory. No inference-time module is added.

## Reporting

Report mean, median, standard deviation, throughput, peak allocated/reserved GPU memory, and Ours relative overhead:

```text
(Ours − Global) / Global × 100%
```

Latency differences are treated as benchmark measurements with noise. Equal architecture/parameter counts and the absence of an inference loss branch are the primary architectural conclusion; a small measured latency delta is not called zero overhead unless the benchmark supports that statement.
