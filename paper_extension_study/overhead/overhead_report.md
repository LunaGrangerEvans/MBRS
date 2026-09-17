# Overhead Benchmark Report

## Protocol

The benchmark uses `20` warm-up and `100` measured iterations on a fixed validation batch of shape `(16, 3, 128, 128)`, FP32, with CUDA synchronization. Training timing includes forward, loss computation, backward, and Adam step. Inference timing covers the encoder-plus-decoder path with no crop/noise module.

## Results

| Mode | Method | Mean ms | Median ms | Std ms | Images/s | Peak allocated MiB | Peak reserved MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| training | Global | 105.7795 | 102.7065 | 10.6367 | 151.258 | 6975.47 | 7990.00 |
| training | Ours | 107.2207 | 105.1214 | 7.7709 | 149.225 | 6976.96 | 7990.00 |
| inference | Global | 22.7025 | 21.6057 | 4.8766 | 704.769 | 903.29 | 7990.00 |
| inference | Ours | 21.7693 | 21.6014 | 1.1677 | 734.980 | 903.29 | 7990.00 |

## Relative timing

- Training-step mean relative overhead, Ours vs Global: `+1.36%`.
- Inference mean relative timing difference, Ours vs Global: `-4.11%`; this is a noisy runtime measurement, not an architectural-overhead claim.

## Architecture

- Global parameters: `20805391` total, `20805391` trainable.
- Ours parameters: `20805391` total, `20805391` trainable.
- Architecture identity: `EncoderDecoder(Encoder_MP, Identity, Decoder)` for both paths.
- Additional inference module: **NONE**.

The scientifically important result is that the proposed Tail and OKLab terms are training-only and introduce no additional inference module or parameters. Any measured inference latency difference is reported as runtime noise/benchmark variation and is not called zero without a noise analysis.
