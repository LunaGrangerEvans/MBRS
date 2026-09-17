# RNG and Alignment Audit

This run is part of **stochastic continuation-seed sensitivity from a common frozen source checkpoint**, not full multi-seed reproducibility.

- Python, NumPy, Torch CPU, and Torch CUDA seeds are set to the run seed.
- Deterministic cuDNN/algorithm settings are enabled; one visible GPU is used.
- DataLoader uses `num_workers=0`, sorted deterministic filenames, and generators derived as `seed + 200003` for train and `seed + 300007` for validation.
- Message generation uses Torch RNG in the fixed trainer call order.
- `RandomCrop(0.3, 1.0)` uses NumPy RNG in the fixed trainer call order.
- Local-tail selection and global OKLab computation consume no RNG.
- Global, Hard, and Ours use the same source, data pipeline, batch size, and call order within a continuation seed; only the registered objective and continuation seed vary.

If any call-order or source-restoration mismatch is discovered, stop that run and document it rather than continuing.

Registered continuation seed: `42`.
