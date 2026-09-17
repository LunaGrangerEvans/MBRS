# Method Interpretation: Empirical Upper-Tail Risk

For an image with patch errors `e_1, ..., e_N`, the Hard Local-Tail term can be written as:

```text
L_tail = (1/K) sum of the K largest patch errors
```

where the frozen method uses the highest-error hard Top-10% of the P16/S8 overlapping patch grid (`K = 23` of `N = 225`). Global RGB MSE averages distortion over all pixels, so it represents an average-risk or average-distortion objective. Hard Local-Tail instead emphasizes the empirical upper tail of local distortion by supervising the largest observed patch errors.

Accordingly, Hard Local-Tail **can be viewed as** an empirical upper-tail risk objective or **CVaR-like** upper-tail supervision. The terminology is intentionally cautious. The implemented term is the finite-sample mean of a hard top-k set on overlapping patches; this note does not establish exact equivalence to formal CVaR, which depends on a specified loss distribution, tail probability, and risk formulation.

The conceptual decomposition is:

```text
Global RGB MSE       → average distortion control
Hard Local-Tail      → upper-tail local distortion control
Global OKLab         → color-aware refinement
```

Both Hard Local-Tail and global OKLab are training-only losses. The inference architecture remains unchanged and no additional inference module is introduced. This note is interpretation guidance only; it does not change the frozen optimization method or authorize new claims.
