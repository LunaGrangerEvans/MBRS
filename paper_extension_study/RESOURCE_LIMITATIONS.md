# Extension Study Limitations and Blocked Phases

## Multi-seed source-checkpoint gate

The pre-registered multi-seed set is `17, 23, 42`. Exact epoch-100 crop-trained Global source checkpoints were found for seed17, seed29, and seed41, but not for seed23 or seed42.

The registered protocol requires seed-specific epoch-100 sources. Reusing seed17's source for seeds23/42, switching to full-scratch training, or substituting seed29/41 would change the experiment design. Therefore the multi-seed reproducibility phase is blocked and no seed23/42 training is launched.

This is a protocol-availability limitation, not a performance conclusion. The frozen seed17 paper evidence remains unchanged.
