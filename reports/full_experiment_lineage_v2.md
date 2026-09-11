# Corrected experiment lineage v2

> 已被 [v3](full_experiment_lineage_v3.md) 替代。v2虽然每行列数一致，但从旧CSV截断后残留语义错位，不能作为配置来源；v3直接读取每个run的resolved config和checkpoint。

This version corrects two audit issues in the original lineage:

1. Historical candidate and parallel suite rows were direct random-initialization runs. They did not load an external crop-trained Global checkpoint. If interrupted, they resumed their own checkpoint; this is not a controlled continuation.
2. The old label `patch8_nonoverlap_top25_global0.25_local0.25` was inconsistent with its actual weights. The corrected configuration is `patch8_nonoverlap_top25_global0.75_local0.25`.

## Training-mode definitions

- `scratch; same-run resume only if interrupted`: model and optimizer start from random initialization; a resume only continues the same scratch run after interruption.
- `continuation`: model and optimizer are initialized from the specified epoch-100 Global checkpoint, then trained for the controlled continuation schedule.
- `analysis-only`: no parameter update; existing checkpoints or fixed tensors are re-evaluated.

## Evidence boundary

Historical rows remain exploratory because their device topology, deterministic settings, file ordering, or run duplication are not equivalent to the current seed17 protocol. The only formal comparison table is [current_controlled_master_table.csv](current_controlled_master_table.csv).

The corrected machine-readable lineage is [full_experiment_lineage_v2.csv](full_experiment_lineage_v2.csv).

The alpha2 content-aware continuation is included as a controlled branch. It was explicitly requested after the frozen validation analysis, although the earlier validation gate had selected alpha1; it is therefore reported as a user-authorized alpha2 ablation, not as a validation-selected formulation.
