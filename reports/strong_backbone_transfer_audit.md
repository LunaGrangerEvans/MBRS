# Strong-backbone transfer audit: TrustMark-Q

审计日期：2026-09-11。范围是 `/root/workspace/GLX/icassp/MBRS` 当前
`external_baselines`、本地 Adobe TrustMark checkout，以及一次受控的最小
adapter smoke test。没有启动长时间训练，也没有修改现有 MBRS 训练文件。

## 结论先行

当前最适合作为新 backbone 的方法是 **TrustMark-Q**，但必须准确称为
“基于官方预训练权重的自定义 transfer adapter”，不能称为官方 TrustMark
fine-tuning 或官方训练复现。

- **质量/裁剪综合首选：TrustMark-Q。** 最新冻结 external reference 表中，Q
  的 dataset-mean PSNR 为 `42.293755 dB`、full LPIPS 为 `0.000964749`；其
  TrustMark-specific pre-ECC packet BER 在 retained area `70/50/40/30%`
  为 `0.00788/0.05192/0.11640/0.19052`，ECC exact-message success 为
  `97.2/58.4/11.2/0%`。见
  [`paper/external_reference_table.md`](../paper/external_reference_table.md:3-25)。
- **P 是质量上限，不是当前综合首选。** P 的 PSNR/LPIPS 更好
  (`48.107281` / `0.000304701`)，但所有 partial crop 的 exact recovery 都
  低于 Q；因此 Q 更符合“质量 + crop robustness + 可迁移性”的目标。
- **TrustMark-Q 的图计算路径可 fine-tune。** 本地实测 encoder 38/38 个参数张量
  收到有限非零梯度，`8,642,179` 个 encoder 参数可训练；一次显式 backward、
  一步更新、一个单 batch epoch、官方 decoder decode 和 adapter checkpoint
  restore 均通过。
- **TrustMark-Q 的官方训练续训路径不可复现。** 官方当前 main 发布了模型定义、
  推理 loader 和权重下载校验，但没有完整 encoder/decoder `training_step`、
  optimizer/data/noise runner；Q YAML 引用的 `trustmark.loss`、`trustmark.munit`、
  `trustmark.utils.transformations2` 和数据模块也不在 checkout 中。因此官方
  checkpoint 只能作为权重级 warm start，不能恢复原 optimizer/scheduler/epoch
  或原训练轨迹。
- **是否值得继续：有条件地值得。** 若目标是判断既有 Hard local-tail 策略能否迁移
  到更强 backbone，Q 的 adapter smoke 已证明技术链路成立，值得做一个短的、
  encoder-only、冻结 decoder 的 transfer screen；在获得/重建可信的原始 image
  objective 前，不应启动正式大规模训练或宣称“TrustMark 官方 fine-tune”。

## Ranked synthesis

| Rank | 方法/用途 | Quality + crop 证据 | Trainability / weights | 判断 |
|---|---|---|---|---|
| 1 | **TrustMark-Q**：新 backbone、custom adapter | 当前 external 中质量和 crop 的平衡最好；统一表格是 reference，不是严格同 payload 排名 | 官方 encoder/decoder 权重可验证，底层 autograd 可用；官方训练发布不完整 | **首选** |
| 2 | TrustMark-P：质量上限/补充端点 | PSNR/LPIPS 最好，但 crop recovery 明显弱于 Q | 与 Q 共享“有权重、无完整官方训练 pipeline”的缺口 | 只作 quality upper bound |
| 3 | HiDDeN-PyTorch：可启动的第三方 fallback | 仓库有自报 crop/combined-noise 数字，但不是本项目统一协议，且作者明确未完全复现原论文 | 有 Python training CLI 和仓库内实验 checkpoint，但不是作者官方发布 | 可训练 fallback，不是更强 backbone |
| 4 | StegaStamp / 作者 HiDDeN：官方训练候选 | 当前没有进入本项目统一 50-image quality/crop 评测 | 有官方训练代码，但当前没有可验证 pretrained checkpoint；StegaStamp 还依赖 TF1.x | 若硬性要求官方训练，需从头重训，暂不选 |

**严格筛选结果：当前没有一种 external baseline 同时满足“官方完整训练代码 +
官方预训练权重 + 当前统一评测”。** TrustMark-Q 是证据最强、性能最好的外部
reference，并通过本次 adapter 证明“可以自定义微调”；HiDDeN-PyTorch 是
trainability fallback，但没有足够证据证明它是更强 backbone。

## 1. TrustMark 官方 repo 与本地环境检查

### 1.1 Architecture 是否可访问

**Evidence**

- Q 配置指定 `trustmark.unet.Unet1` encoder、`trustmark.unet.SecretDecoder`
  的 `resnet50` decoder，见
  [`trustmark_Q.yaml:1-22`](../external_baselines/repos/trustmark/python/trustmark/models/trustmark_Q.yaml:1)。
- `Unet1` 包含 secret-to-image dense embedding、4 次下采样/跳连/上采样和
  3-channel `tanh` 输出，见
  [`unet.py:239-308`](../external_baselines/repos/trustmark/python/trustmark/unet.py:239)。
- `SecretDecoder(arch='resnet50')` 使用 ResNet-50 并将 fc 替换成 100-bit 输出，
  见 [`unet.py:409-461`](../external_baselines/repos/trustmark/python/trustmark/unet.py:409)。
- 官方 `TrustMark` 对象暴露独立的 encoder/decoder loader，见
  [`trustmark.py:95-126`](../external_baselines/repos/trustmark/python/trustmark/trustmark.py:95)。

本地运行时类型为：

```text
encoder_wrapper = TrustMark_Arch
encoder_core    = Unet1
decoder_wrapper = TrustMark_Arch
decoder_core    = SecretDecoder(resnet50)
```

### 1.2 Q checkpoint 是否可加载

**Evidence**

官方 loader 在 [`trustmark.py:141-199`](../external_baselines/repos/trustmark/python/trustmark/trustmark.py:141)
中检查 MD5、加载 YAML、实例化模块并调用 `load_state_dict`。本地 Q 文件和官方
声明完全一致：

| 文件 | 大小 | 本地 MD5 |
|---|---:|---|
| `trustmark_Q.yaml` | 2,009 bytes | `fe40df84a7feeebfceb7a7678d7e6ec6` |
| `encoder_Q.ckpt` | 17,302,074 bytes | `700328b8754db934b2f6cb5e5185d81f` |
| `decoder_Q.ckpt` | 47,652,460 bytes | `4ced90e9cfe13e3295ad082887fe9187` |

直接加载结果：

```text
encoder: missing=0, unexpected=0
decoder: missing=0, unexpected=0
```

官方仓库也说明模型文件不随 Git 打包，而是在首次使用时按 URL/MD5 下载，见
[`README.md:15-22`](../external_baselines/repos/trustmark/README.md:15)。本地
checkout 为官方 Adobe repo，commit `2ecb73ad28d1a3f66ac9dc19e1b667711f14314f`。

### 1.3 Encoder 是否支持 gradient/backward

**Evidence**

`TrustMark_Arch.forward` 不含 `detach`，只将 encoder 输出包装为 stego/residual，
见 [`model.py:127-133`](../external_baselines/repos/trustmark/python/trustmark/model.py:127)。
官方模型加载后只调用 `.eval()`，没有将 encoder 参数设为不可训练，见
[`trustmark.py:187-199`](../external_baselines/repos/trustmark/python/trustmark/trustmark.py:187)。

本次 adapter smoke 在 CUDA 上实测：

```text
encoder trainable parameters = 8,642,179
encoder gradient parameters  = 38 / 38
one batch backward           = finite, all 38 gradient tensors nonzero
```

**Important boundary：** 高层 `TrustMark.encode(PIL, ...)` 被
`@torch.no_grad()` 修饰，内部还会走 `no_grad`、NumPy 和 PIL，见
[`trustmark.py:464-507`](../external_baselines/repos/trustmark/python/trustmark/trustmark.py:464)。
所以 adapter 必须直接调用张量路径 `tm.encoder(cover, message)` 和
`tm.decoder.decoder(stego)`；不能把 `TrustMark.encode` 当作可反传训练 API。

### 1.4 是否有可信官方 training / fine-tuning pipeline

**Evidence**

当前官方 `TrustMark_Arch` 只有构造、checkpoint 初始化、forward 和 logging，
见 [`model.py:31-170`](../external_baselines/repos/trustmark/python/trustmark/model.py:31)。
文件中没有 encoder/decoder 的 `training_step` 或 `configure_optimizers`。

Q YAML 仍保留训练配方的线索：

- `loss_config`：`recon_type=ffl+yuv`、`recon_weight=1.5`、
  `perceptual_weight=1.0`、`secret_weight=20.0`；
- discriminator：`trustmark.munit.MsDCDisGP`；
- noise：`trustmark.utils.transformations2.TransformNet`；
- 数据模块和 Lightning checkpoint 配置。

见 [`trustmark_Q.yaml:24-85`](../external_baselines/repos/trustmark/python/trustmark/models/trustmark_Q.yaml:24)。
但这些训练模块在当前官方 checkout 中不存在；Q 配置完整实例化会先失败于：

```text
ModuleNotFoundError: No module named 'trustmark.loss'
```

官方 loader 为了推理会把另一个网络、loss、discriminator、noise 替换为
`Identity`，见 [`trustmark.py:163-185`](../external_baselines/repos/trustmark/python/trustmark/trustmark.py:163)。
仓库内出现的 `training_step`/optimizer 代码属于 watermark remover，并冻结
watermark embedder，不是 Q encoder/decoder training，见
[`KBNet/kbnet.py:26-52`](../external_baselines/repos/trustmark/python/trustmark/KBNet/kbnet.py:26)。

**Inference：** 当前公开 main 更像面向部署的 inference release，而不是完整的
研究训练 release。官方 GitHub README 只承诺 encode/decode/remove 的 Python
实现和首次下载模型，不提供 encoder/decoder 训练命令；当前官方 raw
[`python/trustmark/loss.py`](https://raw.githubusercontent.com/adobe/trustmark/main/python/trustmark/loss.py)
也不存在（404）。

### 1.5 checkpoint 是否包含足够信息继续训练

**Evidence**

Q encoder checkpoint 是 40 个 tensor entries，decoder 是 322 个 tensor entries；
两者都没有 `global_step`、`epoch`、`optimizer_states` 或 scheduler state。官方
loader 也只从单独权重文件读取 state dict，见
[`trustmark.py:187-195`](../external_baselines/repos/trustmark/python/trustmark/trustmark.py:187)。

因此：

| 目标 | 结论 |
|---|---|
| 作为自定义 adapter 的初始化权重 | **可以** |
| 精确恢复官方训练 epoch/optimizer/scheduler | **不可以** |
| 恢复官方 noise/discriminator/data objective | **不可以** |

## 2. 已实现的最小 adapter

新增：

- [`external_baselines/trustmark_q_adapter.py`](../external_baselines/trustmark_q_adapter.py)
- [`external_baselines/smoke_test_trustmark_q_adapter.py`](../external_baselines/smoke_test_trustmark_q_adapter.py)

设计限制是有意的：

- **不改 TrustMark-Q architecture。** 保留 `Unet1` encoder、100 internal bits、
  ResNet-50 decoder 和 TrustMark 的 256px residual post-processing。
- **第一阶段只训练 encoder。** decoder 固定并保持 `eval()`，避免单 batch 改变
  ResNet BatchNorm running statistics，也保持原 decode message contract。
- **保留 differentiable tensor path。** 训练时只去掉官方 inference API 的
  `no_grad/PIL` 边界，不改 message pipeline。
- **Hard local-tail RGB loss：** 16×16 patch、stride 8、每 patch RGB MSE、
  每张图最高 10% patch 的平均；256×256 时是 31×31 patch grid、`ceil(961×0.1)=97`
  个 selected patches。
- **可选 global OKLab loss：** 复用现有
  [`experiments/oklab_global.py`](../experiments/oklab_global.py:65) 的可微全局
  OKLab pixel distance，默认权重 `0`，不改变 backbone。

### Loss 插入位置

正确的最终插入点是 TrustMark 原始图像/message objective 的外层，而不是 decoder
内部：

```text
cover + 100-bit message
        │
        ▼
TrustMark-Q Unet1 encoder
        │
        ▼
official residual clamp / channel-mean removal / 256px merge
        │                         │
        │                         └── cover comparison
        ▼
stego ───────────────► TrustMark-Q ResNet-50 decoder logits
  │                              │
  └─ Hard P16/s8/Top10 RGB loss   └─ message loss
        │
        └─ optional global OKLab loss

L = L_original_image/message
    + λ_local L_hard_local_tail_rgb
    + λ_oklab L_global_oklab
```

**Crucial limitation：** 因为官方 `ImageSecretLoss` 源码缺失，本 adapter 的 smoke
objective 使用透明的最小 proxy：

```text
L_base = 1.5 * RGB_MSE(stego, cover)
       + 20.0 * BCEWithLogits(decoder_logits, message)
L      = L_base + 0.5 * HardLocalTailRGB
       + λ_oklab * GlobalOKLab
```

这与 Q YAML 中记录的权重意图一致，但不声称复现其 `ffl+yuv + perceptual + GAN`
实现。正式 transfer 必须在拿到可信原始 loss/noise/data 实现后，把同一个
Hard local-tail 项加到真正的 `L_original_image/message` 上；若拿不到，只能把
实验定义为本项目的 custom objective，并在论文中明确说明。

## 3. Smoke-test evidence

运行命令：

```bash
cd /root/workspace/GLX/icassp/MBRS
PYTHONPATH=$PWD \
  external_baselines/envs/trustmark/bin/python \
  external_baselines/smoke_test_trustmark_q_adapter.py \
  --device cuda:0 \
  --checkpoint /tmp/trustmark_q_local_tail_adapter_smoke.pth
```

### Default adapter run

```text
status                                = pass
encoder_gradient_parameters           = 38 / 38
encoder_trainable_parameters           = 8,642,179
patch_grid                            = P16/stride8/Top10
topk_count_at_256                     = 97
one_batch_backward_loss               = 1.23451030
post_backward_loss                    = 1.21779037
epoch_pre_loss                        = 1.21779037
epoch_post_loss                       = 1.17378211
loss_decreased_after_backward         = true
loss_decreased_during_epoch           = true
decode.present                        = true
decode.schema                         = 1
decode.exact_first_payload            = true
checkpoint_restore_format             = trustmark-q-local-tail-adapter-v1
checkpoint_restore_max_abs_difference = 0.0
checkpoint_restore_exact              = true
checkpoint_bytes                      = 198,948,837
wall_seconds                         = 11.031
long_training_started                 = false
```

这里的“1 epoch”是有意限定为重复同一 batch 的 one-step smoke，不是数据集训练。
它只验证梯度、loss 方向、decode 和 save/restore，不构成性能结论。

### Optional global OKLab path

用 `--global-oklab-weight 0.05` 的同一 smoke 也通过：

```text
encoder_gradient_parameters           = 38 / 38
one_batch_backward_loss               = 1.23472321
epoch_post_loss                       = 1.16810811
decode.exact_first_payload            = true
checkpoint_restore_exact              = true
```

### Verification commands

- `ruff check external_baselines/trustmark_q_adapter.py external_baselines/smoke_test_trustmark_q_adapter.py`：通过。
- isolated TrustMark environment `py_compile`：通过。
- `python experiments/test_patch_mse.py`：通过。
- `python experiments/test_local_loss_strategy.py`：通过。
- `python -m experiments.test_oklab_global`：13 tests passed。
- Hard-tail adapter 与现有 MBRS `image_loss_components(mode="topk")` 在随机输入上的绝对差为 `2.38e-7`，来自 reduction 顺序；patch size/stride/top-k 语义一致。
- 本机主环境没有 `pytest` 模块，因此没有伪造 pytest 通过结果；直接测试入口与 TrustMark CUDA smoke 已完成。

## 4. External fallback assessment

**Evidence**

- **StegaStamp** 有作者官方 encoder/decoder training code，默认 140k steps，含
  geometry/color/JPEG/LPIPS/GAN schedule，见
  [`stegastamp/train.py:43-219`](../external_baselines/repos/stegastamp/train.py:43)。
  但当前 pinned checkout 没有可验证的 `saved_models/stegastamp_pretrained`，且
  依赖 TF1.x/`tensorflow.contrib`；不能作为当前已验证强 backbone。
- **作者 HiDDeN** 有 Torch7 训练入口，但 README 的 pretrained models 仍是
  “Coming soon”，见
  [`HiDDeN/README.md:11-28`](../external_baselines/repos/HiDDeN/README.md:11)。
- **HiDDeN-PyTorch** 有训练/continue CLI 和仓库内实验 checkpoint；但 README 明确
  这是 work in progress、未完全复现原论文，见
  [`HiDDeN-pytorch/README.md:1-13`](../external_baselines/repos/HiDDeN-pytorch/README.md:1)，
  其 crop 结果是自报的非统一协议数字，不能与 TrustMark-Q 的统一 reference 表
  直接排序。
- 旧 `external_baseline_main_table` 中 TrustMark PSNR `42.632394/48.308034` 是
  旧聚合定义。最新冻结表已经按 dataset-mean MSE 修正为 `42.293755/48.107281`；
  详见 [`external_metric_consistency_audit.md:20-42`](external_metric_consistency_audit.md:20)。

**Inference：** 如果“可训练”只表示仓库有 training code，StegaStamp/HiDDeN 都可
列为从头训练候选；如果同时要求可信 pretrained weights，当前没有合格项。若接受
第三方实现，HiDDeN-PyTorch 是更快的 fallback，但它的 backbone/quality/crop 证据
不足以取代 TrustMark-Q。

## 5. 需要修改的文件与最小成本

### 已修改/新增

本次只新增以下三个文件，未覆盖当前工作树已有的 MBRS 修改：

1. `external_baselines/trustmark_q_adapter.py`：Q loader、differentiable encode、
   Hard local-tail、可选 OKLab、encoder-only optimizer、adapter checkpoint。
2. `external_baselines/smoke_test_trustmark_q_adapter.py`：one-batch backward、
   one-epoch one-step、official decode、save/restore smoke。
3. `reports/strong_backbone_transfer_audit.md`：本审计报告。

### 预计最小训练成本

- 首轮只训练约 `8.64M` Q encoder 参数，冻结 ResNet-50 decoder；输入固定 256×256，
  建议 batch 1–2 或 gradient accumulation，避免一次引入 decoder BN/optimizer
  不稳定性。
- 本次 CUDA smoke 使用 batch 2，只做两次 optimizer step；完整模型 load、一次
  backward、decode 和两次 checkpoint construction 均在短时内完成。该时间不能
  外推为正式训练吞吐，但证明单卡可执行。
- 真实成本的主要未知数是官方 `TransformNet`、GAN discriminator、FFL/YUV/perceptual
  loss 和数据 pipeline 缺失；恢复这些组件后显存/吞吐可能明显高于当前 encoder-only
  proxy。
- 官方 YAML 的 `batch_size=32`、`max_epochs=150` 只能视为历史 recipe 线索，不能
  当作本项目正式训练预算，因为对应 runner 和 objective 未发布。

## 6. 是否继续做正式 transfer experiment

### 建议：继续，但分两道闸门

**Gate A — objective fidelity（当前尚未满足）**

1. 先取得可信的原始 `ImageSecretLoss`、noise/augmentation、data loader 和
   discriminator training 实现，或明确批准使用本报告的 custom RGB-MSE+BCE proxy。
2. 固定 Q encoder checkpoint、decoder 是否冻结、message 100 internal bits、256px
   preprocess、crop protocol 和 checkpoint format。
3. 不改 Q `Unet1` / ResNet-50 architecture，不把 local loss 搬到 decoder 内部。

**Gate B — bounded transfer screen（值得做，但本轮未启动）**

1. Q control：base image/message objective only。
2. Q + Hard P16/stride8/Top10：沿用现有策略。
3. 可选 Q + Hard + small global OKLab：只用一个预注册轻量权重，不开网格。
4. 优先 encoder-only、冻结 decoder；先短 epoch/小 split，观察 global quality、
   local-tail、raw message accuracy 和 fixed crop decode。
5. 只有 control 与 Hard 的方向稳定，且 decode/BER 没有明显退化，才考虑解冻 decoder
   或恢复更完整 objective。

**最终判断：** 这项 transfer 研究值得继续，因为 TrustMark-Q 已表现出明显更高的
reference quality/crop operating point，且梯度链路已通过实测；但当前不能把结果
包装成官方 TrustMark fine-tuning。暂时保持在本次 smoke 规模，不启动长时间训练。

## Unknowns / limits

- 官方内部训练代码、原始 `ImageSecretLoss`、完整 noise schedule、discriminator
  配置实现和 Adobe Stock training data 不在当前公开 main 中。
- 本地 TrustMark clone 是 shallow checkout；结论针对当前官方 main commit，不证明
  历史上从未存在未发布的 training source。
- TrustMark-Q/P 的 crop 是 BCH/ECC exact-message/detection 语义，MBRS 是 raw 64-bit
  BER；两者继续只能标 `REFERENCE`，不能做严格 BER superiority claim。
- 本次 smoke 使用官方 Q weights + 自定义透明 proxy，不代表 crop-trained formal
  transfer 的质量或鲁棒性结果。
- 当前没有启动 StegaStamp、官方 HiDDeN 或新的大规模 Q 训练。
