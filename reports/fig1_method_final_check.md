# Fig. 1 Method Figure Check

## Outputs

- PDF: `/root/workspace/GLX/icassp/MBRS/paper/figures/fig1_method_final.pdf`; page size `1173.1104 x 675.1584 pt`; file size `52111 bytes`.
- PNG: `/root/workspace/GLX/icassp/MBRS/paper/figures/fig1_method_final.png`; raster size `5865 x 3375 px`; embedded DPI `360.0 x 360.0`; file size `735749 bytes`.
- Renderer: `/root/workspace/GLX/icassp/MBRS/paper/render_fig1_method.py`.
- Format: three independent pastel workflow panels, flat fills, no gradients or shadows; PDF uses embedded TrueType vector text.

## Requested workflow coverage

- Step 1 is `Watermark Generation and Crop-Robust Decoding` with Cover image x, 128x128 RGB, Message m, 64 bits, MBRS Encoder, watermarked image xhat, RandomCrop attack channel (training only), crop ratio approximately 0.3-1.0, MBRS Decoder, message prediction mhat, and message loss.
- Step 1 is explicitly tagged `Adopted MBRS backbone`; the footer states `Architecture unchanged` and `no new inference module`.
- Step 2 is `Training-Only Hard Local-Tail Supervision` and tagged `Our contribution`.
- Step 2 shows the paired x/xhat inputs, patch MSE map, P16/S8, 225 overlapping candidates, Hard Top-10% selection, 23/225 highest-error patches, local-tail loss, and a blue-to-red error heatmap with a highlighted selector grid.
- Step 2 states that global loss controls the mean while hard local-tail targets high-error local regions, and marks the branch training-only.
- Step 3 shows the formal main objective exactly as `L = 10 L_msg + 0.5 L_global + 0.5 L_tail` with `L_global = MSE(xhat,x)`.
- Step 3 contains a dashed `Validation-only Color-Aware Extensions` region, marked `Not part of the formal-test main method`, with Global OKLab OR Local chroma-tail as alternatives rather than simultaneous terms.
- Step 3 includes the requested compact key observations, including local distortion/P95/concentration, crop BER preservation, and validation CIEDE2000 effects.

## Method-definition integrity

- No model, loss coefficient, patch geometry, selection count, training configuration, checkpoint, or experiment result was changed.
- No training, inference, checkpoint loading, or model-output modification was performed; this revision changes figure rendering only.
- The attack channel is labeled training-only and is not presented as a permanent inference module.
