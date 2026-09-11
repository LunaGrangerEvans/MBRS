# Controlled hard-vs-soft conclusion

Status: seed17 gate complete; multi-seed conclusion pending temperature screening.

1. Global continuation is the control at PSNR 36.227, WorstPSNR 35.316, BER30 0.11313; source-checkpoint gain is deferred to the final multi-seed evaluator.
2. Hard versus control: ΔPSNR +0.156, ΔWorstPSNR +0.203, ΔBER30 +0.00006.
3. Soft training oscillation versus Hard: PSNR-step std 0.2252 vs 0.2050; BER-step std 0.02081 vs 0.02077.
4. Soft right-tail deltas: Top25MSE -0.000028, P95MSE -0.000040.
5. Soft local-versus-global gains: ΔWorstPSNR +0.104 versus ΔPSNR +0.077.
6. Soft BER30 delta: -0.00013.
7. Independent local perceptual deltas: local SSIM +0.0010, local LPIPS -0.00006.
8. Evidence is not paper-ready until a final temperature is selected and seed29/41 paired results exist.
9. Warm-up remains out of scope for this round.
10. Current claim remains: global average objectives do not explicitly control the high-distortion local tail.

Soft T=0.5 decision: **GO**. Checks: worst/top25 PSNR improves=True, top25 and P95 MSE decrease=True, global PSNR within -0.2 dB=True, BER30 within +0.005=True, local SSIM or LPIPS improves=True, Soft oscillation no worse than Hard=True.
