# External baseline crop protocol

This protocol is frozen for the completed TrustMark Q/P reference evaluation. No new MBRS training was run.

- Formal images: the same 50-image fixed manifest used by MBRS.
- Retained areas: 100%, 70%, 50%, 40%, 30%.
- Crop operation: image-space rectangular mask using the existing controlled crop semantics and fixed manifest coordinates.
- Decoder input: TrustMark's official preprocessing is retained; the crop wrapper applies the existing fixed normalized image-space masks and then invokes the official decoder. No additional resize is inserted by the wrapper.
- MBRS output: raw 64-bit BER.
- External outputs: raw BER only if raw bits are exposed; otherwise decoded-message success/detection rate and ECC-corrected success are separate fields.
- Payload/ECC mismatches force REFERENCE classification.

- TrustMark Q/P have verified official encoder/decoder hashes, successful clean encode/decode, 50 saved outputs per model, per-image hashes, and the crop trial records. They are placed in the table as `REFERENCE`, not `STRICT`, because TrustMark's BCH-5 protected 61-bit payload is not equivalent to MBRS raw 64-bit BER.
- StegaStamp remains blocked without a verified official SavedModel. HiDDeN is `REQUIRES RETRAINING` and was not trained in this phase.
