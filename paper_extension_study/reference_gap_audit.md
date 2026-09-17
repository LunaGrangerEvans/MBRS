# Related-Work and Reference Gap Audit

This audit uses only the authoritative bundle and paper-level audit. The bundled `paper/references.bib` is a TODO placeholder and contains no verified BibTeX entries. The Prism workspace bibliography is also a TODO placeholder. Therefore no authoritative reference is currently verified in the bundle, and every category below remains a TODO until a source is independently verified and explicitly approved.

| Category | Why a reference is needed | Verified in authoritative bundle | Status |
|---|---|---|---|
| Robust deep image watermarking | Establish prior work on image watermarking under cropping or other distortions and position the robustness problem. | None | TODO: verify primary papers. |
| MBRS | Cite the adopted MBRS backbone and distinguish the unchanged inference architecture from the proposed training losses. | None | TODO: verify the original MBRS paper. |
| HiDDeN | Identify the external qualitative reference method used in Figure 2. | None | TODO: verify the primary HiDDeN paper and exact citation. |
| MaskWM | Identify the external qualitative reference method used in Figure 2. | None | TODO: verify the primary MaskWM paper and exact citation. |
| LPIPS / perceptual fidelity | Support the methodological definition and interpretation of LPIPS as an evaluation metric. | None | TODO: verify the primary LPIPS paper. |
| OKLab | Support the color-space definition used by the global OKLab training term. | None | TODO: verify the authoritative OKLab source. |
| CIEDE2000 | Support the methodological definition if CIEDE2000 is explained or emphasized in the paper. | None | TODO: verify the standard/primary methodological source. |
| Tail risk / CVaR-like interpretation | Support the cautious interpretation of Hard Local-Tail as empirical upper-tail risk or CVaR-like supervision if this language enters the paper. | None | TODO: verify a primary tail-risk/CVaR source; do not claim exact equivalence. |

## Constraints

- Do not fabricate references or BibTeX entries.
- Do not import citations from historical paper drafts or non-authoritative repository files.
- Do not add a citation merely because a name appears in a provenance path.
- If the CVaR-like interpretation is not cited, keep the wording explicitly heuristic (“can be viewed as”, “CVaR-like”).
