# Missing citations for draft v1

No citation has been invented. Every [CITATION NEEDED] marker in [draft_v1.md](draft_v1.md) should be resolved with a verified primary source before submission.

| Draft location | Statement needing support | Literature type to locate | Citation status |
|---|---|---|---|
| Abstract / Introduction | Learned watermarking balances message recovery and image fidelity | Primary deep image-watermarking paper or survey | [CITATION NEEDED] |
| Introduction / Related Work §3.1 | HiDDeN, MBRS, StegaStamp, and TrustMark as learned watermarking contexts | Original papers and official model/repository documentation | [CITATION NEEDED] |
| Related Work §3.1 | Exact MBRS architecture, training channel, and protocol attribution | Original MBRS paper and official repository | [CITATION NEEDED] |
| Related Work §3.2 | Crop / partial-removal robustness as a watermarking problem | Primary robust watermarking papers that evaluate cropping or erasure | [CITATION NEEDED] |
| Related Work §3.3 | PSNR, SSIM, MS-SSIM, and LPIPS measure different image-quality properties | Original metric papers and official metric documentation | [CITATION NEEDED] |
| Related Work §3.3 | Hard top-k / upper-tail risk objective terminology | Primary tail-risk, CVaR, top-k loss, or robust optimization paper; avoid claiming novelty without checking | [CITATION NEEDED] |
| Related Work §3.3 / §5.4 | Gini and CV as distribution or inequality statistics | Original Gini source and a statistical reference for coefficient of variation | [CITATION NEEDED] |
| Related Work §3.4 / Method §4.4 | OKLab design, standard linear-sRGB conversion, and perceptual color coordinates | Björn Ottosson's Oklab source plus a peer-reviewed color-science reference if required by venue | [CITATION NEEDED] |
| Method §4.4 / §5.4 | CIELAB D65 and CIEDE2000 definitions | CIEDE2000 standard/original paper and color-management reference | [CITATION NEEDED] |
| Experimental Setup §5.1 | DIV2K dataset and the project split context | DIV2K dataset paper; clearly distinguish project split from official DIV2K test set | [CITATION NEEDED] |
| Experimental Setup §5.3 | AlexNet LPIPS v0.1 and learned perceptual similarity interpretation | Original LPIPS paper and official implementation/version documentation | [CITATION NEEDED] |
| External Reference §11 | Official TrustMark Q/P release and model semantics | TrustMark paper plus official Adobe repository/model documentation | [CITATION NEEDED] |
| External Reference §11 | StegaStamp and HiDDeN attribution, if discussed in surrounding text | Original paper and official repository/checkpoint documentation | [CITATION NEEDED] |
| Introduction / Conclusion | Any claim that the proposed objective is novel or first | Targeted literature search on local distortion/tail supervision in watermarking | [CITATION NEEDED] |

## Citation handling rules

1. Do not replace a marker with a citation merely because a repository exists; the citation must support the exact paper claim.
2. Cite the original metric or method source where the method is introduced, and cite a repository only for implementation/checkpoint provenance.
3. Do not use TrustMark or other external results as a citation for strict superiority; they are reference measurements under different payload/ECC and preprocessing semantics.
4. Do not add a cross-seed significance citation or claim: the current teacher-defined evidence is seed17-only.
5. If the local-tail formulation overlaps prior work, weaken the contribution language and cite that work instead of presenting the formulation as new.
