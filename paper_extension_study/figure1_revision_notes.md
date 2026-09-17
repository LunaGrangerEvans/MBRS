# Figure 1 Revision Notes

Do not modify or regenerate the frozen Figure 1 in this study. The bundled Figure 1 already identifies the adopted MBRS encoder/decoder path, the training-only crop channel, and the three final loss branches.

## Recommended communication hierarchy

Make the three branches read as a direct conceptual progression:

```text
Global RGB        → mean distortion control
Hard Local-Tail   → upper-tail local distortion control
Global OKLab      → color-aware refinement
```

Recommended wording changes for a future design pass are labels/visual hierarchy only:

- label the Global RGB branch with “mean distortion” or “average distortion”;
- label Hard Local-Tail with “upper-tail local distortion” and retain P16/S8, 225 candidates, and 23 selected patches;
- label OKLab with “color-aware refinement” and retain `lambda_OK = 0.059149764`;
- keep the inference row visually separate and explicitly unchanged;
- keep “training losses only” and “no additional inference module” visible;
- do not add Gini as a key effect, validation-only OKLab wording, Ours-base naming, or old six-column Figure 2 content.

The recommendation is not evidence and does not alter the frozen method or figure asset.
