"""Regression tests for the validation-only Figure 2 search primitives."""

import unittest

import numpy as np
import torch

from experiments.generate_fig2_qualitative_search import (
    MANIFEST,
    ROI_SIZES,
    gini,
    roi_search,
    stressed,
)


class Figure2SearchTests(unittest.TestCase):
    def test_manifest_constant_is_validation_not_formal_test(self):
        self.assertIn("content_selector/validation_manifest.pt", str(MANIFEST))
        self.assertNotIn("uniform_eval_manifest", str(MANIFEST))

    def test_gini_contract(self):
        self.assertEqual(gini(np.zeros(4)), 0.0)
        self.assertAlmostEqual(gini(np.ones(4)), 0.0)
        self.assertAlmostEqual(gini(np.array([0.0, 0.0, 0.0, 1.0])), 0.75)

    def test_equal_alpha_stress_clips_and_reports_fraction(self):
        reference = torch.full((1, 3, 2, 2), 0.5)
        encoded = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]] * 3])
        value, fraction = stressed(reference, encoded, 2.0)
        self.assertTrue(torch.equal(value, encoded))
        self.assertEqual(fraction, 1.0)

    def test_roi_uses_base_minus_ours_improvement(self):
        ramp = torch.linspace(0, 1, 128).repeat(128, 1)
        original = torch.stack((ramp, ramp.T, ramp))
        base = original.clone()
        ours = original.clone()
        base[:, 48:80, 48:80] += 0.1
        roi, details = roi_search(original, base, ours)
        x, y, size, _ = roi
        self.assertIn(size, ROI_SIZES)
        self.assertGreater(details["roi_mse_improvement"], 0)
        self.assertLess(x, 80)
        self.assertLess(y, 80)
        self.assertGreater(x + size, 48)
        self.assertGreater(y + size, 48)


if __name__ == "__main__":
    unittest.main()
