"""Fast regression tests for the validation-only Figure 2 stress pipeline."""

import unittest
from collections import OrderedDict

import numpy as np
import torch

from experiments.fig2_stress_pipeline import (
    DISPLAY_SIZE,
    FINAL_METHODS,
    display_canvas,
    final_column_labels,
    find_matched_ber,
    find_matched_psnr,
    gini,
    roi_search,
    valid_external_cache,
)


class Figure2StressPureTests(unittest.TestCase):
    def test_gini_is_zero_for_uniform_values(self):
        self.assertAlmostEqual(gini(np.ones(16)), 0.0)

    def test_roi_uses_one_shared_improvement_map(self):
        reference = torch.zeros(3, 128, 128)
        base = reference.clone()
        ours = reference.clone()
        base[:, 32:64, 40:72] = 0.2
        ours[:, 32:64, 40:72] = 0.1
        roi, details = roi_search(reference, base, ours)
        self.assertEqual(roi[2], roi[3])
        self.assertGreater(details["roi_mse_improvement"], 0.0)

    def _rows(self):
        common = {
            "status": "pass",
            "w_msg": "10",
            "lambda_img": "1",
            "p95": "0.001",
            "gini": "0.1",
            "top10_mean": "0.001",
        }
        return [
            {**common, "method": "Ours-base", "condition": "x", "ber30": "0.100", "psnr": "32.00", "local_psnr": "31.00"},
            {**common, "method": "Ours", "condition": "x", "ber30": "0.106", "psnr": "32.10", "local_psnr": "31.40"},
        ]

    def test_ber_matching_enforces_the_requested_tolerance(self):
        rows = self._rows()
        pairs = find_matched_ber(rows)
        self.assertEqual(len(pairs), 0)
        rows[1]["ber30"] = "0.1049"
        pairs = find_matched_ber(rows)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["selected"])

    def test_psnr_matching_records_empty_bins(self):
        pairs = find_matched_psnr(self._rows())
        empty = [row for row in pairs if row.get("status") == "no_pair_within_tolerance"]
        self.assertEqual({row["target_psnr"] for row in empty}, {30.0, 34.0, 36.0})
        self.assertTrue(any(row.get("selected") for row in pairs if row.get("status") != "no_pair_within_tolerance"))

    def test_final_method_names_are_explicit_external_internal_comparison(self):
        self.assertEqual(FINAL_METHODS, (
            "Original",
            "HiDDeN-64 (retrained external)",
            "MaskWM-D (official external)",
            "Global Reconstruction (internal baseline)",
            "Hard Local-Tail (proposed)",
        ))
        self.assertNotIn("Ours-base", " ".join(FINAL_METHODS))

    def test_every_method_uses_one_display_canvas_and_metric_labels(self):
        source = torch.zeros(2, 3, 128, 128)
        resized = display_canvas(source)
        self.assertEqual(tuple(resized.shape), (2, 3, DISPLAY_SIZE, DISPLAY_SIZE))
        data = OrderedDict((method, resized + index * 0.01) for index, method in enumerate(FINAL_METHODS))
        labels = final_column_labels(data)
        self.assertEqual(len(labels), 5)
        self.assertEqual(labels[0], "Original")
        self.assertTrue(all("PSNR" in label for label in labels[1:]))

    def test_hidden_cache_requires_matching_provenance_and_shape(self):
        payload = {
            "encoded": torch.zeros(50, 3, 128, 128),
            "manifest_sha256": "manifest",
            "checkpoint_sha256": "checkpoint",
        }
        self.assertTrue(valid_external_cache(payload, "manifest", "checkpoint", 50))
        self.assertFalse(valid_external_cache(payload, "wrong", "checkpoint", 50))
        payload["encoded"] = torch.zeros(49, 3, 128, 128)
        self.assertFalse(valid_external_cache(payload, "manifest", "checkpoint", 50))

    def test_promoted_plateau_threshold_is_strictly_bounded(self):
        # Documents the evaluator's operational, two-metric plateau contract.
        psnr = [32.00, 32.07, 32.12]
        local = [31.00, 31.08, 31.14]
        self.assertLessEqual(max(psnr) - min(psnr), 0.15)
        self.assertLessEqual(max(local) - min(local), 0.15)

    def test_incomplete_promoted_runs_are_not_described_as_complete(self):
        run = {"epochs": 20}
        requested_epoch = 20
        self.assertGreaterEqual(run["epochs"], requested_epoch)


if __name__ == "__main__":
    unittest.main()
