"""Synthetic-only checks: no checkpoints, validation, or formal test files read."""

import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn

from experiments.evaluate_content_selector_final import (
    ATTACK_AREAS, REPEATS, alignment, claim_test_once, comparison_deltas,
    evaluate_with_per_image_ber, tail_values,
)


class SyntheticDecoder(nn.Module):
    def forward(self, encoded):
        return encoded.mean((1, 2, 3))[:, None].expand(-1, 64).gt(0).float()


class SyntheticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = SyntheticDecoder()
        self.encoder_calls = 0

    def encoder(self, images, _messages):
        self.encoder_calls += 1
        return images + .01


class FinalEvaluationTests(unittest.TestCase):
    def test_guard_rejects_rerun(self):
        with tempfile.TemporaryDirectory() as folder:
            claim_test_once(Path(folder))
            with self.assertRaises(FileExistsError):
                claim_test_once(Path(folder))

    def test_tails_have_explicit_direction_and_pooled_percentiles(self):
        values = torch.tensor([[1., 2., 3., 4.], [10., 20., 30., 40.]])
        upper, per_image = tail_values(values, "mse")
        lower, _ = tail_values(values, "ssim", higher_is_worse=False)
        self.assertEqual(upper["mse_top25"], 22.)
        self.assertEqual(lower["ssim_top25"], 5.5)
        self.assertEqual(per_image["mse_worst"].tolist(), [4., 40.])
        self.assertAlmostEqual(upper["mse_p90"], 33., places=5)

    def test_alignment_patch_identity_and_degradation_direction(self):
        scores = torch.arange(225).float()[None]
        rho, jac = alignment(scores, scores)
        self.assertEqual(float(rho[0]), 1.)
        self.assertEqual(float(jac[0]), 1.)
        rho, jac = alignment(scores, -scores)
        self.assertEqual(float(rho[0]), -1.)
        self.assertEqual(float(jac[0]), 0.)

    def test_local_specific_delta(self):
        result = comparison_deltas({"control": {"psnr": 30., "worst_psnr": 27.},
                                    "new": {"psnr": 30.2, "worst_psnr": 27.5}}, "control")
        self.assertAlmostEqual(result["new"]["local_specific_gain_db"], .3)

    def test_ber_hook_reproduces_aggregate_in_one_encoder_pass(self):
        torch.set_num_threads(2)
        images = torch.zeros(17, 3, 128, 128)
        images[::2] = -.2
        messages = torch.zeros(17, 64)
        messages[1::3] = 1
        masks = {name: [[torch.full((1, 1, 128, 128), float(repeat % 2))
                         for _ in range(2)] for repeat in range(REPEATS)]
                 for name in ATTACK_AREAS}
        model = SyntheticModel()
        def synthetic_metric(left, right):
            return (left - right).square().mean((1, 2, 3))
        (metrics, _, _), individual = evaluate_with_per_image_ber(
            model, images, messages, masks, synthetic_metric, torch.device("cpu"))
        self.assertEqual(model.encoder_calls, 2)
        self.assertEqual(len(model.decoder._forward_hooks), 0)
        for values in individual.values():
            self.assertEqual(values.shape, (17,))
        for key, values in individual.items():
            self.assertAlmostEqual(float(values.mean()), metrics[key], places=6)
        # Positive encoded samples decode 1 for two nonzero-mask repeats, else 0.
        self.assertAlmostEqual(float(individual["ber30"][1]), .6, places=6)
        self.assertEqual(float(individual["ber30"][0]), 0.)


if __name__ == "__main__":
    unittest.main()
