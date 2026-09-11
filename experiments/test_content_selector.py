"""Protocol checks for detached content ranking, independent of checkpoints."""

import sys
import unittest
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.content_selector import (
    content_activity,
    patches,
    robust_normalize,
    selected_indices,
    selector_scores,
)
from experiments.losses import image_loss_components


class ContentSelectorTests(unittest.TestCase):
    def test_fixed_grid_is_row_major_with_no_padding(self):
        images = torch.arange(2 * 3 * 128 * 128).reshape(2, 3, 128, 128).float()
        actual = patches(images)
        expected = torch.stack(
            [images[:, :, y:y + 16, x:x + 16]
             for y in range(0, 113, 8) for x in range(0, 113, 8)], dim=1
        )
        self.assertEqual(actual.shape, (2, 225, 3, 16, 16))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_grid_rejects_non_protocol_image_shapes(self):
        for shape in [(3, 128, 128), (1, 3, 127, 128), (1, 3, 128, 136)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                patches(torch.zeros(shape))

    def test_original_features_use_luminance_and_are_detached(self):
        # A red-only horizontal ramp has known luminance slope and variance.
        x = torch.arange(128, dtype=torch.float64) / 127
        original = torch.full((1, 3, 128, 128), -1.0, dtype=torch.float64)
        original[:, 0] = 2 * x - 1
        original.requires_grad_()
        activity = content_activity(original)
        slope = 0.299 / 127
        torch.testing.assert_close(
            activity["gradient"], torch.full((1, 225), slope, dtype=torch.float64)
        )
        torch.testing.assert_close(
            activity["variance"],
            torch.full((1, 225), slope ** 2 * (16 ** 2 - 1) / 12,
                       dtype=torch.float64),
        )
        self.assertLess(activity["hf"].max().item(), 1e-28)
        for name, values in activity.items():
            with self.subTest(feature=name):
                self.assertEqual(values.shape, (1, 225))
                self.assertFalse(values.requires_grad)
                self.assertIsNone(values.grad_fn)
        # Distortion is deliberately varied while the sole feature input stays fixed.
        for encoded in (original.detach(), -original.detach()):
            mse = patches(encoded - original).square().mean((2, 3, 4))
            scores = selector_scores(mse, activity, "gradient", 1.0)
            self.assertFalse(scores.requires_grad)
        for name, values in content_activity(original).items():
            torch.testing.assert_close(values, activity[name])

    def test_constant_image_has_zero_activity_without_padding_edges(self):
        activity = content_activity(torch.full((2, 3, 128, 128), 0.25))
        for name, values in activity.items():
            with self.subTest(feature=name):
                torch.testing.assert_close(values, torch.zeros_like(values), rtol=0, atol=0)

    def test_robust_normalization_is_per_image_and_clips_outliers(self):
        values = torch.arange(101, dtype=torch.float64)
        values[0], values[-1] = -1e12, 1e12
        batch = torch.stack((values, 3 * values + 100, torch.ones(101)))
        batch.requires_grad_()
        actual = robust_normalize(batch)
        expected = ((torch.arange(101, dtype=torch.float64) - 5) / 90).clamp(0, 1)
        torch.testing.assert_close(actual[0], expected)
        torch.testing.assert_close(actual[1], expected)
        torch.testing.assert_close(actual[2], torch.zeros(101, dtype=torch.float64))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertFalse(actual.requires_grad)
        tiny = torch.linspace(0, 1e-13, 225, dtype=torch.float64).unsqueeze(0)
        torch.testing.assert_close(robust_normalize(tiny), torch.zeros_like(tiny))

    def test_exact_frozen_candidate_formulas(self):
        mse = torch.tensor([[3.0, 4.0, 5.0]], requires_grad=True)
        gradient = torch.tensor([[0.0, 0.5, 1.0]], requires_grad=True)
        hf = torch.tensor([[1.0, 0.5, 0.0]], requires_grad=True)
        activity = {"gradient_normalized": gradient, "hf_normalized": hf}
        torch.testing.assert_close(selector_scores(mse, activity), mse.detach())
        for feature in ("gradient", "gradient_hf"):
            value = gradient if feature == "gradient" else (gradient + hf) / 2
            for alpha in (0.5, 1.0, 2.0):
                with self.subTest(feature=feature, alpha=alpha):
                    scores = selector_scores(mse, activity, feature, alpha)
                    torch.testing.assert_close(scores, mse.detach() / (1 + alpha * value))
                    self.assertFalse(scores.requires_grad)

    def test_invalid_candidates_and_ratios_are_rejected(self):
        mse = torch.ones(1, 225)
        activity = {"gradient_normalized": mse, "hf_normalized": mse}
        for feature, alpha in [("hf", 1.0), ("variance", 1.0), ("gradient", 0.0),
                               ("gradient", -1.0), ("gradient_hf", 0.75)]:
            with self.subTest(feature=feature, alpha=alpha), self.assertRaises(ValueError):
                selector_scores(mse, activity, feature, alpha)
        for ratio in (0, 0.05, 0.2, 0.5, 1.0):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                selected_indices(mse, ratio)

    def test_stable_ties_and_ceiling_counts(self):
        scores = torch.ones(2, 225, requires_grad=True)
        for ratio, count in [(0.1, 23), (0.25, 57)]:
            indices = selected_indices(scores, ratio)
            torch.testing.assert_close(indices, torch.arange(count).expand(2, -1))
        mixed = scores.detach().clone()
        mixed[:, 100:110] = 2
        expected = torch.cat((torch.arange(100, 110), torch.arange(13))).expand(2, -1)
        torch.testing.assert_close(selected_indices(mixed), expected)

    def test_selection_normalizes_ranking_but_loss_uses_original_mse(self):
        mse = torch.ones(1, 225)
        mse[:, :23] = 3.0
        mse[:, 23:46] = 2.0
        mse.requires_grad_()
        activity = torch.zeros_like(mse)
        activity[:, :23] = 1.0
        activity[:, 23:46] = 0.25
        activity.requires_grad_()
        scores = selector_scores(mse, {"gradient_normalized": activity}, "gradient", 2.0)
        indices = selected_indices(scores)
        torch.testing.assert_close(indices, torch.arange(23, 46).unsqueeze(0))
        self.assertFalse(scores.requires_grad)
        # This is the authorized future loss contract, not a training integration test.
        loss = mse.gather(1, indices).mean()
        self.assertEqual(loss.item(), 2.0)
        self.assertNotAlmostEqual(loss.item(), scores.gather(1, indices).mean().item())
        loss.backward()
        expected_gradient = torch.zeros_like(mse)
        expected_gradient[:, 23:46] = 1 / 23
        torch.testing.assert_close(mse.grad, expected_gradient)
        self.assertIsNone(activity.grad)

    def test_training_features_match_full_diagnostics(self):
        generator = torch.Generator().manual_seed(170903)
        original = torch.rand(2, 3, 128, 128, generator=generator) * 2 - 1
        original.requires_grad_()
        full = content_activity(original)
        training = content_activity(original, diagnostics=False)
        self.assertEqual(set(training), {"gradient_normalized"})
        torch.testing.assert_close(
            training["gradient_normalized"], full["gradient_normalized"], rtol=0, atol=0
        )
        self.assertFalse(training["gradient_normalized"].requires_grad)

    @staticmethod
    def training_fixture(device="cpu"):
        generator = torch.Generator().manual_seed(170904)
        cover = (torch.rand(2, 3, 128, 128, generator=generator) * 2 - 1).to(device)
        residual = (torch.randn(2, 3, 128, 128, generator=generator) * 0.1).to(device)
        return (cover + residual).requires_grad_(), cover.requires_grad_()

    @staticmethod
    def independent_patch_mse(encoded, cover):
        return torch.stack([
            (encoded[:, :, y:y + 16, x:x + 16]
             - cover[:, :, y:y + 16, x:x + 16]).square().mean((1, 2, 3))
            for y in range(0, 113, 8) for x in range(0, 113, 8)
        ], dim=1)

    def test_integrated_training_loss_and_gradients_use_original_selected_mse(self):
        encoded, cover = self.training_fixture()
        raw = self.independent_patch_mse(encoded, cover)
        activity = content_activity(cover)["gradient_normalized"]
        normalized = raw.detach() / (1 + activity)
        indices = torch.argsort(normalized, descending=True, stable=True, dim=1)[:, :23]
        expected = raw.gather(1, indices).mean()
        actual = image_loss_components(
            encoded, cover, mode="contentaware_gradient", patch_size=16,
            patch_stride=8, topk_ratio=0.1,
        )["local_mse"]
        torch.testing.assert_close(actual, expected)
        self.assertGreater(abs(actual.item() - normalized.gather(1, indices).mean().item()), 1e-5)
        actual_gradients = torch.autograd.grad(actual, (encoded, cover), retain_graph=True)
        expected_gradients = torch.autograd.grad(expected, (encoded, cover))
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertTrue(torch.count_nonzero(actual_gradients[0]).item() > 0)

    def test_integrated_training_mode_rejects_non_frozen_grid_and_ratio(self):
        encoded, cover = self.training_fixture()
        for size, stride, ratio in [(32, 8, 0.1), (16, 16, 0.1),
                                    (16, None, 0.1), (16, 8, 0.25)]:
            with self.subTest(size=size, stride=stride, ratio=ratio):
                with self.assertRaisesRegex(ValueError, "Patch16/stride8/Top10"):
                    image_loss_components(encoded, cover, mode="contentaware_gradient",
                                          patch_size=size, patch_stride=stride, topk_ratio=ratio)
        with self.assertRaisesRegex(ValueError, "128x128"):
            image_loss_components(encoded[:, :, :120], cover[:, :, :120],
                                  mode="contentaware_gradient", patch_size=16,
                                  patch_stride=8, topk_ratio=0.1)

    def test_legacy_topk_and_mean_losses_keep_original_formulas(self):
        encoded, cover = self.training_fixture()
        raw = self.independent_patch_mse(encoded, cover)
        for mode, expected in [("topk", raw.topk(23, dim=1).values.mean()),
                               ("mean", raw.mean())]:
            with self.subTest(mode=mode):
                actual = image_loss_components(encoded, cover, mode=mode, patch_size=16,
                                               patch_stride=8, topk_ratio=0.1)["local_mse"]
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(
                    torch.autograd.grad(actual, encoded, retain_graph=True)[0],
                    torch.autograd.grad(expected, encoded, retain_graph=True)[0],
                )

    def check_rng_unchanged(self, device):
        encoded, cover = self.training_fixture(device)
        cpu_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        loss = image_loss_components(encoded, cover, mode="contentaware_gradient",
                                     patch_size=16, patch_stride=8, topk_ratio=0.1)["local_mse"]
        loss.backward()
        torch.testing.assert_close(torch.get_rng_state(), cpu_before, rtol=0, atol=0)
        for before, after in zip(cuda_before, torch.cuda.get_rng_state_all() if cuda_before else []):
            torch.testing.assert_close(after, before, rtol=0, atol=0)

    def test_cpu_training_loss_does_not_consume_rng(self):
        self.check_rng_unchanged("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_training_loss_does_not_consume_rng(self):
        self.check_rng_unchanged("cuda")


if __name__ == "__main__":
    unittest.main()
