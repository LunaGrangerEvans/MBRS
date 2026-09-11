"""Run with: python -m unittest experiments.test_oklab_global -v."""

import unittest

import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab

from experiments.losses import image_loss_components
from experiments.oklab_global import global_oklab_components, rgb_to_oklab


class TestOKLabGlobal(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(17)

    def random_rgb(self, shape=(2, 3, 4, 5), dtype=torch.float64):
        return torch.rand(shape, dtype=dtype, generator=self.generator)

    def test_known_srgb_primaries_and_white(self):
        rgb = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
            dtype=torch.float64,
        ).reshape(4, 3, 1, 1)
        expected = torch.tensor(
            [
                [0.62795536, 0.22486306, 0.12584630],
                [0.86643961, -0.23388757, 0.17949848],
                [0.45201372, -0.03245698, -0.31152815],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ).reshape_as(rgb)
        actual = rgb_to_oklab(rgb)
        self.assertEqual(actual.dtype, rgb.dtype)
        self.assertEqual(actual.device, rgb.device)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)

    def test_neutrals_and_both_inverse_transfer_branches(self):
        levels = torch.tensor(
            [0, 1e-8, 0.01, 0.04044, 0.04045, 0.04046, 0.18, 0.5, 1],
            dtype=torch.float64,
        )
        rgb = levels.view(-1, 1, 1, 1).expand(-1, 3, 1, 1)
        actual = rgb_to_oklab(rgb)
        expected_l = torch.tensor(
            [
                (x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4)
                ** (1.0 / 3.0)
                for x in levels.tolist()
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(actual[:, 0, 0, 0], expected_l, atol=1e-7, rtol=0)
        torch.testing.assert_close(
            actual[:, 1:], torch.zeros_like(actual[:, 1:]), atol=1e-7, rtol=0
        )
        self.assertTrue(torch.equal(actual[0], torch.zeros_like(actual[0])))

    def test_documented_near_black_regularization(self):
        # Below LMS eps the forward really is linear and black stays zero.
        rgb = torch.full((1, 3, 1, 1), 12.92 * 0.25e-12, dtype=torch.float64)
        lab = rgb_to_oklab(rgb)
        torch.testing.assert_close(rgb_to_oklab(2 * rgb), 2 * lab, atol=1e-15, rtol=0)
        self.assertAlmostEqual(lab[0, 0, 0, 0].item(), 0.25e-4, delta=1e-12)

    def test_identity_is_exactly_zero(self):
        for dtype in (torch.float32, torch.float64):
            for image in (
                torch.full((2, 3, 4, 5), -1.0, dtype=dtype),
                torch.zeros(2, 3, 4, 5, dtype=dtype),
                torch.ones(2, 3, 4, 5, dtype=dtype),
                self.random_rgb(dtype=dtype) * 2 - 1,
            ):
                for key, value in global_oklab_components(image, image).items():
                    with self.subTest(dtype=dtype, component=key):
                        self.assertEqual(value.shape, torch.Size([]))
                        self.assertEqual(value.item(), 0.0)

    def test_tint_grows_and_signed_deltas_reverse(self):
        cover = torch.full((2, 3, 4, 5), 0.5, dtype=torch.float64)
        previous = {"global_oklab": 0.0, "global_chroma": 0.0}
        for tint in (0.02, 0.08, 0.2):
            encoded = cover.clone()
            encoded[:, 0] += tint
            forward = global_oklab_components(encoded * 2 - 1, cover * 2 - 1)
            reverse = global_oklab_components(cover * 2 - 1, encoded * 2 - 1)
            for key in previous:
                self.assertGreater(forward[key].item(), previous[key])
                torch.testing.assert_close(forward[key], reverse[key])
                previous[key] = forward[key].item()
            for key in ("signed_oklab_da", "signed_oklab_db"):
                self.assertGreater(forward[key].item(), 0)
                torch.testing.assert_close(forward[key], -reverse[key])

    def test_pixel_norms_do_not_cancel_with_opposite_shifts(self):
        # Red/green swapped across pixels: mean color is identical, loss is not.
        encoded = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 0.0]]]], dtype=torch.float64
        )
        cover = encoded.flip(-1)
        result = global_oklab_components(encoded * 2 - 1, cover * 2 - 1)
        delta = np.array([0.62795536, 0.22486306, 0.12584630]) - np.array(
            [0.86643961, -0.23388757, 0.17949848]
        )
        self.assertAlmostEqual(
            result["global_oklab"].item(),
            np.sqrt(delta @ delta + 1e-12) - 1e-6,
            delta=2e-8,
        )
        self.assertAlmostEqual(
            result["global_chroma"].item(),
            np.sqrt(delta[1:] @ delta[1:] + 1e-12) - 1e-6,
            delta=2e-8,
        )
        self.assertEqual(result["signed_oklab_da"].item(), 0.0)
        self.assertEqual(result["signed_oklab_db"].item(), 0.0)
        # Adding an identical image to the batch halves all scalar means.
        batched = global_oklab_components(
            torch.cat((encoded, cover)) * 2 - 1, torch.cat((cover, cover)) * 2 - 1
        )
        for key in result:
            torch.testing.assert_close(
                batched[key], result[key] / 2, atol=1e-15, rtol=0
            )

    def test_conversion_backward_at_black_neutral_and_arbitrary_images(self):
        for dtype in (torch.float32, torch.float64):
            for level in (0.0, 1e-13, 0.5, 1.0, None):
                rgb = self.random_rgb(dtype=dtype)
                if level is not None:
                    rgb.fill_(level)
                rgb.requires_grad_()
                lab = rgb_to_oklab(rgb)
                for channel in range(3):
                    with self.subTest(dtype=dtype, level=level, channel=channel):
                        grad = torch.autograd.grad(
                            lab[:, channel].sum(), rgb, retain_graph=True
                        )[0]
                        self.assertTrue(torch.isfinite(lab).all())
                        self.assertTrue(torch.isfinite(grad).all())
                        self.assertGreater(grad.abs().sum().item(), 0)

    def test_each_component_retains_finite_gradients_for_both_inputs(self):
        for dtype in (torch.float32, torch.float64):
            black = torch.full((2, 3, 4, 5), -1.0, dtype=dtype)
            neutral = torch.zeros_like(black)
            arbitrary = self.random_rgb(dtype=dtype) * 2 - 1
            for first, second in (
                (black, black),
                (neutral, neutral),
                (arbitrary, arbitrary),
                (black, neutral),
                (neutral, black),
                (arbitrary, neutral),
            ):
                encoded = first.clone().requires_grad_()
                cover = second.clone().requires_grad_()
                result = global_oklab_components(encoded, cover)
                self.assertEqual(
                    set(result),
                    {
                        "global_oklab",
                        "global_chroma",
                        "signed_oklab_da",
                        "signed_oklab_db",
                    },
                )
                for key, value in result.items():
                    with self.subTest(dtype=dtype, component=key):
                        self.assertTrue(value.requires_grad)
                        self.assertTrue(torch.isfinite(value))
                        gradients = torch.autograd.grad(
                            value, (encoded, cover), retain_graph=True
                        )
                        for gradient in gradients:
                            self.assertTrue(torch.isfinite(gradient).all())
                            if first is arbitrary and second is neutral:
                                self.assertGreater(gradient.abs().sum().item(), 0)

    def test_conversion_gradcheck_away_from_zero(self):
        rgb = (self.random_rgb((1, 3, 2, 2)) * 0.6 + 0.2).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(rgb_to_oklab, (rgb,)))

    def test_components_gradcheck_away_from_zero(self):
        encoded = (self.random_rgb((1, 3, 2, 2)) * 1.2 - 0.6).requires_grad_()
        cover = (self.random_rgb((1, 3, 2, 2)) * 1.2 - 0.6).requires_grad_()

        def components_as_tuple(first, second):
            return tuple(global_oklab_components(first, second).values())

        self.assertTrue(torch.autograd.gradcheck(components_as_tuple, (encoded, cover)))

    def test_skimage_ciede2000_identity_and_tint(self):
        # CIEDE2000 is an independent perceptual sanity check, not an OKLab oracle.
        cover = np.full((4, 5, 3), 0.5, dtype=np.float64)
        reference = rgb2lab(cover)
        np.testing.assert_array_equal(deltaE_ciede2000(reference, reference), 0)
        previous = 0.0
        for tint in (0.02, 0.08, 0.2):
            encoded = cover.copy()
            encoded[..., 0] += tint
            distance = deltaE_ciede2000(reference, rgb2lab(encoded))
            self.assertTrue(np.isfinite(distance).all())
            self.assertGreater(float(distance.mean()), previous)
            previous = float(distance.mean())

    def test_additive_hard16_stride8_top10_backward(self):
        encoded = (self.random_rgb((1, 3, 32, 32)) * 2 - 1).requires_grad_()
        cover = self.random_rgb((1, 3, 32, 32)) * 2 - 1
        local = image_loss_components(
            encoded, cover, mode="topk", patch_size=16, patch_stride=8, topk_ratio=0.1
        )["local_mse"]
        global_loss = global_oklab_components(encoded, cover)["global_oklab"]
        local_grad = torch.autograd.grad(local, encoded, retain_graph=True)[0]
        global_grad = torch.autograd.grad(global_loss, encoded, retain_graph=True)[0]
        (local + 0.1 * global_loss).backward()
        torch.testing.assert_close(encoded.grad, local_grad + 0.1 * global_grad)
        self.assertTrue(torch.isfinite(encoded.grad).all())
        self.assertGreater(global_grad.abs().sum().item(), 0)

    def test_invalid_shapes_and_nonfloating_rgb(self):
        for shape in ((3, 4, 5), (1, 4, 4, 5), (1, 3, 4, 5, 1)):
            with self.assertRaisesRegex(ValueError, "NCHW RGB"):
                rgb_to_oklab(torch.zeros(shape))
        with self.assertRaisesRegex(TypeError, "floating-point"):
            rgb_to_oklab(torch.zeros(1, 3, 2, 2, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "same shape"):
            global_oklab_components(torch.zeros(2, 3, 4, 5), torch.zeros(1, 3, 4, 5))


if __name__ == "__main__":
    unittest.main()
