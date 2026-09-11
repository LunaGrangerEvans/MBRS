import unittest

import torch

from experiments.color_losses import oklab_pixel_distances, oklab_tail_components, rgb_to_oklab


class TestColorLosses(unittest.TestCase):
    def test_identical_rgb_has_zero_oklab_distance(self):
        image = torch.rand(2, 3, 32, 32)
        distance, chroma = oklab_pixel_distances(image * 2 - 1, image * 2 - 1)
        self.assertLess(float(distance.max()), 1e-5)
        self.assertLess(float(chroma.max()), 1e-5)

    def test_color_shift_increases_distance(self):
        image = torch.full((1, 3, 16, 16), 0.5)
        shifted = image.clone()
        shifted[:, 0] = 0.8
        base = oklab_pixel_distances(image * 2 - 1, image * 2 - 1)[0].mean()
        changed = oklab_pixel_distances(shifted * 2 - 1, image * 2 - 1)[0].mean()
        self.assertAlmostEqual(float(base), 0.0, places=5)
        self.assertGreater(float(changed), 0.01)

    def test_oklab_tail_backpropagates(self):
        encoded = (torch.rand(2, 3, 16, 16) * 2 - 1).requires_grad_()
        cover = torch.rand(2, 3, 16, 16) * 2 - 1
        components = oklab_tail_components(encoded, cover, patch_size=5, patch_stride=1, topk_ratio=0.1)
        self.assertTrue(torch.isfinite(components["local_oklab"]))
        components["local_oklab"].backward()
        self.assertIsNotNone(encoded.grad)
        self.assertTrue(torch.isfinite(encoded.grad).all())

    def test_oklab_conversion_is_finite(self):
        lab = rgb_to_oklab(torch.rand(2, 3, 16, 16))
        self.assertTrue(torch.isfinite(lab).all())


if __name__ == "__main__":
    unittest.main()
