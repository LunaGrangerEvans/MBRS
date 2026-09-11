import torch
import torch.nn as nn
import torch.nn.functional as F

try:
	from kornia.filters import GaussianBlur2d
except ImportError:
	GaussianBlur2d = None


class GF(nn.Module):

	def __init__(self, sigma, kernel=7):
		super(GF, self).__init__()
		if GaussianBlur2d is not None:
			self.gaussian_filter = GaussianBlur2d((kernel, kernel), (sigma, sigma))
			self.kernel = None
		else:
			if kernel <= 0 or kernel % 2 == 0:
				raise ValueError("gaussian kernel must be a positive odd number")
			coords = torch.arange(kernel, dtype=torch.float) - kernel // 2
			one_dim = torch.exp(-(coords * coords) / (2 * sigma * sigma))
			gaussian = one_dim[:, None] * one_dim[None, :]
			gaussian = gaussian / gaussian.sum()
			self.register_buffer("kernel", gaussian.view(1, 1, kernel, kernel))
			self.gaussian_filter = None

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		if self.gaussian_filter is not None:
			return self.gaussian_filter(image)
		channels = image.shape[1]
		kernel = self.kernel.expand(channels, 1, -1, -1)
		return F.conv2d(image, kernel, padding=self.kernel.shape[-1] // 2, groups=channels)
