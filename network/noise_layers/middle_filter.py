import torch
import torch.nn as nn
import torch.nn.functional as F

try:
	from kornia.filters import MedianBlur
except ImportError:
	MedianBlur = None


class MF(nn.Module):

	def __init__(self, kernel):
		super(MF, self).__init__()
		if kernel <= 0 or kernel % 2 == 0:
			raise ValueError("median kernel must be a positive odd number")
		self.kernel_size = kernel
		self.middle_filter = MedianBlur((kernel, kernel)) if MedianBlur is not None else None

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		if self.middle_filter is not None:
			return self.middle_filter(image)
		padding = self.kernel_size // 2
		padded = F.pad(image, (padding, padding, padding, padding), mode="reflect")
		patches = F.unfold(padded, self.kernel_size)
		batch, channels, height, width = image.shape
		patches = patches.view(batch, channels, self.kernel_size * self.kernel_size, height * width)
		return patches.median(dim=2)[0].view(batch, channels, height, width)
