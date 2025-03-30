import os

import numpy as np
import torch
from torch.nn import Module

class AshBPruningModule(Module):
	def __init__(self, percentage=65):
		"""Reimplementaion of the AshB method from the paper.

		Args:
			percentage (int, optional): The percentage of pruning. Defaults to 65.
		"""		
		super().__init__()
		self.percentage = percentage

	def forward(self, x:torch.Tensor) -> torch.Tensor:
		"""Forward pass of the AshB module.

		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

		Returns:
			torch.Tensor: Output tensor after applying the AshB pruning.
		"""		
		x = self.ash_b(x, self.percentage)
		return x

	def ash_b(self, x, percentile):
		assert x.dim() == 4
		assert 0 <= percentile <= 100
		b, c, h, w = x.shape

		# calculate the sum of the input per sample
		s1 = x.sum(dim=[1, 2, 3])

		n = x.shape[1:].numel()
		k = n - int(np.round(n * percentile / 100.0))
		t = x.view((b, c * h * w))
		v, i = torch.topk(t, k, dim=1)
		fill = s1 / k
		fill = fill.unsqueeze(dim=1).expand(v.shape)
		t.zero_().scatter_(dim=1, index=i, src=fill)
		return x

	def __str__(self):
		return super().__str__() + f' with percentage {self.percentage}'


class SCALEModule(Module):
	def __init__(self, percentage=90):
		"""Reimplementation of the SCALE method.

		Args:
			percentage (int, optional): Percentage of pruning. Defaults to 90.
		"""		
		super().__init__()
		self.percentage = percentage

	def forward(self, x):
		x = self.scale(x, self.percentage)
		return x

	def scale(self, x, percentile=85):
		input = x.clone()
		assert x.dim() == 4
		assert 0 <= percentile <= 100
		b, c, h, w = x.shape

		# calculate the sum of the input per sample
		s1 = x.sum(dim=[1, 2, 3])
		n = x.shape[1:].numel()
		k = n - int(np.round(n * percentile / 100.0))
		t = x.view((b, c * h * w))
		v, i = torch.topk(t, k, dim=1)
		t.zero_().scatter_(dim=1, index=i, src=v)

		# calculate new sum of the input per sample after pruning
		s2 = x.sum(dim=[1, 2, 3])

		# apply sharpening
		scale = s1 / s2

		return input * torch.exp(scale[:, None, None, None])

	def __str__(self):
		return super().__str__() + f' with percentage {self.percentage}'


class AshSPruningModule(Module):
	def __init__(self, percentage=90):
		"""Reimplementation of the AshS method.

		Args:
			percentage (int, optional): Pruning percentage. Defaults to 90.
		"""		
		super().__init__()
		self.percentage = int(percentage)

	def forward(self, x):
		x = self.ash_s(x, self.percentage)
		return x

	def ash_s(self, x, percentile):
		percentile = int(percentile)
		assert x.dim() == 4
		assert 0 <= percentile <= 100
		b, c, h, w = x.shape

		# calculate the sum of the input per sample
		s1 = x.sum(dim=[1, 2, 3])
		n = x.shape[1:].numel()
		k = n - int(np.round(n * percentile / 100.0))
		t = x.view((b, c * h * w))
		v, i = torch.topk(t, k, dim=1)
		t.zero_().scatter_(dim=1, index=i, src=v)

		# calculate new sum of the input per sample after pruning
		s2 = x.sum(dim=[1, 2, 3])

		# apply sharpening
		scale = s1 / s2
		x = x * torch.exp(scale[:, None, None, None])

		return x

	def __str__(self):
		return super().__str__() + f' with percentage {self.percentage}'


class AshPPruningModule(Module):
	def __init__(self, percentage=50):
		"""REimplementation of the AshP method.

		Args:
			percentage (int, optional): Pruning percentage. Defaults to 50.
		"""		
		super(AshPPruningModule, self).__init__()
		self.percentage = percentage

	def forward(self, x):
		x = self.ash_p(x, self.percentage)
		return x

	def ash_p(self, x, percentile):
		assert x.dim() == 4
		assert 0 <= percentile <= 100

		b, c, h, w = x.shape

		n = x.shape[1:].numel()
		k = n - int(np.round(n * percentile / 100.0))
		t = x.view((b, c * h * w))
		v, i = torch.topk(t, k, dim=1)
		t.zero_().scatter_(dim=1, index=i, src=v)

		return x

	def __str__(self):
		return super().__str__() + f' with percentage {self.percentage}'


class ReactRectModule(Module):
	def __init__(self, threshold=1.0):
		"""Reimplementation of the ReactRect method.

		Args:
			threshold (float, optional): Threshold for clipping the values. Defaults to 1.0.
		"""		
		super(ReactRectModule, self).__init__()
		self.threshold = threshold

	def forward(self, x):
		return x.clip(max=self.threshold)

	def __str__(self):
		return super().__str__() + f' with threshold {self.threshold}'
