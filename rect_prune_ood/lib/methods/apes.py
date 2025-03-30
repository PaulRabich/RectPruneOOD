import os

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from rect_prune_ood.lib.util.paths import get_top_level_path


class APES(Module):
	def __init__(self, mask_name: str) -> None:
		"""Main implementation of APES. The mask is loaded from the results directory.

		Args:
			mask_name (str): Name of the mask file without the .npy extension or path.

		Raises:
			FileNotFoundError: If the mask file does not exist in the results directory.
		"""
		super().__init__()
		mask_path = os.path.join(
			get_top_level_path(), 'results', f'masks/{mask_name}.npy'
		)
		if not os.path.exists(mask_path):
			raise FileNotFoundError(f'Mask file {mask_path} does not exist.')
		self.mask = np.load(mask_path)
		self.mask = torch.from_numpy(self.mask).float()
		self.mask = self.mask.reshape(-1, 1, 1).cuda()

	def forward(self, x: Tensor) -> Tensor:
		assert x.dim() == 4

		x = x * self.mask

		return x
