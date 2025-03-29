import os
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from rect_prune_ood.lib.util.paths import get_top_level_path


class APES(Module):
	def __init__(self, mask_name:str) -> None:
		super().__init__()
		mask_path = os.path.join(get_top_level_path(), f"mask/{mask_name}.npy")
		if not os.path.exists(mask_path):
			raise FileNotFoundError(f"Mask file {mask_path} does not exist.")
		self.mask = np.load(mask_path)	

	def forward(self, x: Tensor) -> Tensor:
		x = x * self.mask
		return x

