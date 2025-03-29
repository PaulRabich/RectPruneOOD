from typing import Any

import torch
from torch import nn
from torch.nn import Module


# class to extract features of the penultimate layer after avg pooling from mobilenet
class MobileNetFeatureExtractor(Module):
	def __init__(self, backbone: Module) -> None:
		super().__init__()
		# bacbone is mobilenetV2
		self.backbone = backbone
		self.fc_layer: Module = self.backbone.classifier

	# get pooled features on forward call
	def forward(self, x: Any) -> torch.Tensor:
		x = self.backbone.features(x)
		x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
		return x

	# return fully connected layer for pipeline
	def get_fc_layer(self) -> Module:
		return self.fc_layer
