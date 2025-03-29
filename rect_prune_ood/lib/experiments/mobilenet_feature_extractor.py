from typing import Any

import torch
from torch import nn
from torch.nn import Module


# class to extract features of the penultimate layer after avg pooling from mobilenet
class MobileNetFeatureExtractor(Module):
	"""Feature extractor for MobileNetV2.

	Args:
		backbone (Module): Backbone model, typically MobileNetV2.
	"""	
	def __init__(self, backbone: Module) -> None:
		super().__init__()
		# bacbone is mobilenetV2
		self.backbone = backbone
		self.fc_layer: Module = self.backbone.classifier

	# get pooled features on forward call
	def forward(self, x: Any) -> torch.Tensor:
		"""Forward pass through the feature extractor.

		Args:
			x (Any): Input tensor.

		Returns:
			torch.Tensor: Pooled features from the backbone.
		"""		
		x = self.backbone.features(x)
		x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
		return x

	# return fully connected layer for pipeline
	def get_fc_layer(self) -> Module:
		"""Get the fully connected layer.

		Returns:
			Module: Fully connected layer of the backbone.
		"""		
		return self.fc_layer
