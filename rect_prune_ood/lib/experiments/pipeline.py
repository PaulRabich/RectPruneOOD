import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.nn import Module
from torchmetrics import MetricCollection

from rect_prune_ood.lib.eval.scores import EnergyScore, EntropyScore, MSPScore
from rect_prune_ood.lib.util.names import get_module_list


# pipeline for (APES) methods
class Pipeline(LightningModule):
	"""Applies activation shaping methods to data and performs post hoc OOD prediction on input data

		Args:
			featureExtractor (MobileNetFeatureExtractor | FeatureExtractor): gets activations of penultimate layer of backbone
			method_name (str): String encoding methods to be applied on data. See rect_prune_ood.lib.util.names for instructions on string construction.
			fc_layer (Module): Fully connected layer matching to the backbone to get final prediction result
			caputure_activations (bool, optional): If we want to save the layer activations in addition to the results. Defaults to False.
		"""
	def __init__(
		self,
		featureExtractor,
		method_name: str,
		fc_layer: Module,
		caputure_activations=False,
	) -> None:
		super().__init__()
		self.featureExtractor = featureExtractor
		self.method_name = method_name
		# use string encoding to get list of torch modules to be applied
		self.method_list = get_module_list(method_name)
		# metrics to calculate
		self.metrics = MetricCollection(
			{
				'energy_score': EnergyScore(),
				'msp_score': MSPScore(),
				'entropy_score': EntropyScore(),
			}
		)
		self.prediction_list = []
		self.results = {}

		self.fc_layer = fc_layer
		self.capture_activations = caputure_activations
		if self.capture_activations:
			self.activation_list = []
			self.activations: np.ndarray = None

	# forwards step of methods
	def forward(self, x:torch.Tensor) -> torch.Tensor:
		"""Forward pass through the pipeline.
		Apply the feature extractor, followed by the methods specified in method_list,

		Args:
			x (torch.Tensor): Input tensor.

		Returns:
			torch.Tensor: Output tensor after applying the feature extractor and methods.
		"""		
		# get features of layer
		x = self.featureExtractor(x)
		# potentially save them
		if self.capture_activations:
			self.activation_list.append(x.squeeze())
		# apply all methods
		for method in self.method_list:
			x = method(x)
		# continue as usual
		x = torch.flatten(x, 1)
		x = self.fc_layer(x)

		return x

	def training_step(self, batch:torch.Tensor, batch_idx:int):
		raise NotImplementedError(
			'The aim of this class is only to evaluate the performance of the model'
		)

	def validation_step(self, batch, batch_idx):
		raise NotImplementedError(
			'The aim of this class is only to evaluate the performance of the model'
		)

	# lightning implementation of test step
	def test_step(self, batch:torch.Tensor, batch_idx:int) -> None:
		"""Test step for the pipeline.

		Args:
			batch (torch.Tensor): Input batch of data.
			batch_idx (int): Batch index.
		"""		
		image, label = batch

		images = image.cuda()

		inputs = images.float()
		# predict
		with torch.no_grad():
			logits = self(inputs)
			outputs = F.softmax(logits, dim=1)

			prediction = torch.argmax(outputs, dim=1)
			confidence = torch.max(outputs, dim=1)[0]

			self.prediction_list.append((prediction, label, confidence))

		# compute the metrics
		self.metrics.update(logits)

	def on_test_epoch_end(self) -> None:
		"""Saving the results of the test step and resetting the metrics and prediction list.
		"""		
		# Compute Metrics
		self.results = {}
		self.results = self.metrics.compute()

		self.results['energy_score'] = self.results['energy_score'].cpu().numpy()
		self.results['msp_score'] = self.results['msp_score'].cpu().numpy()
		self.results['entropy_score'] = self.results['entropy_score'].cpu().numpy()

		prediction, label, confidence = zip(*self.prediction_list)
		prediction = torch.cat(prediction).cpu().numpy()
		label = torch.cat(label).cpu().numpy()
		confidence = torch.cat(confidence).cpu().numpy()

		self.results['prediction'] = prediction
		self.results['label'] = label
		self.results['confidence'] = confidence

		self.metrics.reset()
		self.prediction_list = []

		# Save activations and refresh activation cache
		if self.capture_activations:
			self.activations = torch.cat(self.activation_list).cpu().numpy()
			self.activation_list = []
