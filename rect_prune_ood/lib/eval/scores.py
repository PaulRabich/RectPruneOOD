import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric


# implementations of different scoring methods
class EnergyScore(Metric):
	"""A class to compute the energy score of a model's logits."""

	def __init__(self):
		super().__init__()
		self.add_state('scores', default=[], dist_reduce_fx='cat')

	def update(self, logits: Tensor) -> None:
		"""Updates the inner state with the energy score of the logits.

		The energy score is computed as the negative log-sum-exp of the logits.
		The logits are assumed to be the output of a model's forward pass.
		Parameters
		----------
		logits : (Tensor)
			The logits output by the model.
		"""
		scores = torch.logsumexp(logits.data, dim=1)
		self.scores.append(scores)

	def compute(self) -> Tensor:
		"""Computes the energy score.

		Returns
		-------
		Tensor
			The energy score of the logits.
		"""
		scores = torch.cat(self.scores)
		return scores


class EntropyScore(Metric):
	"""A class to compute the entropy score of a model's logits."""

	def __init__(self):
		super().__init__()
		self.add_state('scores', default=[], dist_reduce_fx='cat')

	def update(self, logits: Tensor) -> None:
		"""Updates the inner state with the entropy score of the logits.

		Parameters
		----------
		logits : Tensor
			The logits output by the model.
		"""
		scores = -torch.sum(
			F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1
		)
		self.scores.append(scores)

	def compute(self) -> Tensor:
		"""Computes the entropy score.

		Returns
		-------
		Tensor
			The entropy score of the logits.
		"""
		scores = torch.cat(self.scores)
		return scores


class MSPScore(Metric):
	"""A class to compute the maximum softmax probability (MSP) score of a model's logits."""

	def __init__(self):
		super().__init__()
		self.add_state('scores', default=[], dist_reduce_fx='cat')

	def update(self, logits: Tensor) -> None:
		"""Updates the inner state with the maximum softmax probability (MSP) score of the logits.

		Parameters
		----------
		logits : Tensor
			The logits output by the model.
		"""
		scores = torch.max(F.softmax(logits, dim=1), dim=1).values
		self.scores.append(scores)

	def compute(self) -> Tensor:
		"""Computes the maximum softmax probability (MSP) score.

		Returns
		-------
		Tensor
			The maximum softmax probability (MSP) score of the logits.
		"""
		scores = torch.cat(self.scores)
		return scores
