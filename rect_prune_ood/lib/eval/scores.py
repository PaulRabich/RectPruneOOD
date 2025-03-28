import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchmetrics import Metric



#implementations of different scoring methods
class EnergyScore(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("scores", default=[], dist_reduce_fx="cat")


    def update(self, logits):
        scores = torch.logsumexp(logits.data, dim=1)
        self.scores.append(scores)
        
    def compute(self):
        scores = torch.cat(self.scores)
        return scores
    

class EntropyScore(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("scores", default=[], dist_reduce_fx="cat")


    def update(self, logits):
        scores = -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)
        self.scores.append(scores)
        
    def compute(self):
        scores = torch.cat(self.scores)
        return scores


class MSPScore(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("scores", default=[], dist_reduce_fx="cat")


    def update(self, logits):
        scores = torch.max(F.softmax(logits, dim=1), dim=1).values
        self.scores.append(scores)
        
    def compute(self):
        scores = torch.cat(self.scores)
        return scores



