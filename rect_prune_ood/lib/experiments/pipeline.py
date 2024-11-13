from lightning import LightningModule
from torchmetrics import MetricCollection
import torch
from torch.nn import functional as F
import numpy as np


class Pipeline(LightningModule):
    def __init__(self, featureExtractor, rect_method, prune_method) -> None:
        super().__init__()
        self.featureExtractor = featureExtractor
        self.rect_method = rect_method
        self.prune_method = prune_method
        self.energy_scores = []



    def forward(self, x) -> torch.Tensor:
        out = self.featureExtractor(x)
        rectified_out = self.rect_method(out)
        pruned_out = self.prune_method(rectified_out)

        return pruned_out
    
    def training_step(self, batch, batch_idx) -> None:
        raise NotImplementedError("Only Evaluation is supported")
    
    def validation_step(self, batch, batch_idx) -> None:
        raise NotImplementedError("Only Evaluation is supported")
    
    def test_step(self, batch, batch_idx):
        image, label = batch

        images = image.cuda()
        labels = label.cuda()

        inputs = images.float()
        logits = self(inputs)

        outputs = F.softmax(logits, dim=1)
        outputs = outputs.detach().cpu()
        
        # compute the energy score
        scores = torch.logsumexp(outputs, dim=1).numpy() 
        self.energy_scores.extend(scores)



