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

        prune_fc = self.featureExtractor.get_fc_layer()

        pruned_out = torch.flatten(pruned_out, 1)
        final_out = prune_fc(pruned_out)
        return final_out
    
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
        outputs = outputs.detach().cpu().numpy()
        preds = np.argmax(outputs, axis=1)
        confs = np.max(outputs, axis=1)
        
        # compute the energy score
        scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy() 
        self.energy_scores.extend(scores)



