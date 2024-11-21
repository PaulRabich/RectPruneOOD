import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MetricCollection

from rect_prune_ood.lib.eval.scores import EnergyScore, MSPScore


class Pipeline(LightningModule):
    def __init__(self, featureExtractor, rect_method, prune_method) -> None:
        super().__init__()
        self.featureExtractor = featureExtractor
        self.rect_method = rect_method
        self.prune_method = prune_method
        self.metrics = MetricCollection({"energy_score": EnergyScore(), 
                                         "msp_score": MSPScore()})
        self.prediction_list = [] 
        self.results = {}

    def forward(self, x) -> torch.Tensor:
        out = self.featureExtractor(x)
        rectified_out = self.rect_method(out)
        pruned_out = self.prune_method(rectified_out)

        return pruned_out
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError("The aim of this class is only to evaluate the performance of the model")
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("The aim of this class is only to evaluate the performance of the model")
    
    def test_step(self, batch, batch_idx):
        image, label = batch

        images = image.cuda()

        inputs = images.float()
        with torch.no_grad():
            logits = self(inputs)
            outputs = F.softmax(logits, dim=1)

            prediction = torch.argmax(outputs, dim=1)
            confidence = torch.max(outputs, dim=1)[0]

            self.prediction_list.append((prediction, label, confidence))
            
        # compute the metrics
        self.metrics.update(logits)

    def on_test_epoch_end(self):
        self.results = {}
        metrics = self.metrics.compute()

        prediction, label, confidence = zip(*self.prediction_list)
        prediction = torch.cat(prediction).cpu().numpy()
        label = torch.cat(label).cpu().numpy()
        confidence = torch.cat(confidence).cpu().numpy()

        self.results["prediction"] = prediction
        self.results["label"] = label
        self.results["confidence"] = confidence
        

        self.metrics.reset()
        self.prediction_list = []

        self.results = metrics
        




