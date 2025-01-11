import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MetricCollection
import numpy as np

from rect_prune_ood.lib.methods.layer_scaler import LayerScaler
from rect_prune_ood.lib.eval.scores import EnergyScore, MSPScore, EntropyScore


class Pipeline(LightningModule):
    def __init__(self, featureExtractor, rect_method, prune_method, caputure_activations=False, mask_args:dict =None) -> None:
        super().__init__()                                                                      #mask_args -> {mask_indices:[int], type:str, num_dim:int}
        self.featureExtractor = featureExtractor
        self.rect_method = rect_method
        self.prune_method = prune_method
        self.metrics = MetricCollection({"energy_score": EnergyScore(), 
                                         "msp_score": MSPScore(),
                                         "entropy_score": EntropyScore()})
        self.prediction_list = [] 
        self.results = {}
        
        self.layer_scaler = None
        if mask_args != None: # a mask was given
            self.layer_scaler = LayerScaler(mask_args["mask_type"], mask_args["mask_indices"], mask_args["num_dim"])


        self.capture_activations = caputure_activations
        if self.capture_activations:
            self.activation_list = []
            self.activations: np.ndarray = None

    def forward(self, x) -> torch.Tensor:
        out = self.featureExtractor(x)

        if self.capture_activations:
            self.activation_list.append(out.squeeze())

        rectified_out = self.rect_method(out)
        pruned_out = self.prune_method(rectified_out)

        if self.layer_scaler != None: #we have a layer scaler
            out = self.layer_scaler(out)

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
        # Compute Metrics
        self.results = {}
        self.results = self.metrics.compute()

        self.results["energy_score"] = self.results["energy_score"].cpu().numpy()
        self.results["msp_score"] = self.results["msp_score"].cpu().numpy()
        self.results["entropy_score"] = self.results["entropy_score"].cpu().numpy()

        prediction, label, confidence = zip(*self.prediction_list)
        prediction = torch.cat(prediction).cpu().numpy()
        label = torch.cat(label).cpu().numpy()
        confidence = torch.cat(confidence).cpu().numpy()

        self.results["prediction"] = prediction
        self.results["label"] = label
        self.results["confidence"] = confidence

        self.metrics.reset()
        self.prediction_list = []

        # Save activations
        if self.capture_activations:
            self.activations = torch.cat(self.activation_list).cpu().numpy()
            self.activation_list = []
