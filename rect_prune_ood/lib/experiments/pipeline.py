from lightning import LightningModule
from torchmetrics import MetricCollection




class Pipeline(LightningModule):
    def __init__(self, featureExtractor, rect_method, prune_method):
        super().__init__()
        self.featureExtractor = featureExtractor
        self.rect_method = rect_method
        self.prune_method = prune_method
        #self.metrics = 



    def forward(self, x):
        out = self.featureExtractor(x)
        rectified_out = self.rect_method(out)
        pruned_out = self.prune_method(rectified_out)

        final_out = self.featureExtractor.get_fc_layer()(pruned_out)
        return final_out
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Only Evaluation is supported")
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Only Evaluation is supported")
    
    def test_step(self, batch, batch_idx):
        out = self(batch)



