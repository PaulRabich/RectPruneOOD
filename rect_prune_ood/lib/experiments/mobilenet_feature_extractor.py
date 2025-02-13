from torch.nn import Module
from torch import nn
from typing import Any, Optional
import torch


class MobileNetFeatureExtractor(Module):
    def __init__(self, backbone: Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.fc_layer:Module = self.backbone.classifier
   
        
    def forward(self, x: Any) -> torch.Tensor:
        x = self.backbone.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x
    
    def get_fc_layer(self) -> Module:
        return self.fc_layer
