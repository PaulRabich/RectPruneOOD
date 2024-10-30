from typing import Any
from torch.nn import Module
from rect_prune_ood.lib.models.resnet import resnet50, ResNet50_Weights

class RectificaionOODModule(Module):
    def __init__(self, rectification:Module) -> None:
        super().__init__()
        self.rectification:Module = rectification
        
        self.resnet:Module = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.eval()


    def forward(self, x:Any):
        # get the output of the preultimate layer
        preulitmate = self.resnet(x)

        preulitmate_with_rect = self.rectification(preulitmate)

        # apply last layer from resnet
        out = self.resnet.fc(preulitmate_with_rect)

        return out


