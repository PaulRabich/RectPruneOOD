from torch.nn import Module
from typing import Any, Optional
import torch


class FeatureExtractor(Module):
    def __init__(self, backbone: Module, layer: str, fc_name="fc"):
        super().__init__()
        self.backbone = backbone
        self.layer = layer
        self.feature: Optional[torch.Tensor] = None
        self.fc_layer:Module = self.backbone.__dict__["_modules"][fc_name]

        # Check if the layer is present in the backbone
        if self.backbone.__dict__["_modules"].get(self.layer) is None:
            raise ValueError(f"Layer {self.layer} not found in the backbone")
        
        network_layer = self.backbone.__dict__["_modules"][self.layer]
        

        # Add hook to the network layer and remove all other hooks
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()

        # Register hook to the network layer
        if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(self._forward_hook)
                )
        else:
            self.backbone.hook_handles.append(
                network_layer.register_forward_hook(self._forward_hook)
            )



    def _forward_hook(self, module: Module, input: Any, output: Any):
        #note: other forward hook methods use a dictionary to save the output... this somehow works
        self.feature = output


    def forward(self, x: Any) -> torch.Tensor:
        _ = self.backbone(x)
        return self.feature
    
    def get_fc_layer(self):
        return self.fc_layer
    



if __name__ == "__main__":
    from torchvision.models import resnet50

    resnet = resnet50()
    feature_extractor = FeatureExtractor(resnet, "avgpool")
    x = torch.randn(1, 3, 224, 224)
    print(feature_extractor(x).shape)