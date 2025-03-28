from torch.nn import Module
from typing import Any, Optional
import torch


#Class to extract activations of the specified layer from resnet50
class FeatureExtractor(Module):
    def __init__(self, backbone: Module, layer: str, fc_name:str="fc") -> None:
        super().__init__()
        #init attributes
        self.backbone = backbone
        #layer to extract from
        self.layer = layer
        self.feature: Optional[torch.Tensor] = None
        #get fully connected layer
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



    """
        Note: other forward hook methods use a dictionary to save the output
    """
    #hook that saves the activations of specified layer in the feature list on forwards call
    def _forward_hook(self, module: Module, input: Any, output: Any) -> None:
        self.feature = output


    #perform forward step, feature now contains activations because of hook
    def forward(self, x: Any) -> torch.Tensor:
        _ = self.backbone(x)
        return self.feature
    
    #for pipeline later, return fully connected layer to continue after features
    def get_fc_layer(self) -> Module:
        return self.fc_layer
    



if __name__ == "__main__":
    from torchvision.models import resnet50

    resnet = resnet50()
    feature_extractor = FeatureExtractor(resnet, "avgpool")
    x = torch.randn(1, 3, 224, 224)
    print(feature_extractor(x).shape)