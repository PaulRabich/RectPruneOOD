import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision.models import resnet50

# file that provides methods to obtain backbones for our post hoc method

# pretrained networks weights
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def get_resnet50(pretrained=True):
    # Load and return pretrained resnet50 model
    resnet = resnet50(weights="IMAGENET1K_V1")
    if pretrained:
        resnet.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))

    return resnet


def get_mobilenetV2(pretrained=True):
    # Load and return pretrained mobilenetV2 model
    mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")

    return mobilenet
