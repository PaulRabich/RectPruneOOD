from torchvision.models import resnet50, ResNet50_Weights
import torch.utils.model_zoo as model_zoo
from rect_prune_ood.lib.models.resnet_cifar_react import resnet50_cifar

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def get_resnet50(pretrained=True):
    
    # resnet = resnet50_cifar(num_classes=10, method="energy")
    # Load pretrained resnet50 model
    resnet = resnet50(weights="IMAGENET1K_V1")
    #if pretrained:
    #    resnet.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return resnet

