from torchvision.models import resnet50, ResNet50_Weights


def get_resnet50():
    # Load pretrained resnet50 model
    resnet = resnet50(weights="DEFAULT")
    return resnet