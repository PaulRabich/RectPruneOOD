from torchvision.models import resnet50



def get_resnet50():
    # Load pretrained resnet50 model
    resnet = resnet50(pretrained=True)
    return resnet