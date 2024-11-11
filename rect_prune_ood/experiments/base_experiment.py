import sys
sys.path.append("./")

from torchvision.models import resnet50

resnet = resnet50()

print(resnet.__dict__)
