from typing import Literal
import os

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms
from torchvision.datasets import ImageNet

from rect_prune_ood.lib.util.paths import get_top_level_path

id_datasets = Literal[
    #'cifar10', TODO
    'imagenet'
]
ood_datasets = Literal[
    'inat', 
    'dtd', 
    'sun', 
    'places'
]

imagesize = 32

num_workers = 8

transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

ood_base_path = os.path.abspath('/fastdata/vilab17/out_distribution')
imagenet_base_path = os.path.abspath('/data/vimb06/imagenet')

paths = {
    'cifar10': os.path.join('TODO'), # TODO
    'imagenet_val': os.path.join(imagenet_base_path, 'val'),
    'imagenet_train': os.path.join(imagenet_base_path, 'train'),
    'dtd': os.path.join(ood_base_path, 'dtd', 'images'),
    'places': os.path.join(ood_base_path, 'Places'),
    'inat': os.path.join(ood_base_path, 'iNaturalist'),
    'sun': os.path.join(ood_base_path, 'SUN')
}

def get_id_data_loader(id_dataset : id_datasets, batch_size : int = 100, shuffle : bool = False):
    """
        Loads an in-distribution (ID) dataset and returns a DataLoader for testing.

        Parameters:
        - id_dataset (in_datasets): Name of the in-distribution dataset (must be either 'cifar10' or 'imagenet').
        - batch_size (int, optional): Number of samples per batch. Default is 100.
        - shuffle (bool, optional): Whether to shuffle the dataset. Default is True.

        Returns:
        - val_loader (DataLoader or None): DataLoader for the in-distribution dataset if the dataset name is recognized; None otherwise.
        - num_classes (int): Number of classes in the dataset (10 for CIFAR-10). Defaults to 0 if dataset name is not recognized.
    """

    val_loader = None
    num_classes = 0
    
    if id_dataset == 'cifar10':
        valset = CIFAR10(root=paths['cifar10'], train=False, download=True, transform=transform_test)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

        num_classes = 10

    elif id_dataset == 'imagenet':
        val_loader = DataLoader(
            ImageFolder(root=paths['imagenet_val'], transform=transform_test_largescale),
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
        num_classes = 1000

    return val_loader, num_classes


def get_ood_data_loader(ood_dataset : str, batch_size : int = 100, shuffle : bool = False):
    """
        Loads an out-of-distribution (OOD) dataset and returns a DataLoader for testing.

        Parameters:
        - ood_dataset (str): Name of the out-of-distribution dataset ('inat', 'dtd', 'sun', or 'places365').
        - batch_size (int, optional): Number of samples per batch. Default is 100.
        - shuffle (bool, optional): Whether to shuffle the dataset. Default is True.

        Returns:
        - val_loader (DataLoader or None): DataLoader for the specified OOD dataset if available; None otherwise.
    """

    ood_path = paths[ood_dataset]

    val_loader = None

    if ood_dataset == 'dtd':
        val_loader = DataLoader(
            ImageFolder(root=ood_path, transform=transform_test), 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
    elif ood_dataset == 'places':
        val_loader = DataLoader(
            ImageFolder(root=ood_path, transform=transform_test_largescale), 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
    elif ood_dataset == 'sun':
        val_loader = DataLoader(
            ImageFolder(root=ood_path, transform=transform_test), 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
    elif ood_dataset == 'inat':
        val_loader = DataLoader(
            ImageFolder(root=ood_path, transform=transform_test), 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )

    return val_loader
