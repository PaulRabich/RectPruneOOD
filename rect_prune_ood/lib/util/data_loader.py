from typing import Literal
import os

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms
from torchvision.datasets import ImageNet

from rect_prune_ood.lib.util.paths import get_top_level_path

in_datasets = Literal[
    #'cifar10', TODO
    'imagenet'
]
out_datasets = Literal[
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

ood_base_path = os.path.join("fastdata", "vilab17", "out_distribution")
imagenet_base_path = os.path.join('data', 'vimb06', 'imagenet')

paths = {
    'cifar10': os.path.join('TODO'), # TODO
    'imagenet_val': os.path.join(imagenet_base_path, 'val'),
    'imagenet_train': os.path.join(imagenet_base_path, 'train'),
    'dtd': os.path.join(ood_base_path, 'dtd', 'images'),
    'places': os.path.join(ood_base_path, 'Places'),
    'inat': os.path.join(ood_base_path, 'iNaturalist'),
    'sun': os.path.join(ood_base_path, 'SUN')
}

def get_in_data_loader(in_dataset_name : in_datasets, batch_size : int = 100, shuffle : bool = True):
    """
        Loads an in-distribution (ID) dataset and returns a DataLoader for testing.

        Parameters:
        - in_dataset_name (in_datasets): Name of the in-distribution dataset (must be either 'cifar10' or 'imagenet').
        - batch_size (int, optional): Number of samples per batch. Default is 100.
        - shuffle (bool, optional): Whether to shuffle the dataset. Default is True.

        Returns:
        - val_loader (DataLoader or None): DataLoader for the in-distribution dataset if the dataset name is recognized; None otherwise.
        - num_classes (int): Number of classes in the dataset (10 for CIFAR-10). Defaults to 0 if dataset name is not recognized.
    """

    val_loader = None
    num_classes = 0
    
    if in_dataset_name == 'cifar10':
        valset = CIFAR10(root=paths['cifar10'], train=False, download=True, transform=transform_test)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

        num_classes = 10

    elif in_dataset_name == 'imagenet':
        val_loader = DataLoader(
            ImageFolder(root=paths['imagenet_val'], transform=transform_test_largescale),
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
        num_classes = 1000

    return val_loader, num_classes


def get_out_data_loader(out_dataset_name : str, batch_size : int = 100, shuffle : bool = True):
    """
        Loads an out-of-distribution (OOD) dataset and returns a DataLoader for testing.

        Parameters:
        - out_dataset_name (str): Name of the out-of-distribution dataset ('inat', 'dtd', 'sun', or 'places365').
        - batch_size (int, optional): Number of samples per batch. Default is 100.
        - shuffle (bool, optional): Whether to shuffle the dataset. Default is True.

        Returns:
        - val_loader (DataLoader or None): DataLoader for the specified OOD dataset if available; None otherwise.
    """

    ood_path = paths[out_dataset_name]

    val_loader = None

    if out_dataset_name == 'dtd':
        val_loader = DataLoader(
            ImageFolder(root=ood_path, transform=transform_test), 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
    elif out_dataset_name == 'places':
        val_loader = DataLoader(
            ImageFolder(root=ood_path, transform=transform_test_largescale), 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
    elif out_dataset_name == 'sun':
        val_loader = DataLoader(
            ImageFolder(root=ood_path, transform=transform_test), 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
    elif out_dataset_name == 'inat':
        val_loader = DataLoader(
            ImageFolder(root=ood_path, transform=transform_test), 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )

    return val_loader
