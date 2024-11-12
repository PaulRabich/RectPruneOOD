from typing import Literal

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms

in_datasets = Literal['cifar10', 'imagenet']
out_datasets = Literal['inat', 'dtd', 'sun', 'places365']

imagesize = 32

transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

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

    id_path = f'./datasets/id_datasets/{in_dataset_name}'

    val_loader = None
    num_classes = 0
    
    if in_dataset_name == 'cifar10':
        valset = CIFAR10(root=id_path, train=False, download=True, transform=transform_test)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

        num_classes = 10

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

    ood_path =  f'./datasets/ood_datasets/{out_dataset_name}'

    val_loader = None

    if out_dataset_name == 'dtd':
        val_loader = DataLoader(ImageFolder(root=ood_path + '/images', transform=transform_test), batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return val_loader
