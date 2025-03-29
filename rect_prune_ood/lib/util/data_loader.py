import os
from typing import Literal

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder

id_datasets = Literal[
	#'cifar10', TODO
	'imagenet'
]
ood_datasets = Literal[
	'inat', 'dtd', 'sun', 'places', 'ninco', 'ssb_hard', 'openimage-o'
]

perturbed_datasets = Literal['imagenet_perturbed', 'imagenet_c']

imagesize = 32

num_workers = 8

transform_test = transforms.Compose(
	[
		transforms.Resize((imagesize, imagesize)),
		transforms.CenterCrop(imagesize),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	]
)

transform_test_largescale = transforms.Compose(
	[
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	]
)

ood_base_path = os.path.abspath('/fastdata/vilab17/out_distribution')
imagenet_base_path = os.path.abspath('/fastdata/vilab17/imagenet')

paths = {
	'imagenet_val': os.path.join(imagenet_base_path, 'val'),
	'imagenet_train': os.path.join(imagenet_base_path, 'train'),
	'imagenet_perturbed': os.path.abspath('/fastdata/vilab17/imagenet_perturbed'),
	'imagenet_c': os.path.abspath('/fastdata/vilab17/imagenet_c'),
	'dtd': os.path.join(ood_base_path, 'dtd', 'images'),
	'places': os.path.join(ood_base_path, 'Places'),
	'inat': os.path.join(ood_base_path, 'iNaturalist'),
	'sun': os.path.join(ood_base_path, 'SUN'),
	'ninco': os.path.join(ood_base_path, 'NINCO', 'NINCO_OOD_classes'),
	'ssb_hard': os.path.join(ood_base_path, 'ssb_hard'),
	'openimage-o': os.path.join(ood_base_path, 'openimage-o'),
}


def get_data_loader(dataset: str, batch_size: int = 100, shuffle: bool = False):
	"""
	Loads a dataset and returns a DataLoader for testing.

	Parameters:
	- dataset (str): Name of the dataset ('cifar10', 'imagenet', 'inat', 'dtd', 'sun', 'places', or 'imagenet_perturbed').
	- batch_size (int, optional): Number of samples per batch. Default is 100.
	- shuffle (bool, optional): Whether to shuffle the dataset. Default is True.

	Returns:
	- val_loader (DataLoader or None): DataLoader for the specified dataset if available; None otherwise.
	- num_classes (int): Number of classes in the dataset (10 for CIFAR-10, 1000 for ImageNet). Defaults to 0 if dataset name is not recognized.
	"""

	val_loader = None
	num_classes = 0

	if dataset in ['cifar10', 'imagenet']:
		val_loader, num_classes = _get_id_data_loader(dataset, batch_size, shuffle)

	elif dataset in [
		'inat',
		'dtd',
		'sun',
		'places',
		'ninco',
		'ssb_hard',
		'openimage-o',
	]:
		val_loader = _get_ood_data_loader(
			dataset, batch_size=batch_size, shuffle=shuffle
		)

	elif dataset in ['imagenet_perturbed', 'imagenet_c']:
		val_loader = _get_perturbation_data_loader(
			dataset, batch_size=batch_size, shuffle=shuffle
		)

	return val_loader, num_classes


def _get_id_data_loader(
	id_dataset: id_datasets, batch_size: int = 100, shuffle: bool = False
):
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
		raise NotImplementedError('Cifar10 not implemented, Check Pertubation logic')
		valset = CIFAR10(
			root=paths['cifar10'], train=False, download=True, transform=transform_test
		)
		val_loader = DataLoader(
			valset, batch_size=batch_size, shuffle=shuffle, num_workers=2
		)

		num_classes = 10

	elif id_dataset == 'imagenet':
		imagenet = ImageFolder(
			root=paths['imagenet_val'], transform=transform_test_largescale
		)

		val_loader = DataLoader(
			imagenet, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
		)

		num_classes = 1000

	return val_loader, num_classes


def _get_ood_data_loader(
	ood_dataset: str, batch_size: int = 100, shuffle: bool = False
):
	"""
	Loads an out-of-distribution (OOD) dataset and returns a DataLoader for testing.

	Parameters:
	- ood_dataset (str): Name of the out-of-distribution dataset ('inat', 'dtd', 'sun', or 'places365').
	- batch_size (int, optional): Number of samples per batch. Default is 100.
	- shuffle (bool, optional): Whether to shuffle the dataset. Default is True.

	Returns:
	- val_loader (DataLoader or None): DataLoader for the specified OOD dataset if available; None otherwise.
	"""

	image_folder = ImageFolder(
		root=paths[ood_dataset], transform=transform_test_largescale
	)
	val_loader = DataLoader(
		image_folder, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
	)

	return val_loader


def _get_perturbation_data_loader(
	perturbed_dataset: str = 'imagenet_perturbed',
	batch_size: int = 100,
	shuffle: bool = False,
):
	"""
	Loads a perturbation dataset and returns a DataLoader for testing.

	Parameters:
	- perturbed_dataset (str): Name of the dataset ('imagenet_perturbed' only).
	- batch_size (int, optional): Number of samples per batch. Default is 100.
	- shuffle (bool, optional): Whether to shuffle the dataset. Default is True.

	Returns:
	- val_loader (DataLoader or None): DataLoader for the perturbation dataset if available; None otherwise.
	"""

	dataset = ImageFolder(root=paths[perturbed_dataset], transform=transform_test_largescale)
	val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
	return val_loader

def get_imagenet_c_subset_loader(subset: str, batch_size: int = 100, shuffle: bool = False):
    """
        Loads a subset of the ImageNet-C dataset and returns a DataLoader for testing.

        Parameters:
        - subset (str): Name of the ImageNet-C subset ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate').
        - batch_size (int, optional): Number of samples per batch. Default is 100.
        - shuffle (bool, optional): Whether to shuffle the dataset. Default is True.

        Returns:
        - val_loader (DataLoader or None): DataLoader for the specified ImageNet-C subset if available; None otherwise.
    """

    dataset = ImageFolder(root=os.path.join(paths['imagenet_c'], subset), transform=transform_test_largescale)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return val_loader