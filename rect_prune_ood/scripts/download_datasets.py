import os
import zipfile

import gdown

from rect_prune_ood.lib.util.paths import get_top_level_path

datasets_dir = os.path.join(
    get_top_level_path(), 'datasets'
)
id_datasets_dir = os.path.join(
    datasets_dir, 'id_datasets'
)
perturbed_datasets_dir = os.path.join(
    datasets_dir, 'perturbed_datasets'
)
ood_datasets_dir = os.path.join(
    datasets_dir, 'ood_datasets'
)

# Mapping datasets to their respective directories
dataset_dir_mapping = {
    'imagenet_1k': os.path.join(id_datasets_dir, 'imagenet'),
    'imagenet_c': os.path.join(perturbed_datasets_dir, 'imagenet-c'),
    'openimage_o': os.path.join(ood_datasets_dir, 'openimage-o'),
    'inaturalist': os.path.join(ood_datasets_dir, 'iNaturalist'),
    'sun': os.path.join(ood_datasets_dir, 'SUN'),
    'places': os.path.join(ood_datasets_dir, 'Places'),
    'texture': os.path.join(ood_datasets_dir, 'dtd'),
    'ninco': os.path.join(ood_datasets_dir, 'NINCO'),
    'ssb_hard': os.path.join(ood_datasets_dir, 'ssb-hard'),
}

# Google Drive download IDs for each dataset
download_id_dict = {
    'imagenet_1k': '1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj',
    'imagenet_c': '1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt',
    'openimage_o': '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
    'inaturalist': '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj',
    'sun': '1ISK0STxWzWmg-_uUr4RQ8GSLFW7TZiKp',
    'places': '1fZ8TbPC4JGqUCm-VtvrmkYxqRNp2PoB3',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
    'ninco': '1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6',
    'ssb_hard': '1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE',
}


def require_download(filename, path):
    """Check if a dataset needs to be downloaded."""
    for item in os.listdir(path):
        if item.startswith(filename) or filename.startswith(item):
            return False
    print(f"{filename} needs download.")
    return True


def download_dataset(dataset):
    """Download and extract a dataset if not already present."""
    if dataset not in dataset_dir_mapping:
        print(f"Invalid dataset detected: {dataset}")
        return

    store_path = dataset_dir_mapping[dataset]
    os.makedirs(store_path, exist_ok=True)

    if require_download(dataset, store_path):
        print(f"Downloading {dataset} to {store_path}...")

        gdown.download(id=download_id_dict[dataset], output=store_path, quiet=False)

        file_path = os.path.join(store_path, f"{dataset}.zip")
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(store_path)
        os.remove(file_path)


if __name__ == '__main__':
    datasets_to_download = [
        'imagenet_1k', 'imagenet_c', 'openimage_o', 'inaturalist',
        'sun', 'places', 'texture', 'ninco', 'ssb_hard'
    ]

    for dataset in datasets_to_download:
        download_dataset(dataset)
