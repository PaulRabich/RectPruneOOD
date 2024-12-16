import numpy as np
import os
import torch
from PIL import Image as PILImage
from torch import Tensor
import skimage as sk
from tqdm import tqdm

from rect_prune_ood.lib.util.data_loader import paths

imagenet_val_path = paths['imagenet_val']

severity = 5

def gaussian_noise(x: Tensor) -> Tensor:
    # print("gaussian_noise")
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    noise = torch.normal(mean=0.0, std=c, size=x.size()).to(x.dtype)
    x = x / 255.0
    x = (x + noise).clamp(0, 1) * 255
    return x.to(torch.float32), 'gaussian_noise'

def brightness(x: Tensor) -> Tensor:
    # print("brightness")
    # print(x.dtype)
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    x = x.permute(1, 2, 0).numpy()  # Convert to HWC format and numpy array
    x = x / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    x = torch.tensor(x).permute(2, 0, 1)  # Convert back to CHW format
    return (x * 255).clip(0, 255).to(torch.float32), 'brightness'

def saturate(x: Tensor) -> Tensor:
    # print("saturate")
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    x = x.permute(1, 2, 0).numpy()  # Convert to HWC format and numpy array
    x = x / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    x = torch.tensor(x).permute(2, 0, 1)  # Convert back to CHW format
    return (x * 255).clip(0, 255).to(torch.float32), 'saturate'

def pixelate(x: Tensor) -> Tensor:
    # print("pixelate")
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    x = x.permute(1, 2, 0).numpy()  # Convert to HWC format and numpy array
    img = PILImage.fromarray(x.astype(np.uint8))
    img = img.resize((int(224 * c), int(224 * c)), PILImage.BOX)
    img = img.resize((224, 224), PILImage.BOX)
    x = torch.tensor(np.array(img)).permute(2, 0, 1)  # Convert back to CHW format
    return x.to(torch.float32), 'pixelate'

def apply_perturbation(image: Tensor, perturbation: int) -> Tensor:
        if perturbation == 0:
            return brightness(image)   
        elif perturbation == 1:
            return gaussian_noise(image)
        elif perturbation == 2:
            return saturate(image)
        elif perturbation == 3:
            return pixelate(image)
        else:
            return image

def perturb_imagenet():
    # loop over folders in imagenet_val_path
    for folder_name in tqdm(os.listdir(imagenet_val_path)):
        folder_path = os.path.join(imagenet_val_path, folder_name)

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            image = PILImage.open(image_path)

            if image.mode != 'RGB':
                image = image.convert('RGB')

            img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)

            for i in range(4):              
                perturbed_image, perturbation = apply_perturbation(img_tensor, i)

                perturbed_image = PILImage.fromarray(perturbed_image.permute(1, 2, 0).numpy().astype(np.uint8))

                save_perturbed_image(perturbation, folder_name, image_name, perturbed_image)

def save_perturbed_image(perturbation, folder_name, image_name, image):
    perturbed_path = os.path.abspath('/fastdata/vilab17/imagenet_perturbed')

    if not os.path.exists(perturbed_path):
        os.makedirs(perturbed_path)

    perturbed_folder = os.path.join(perturbed_path, perturbation)

    if not os.path.exists(perturbed_folder):
        os.makedirs(perturbed_folder)

    perturbed_class_folder = os.path.join(perturbed_folder, folder_name)

    if not os.path.exists(perturbed_class_folder):
        os.makedirs(perturbed_class_folder)

    image.save(os.path.join(perturbed_class_folder, image_name))
