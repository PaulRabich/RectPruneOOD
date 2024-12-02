from torch.utils.data import Dataset
from torch import Tensor
import torch
import numpy as np
from PIL import Image as PILImage
import skimage as sk

class PertubationDataset(Dataset):
    def __init__(self, dataset: Dataset, severity: int = 1):
        super().__init__()
        self.dataset: Dataset = dataset
        self.severity: int = severity

    def __getitem__(self, index) -> Tensor:
        original_index: int = index // 5  # find the original item from the dataset
        perturbation_index: int = index % 5  # find the perturbation index

        image, label = self.dataset[original_index]  # Assume dataset returns (image, label)
        perturbed_image = self.apply_perturbation(image, perturbation_index)

        return perturbed_image, label

    def __len__(self) -> int:
        return len(self.dataset) * 5

    def apply_perturbation(self, image: Tensor, perturbation: int) -> Tensor:
        if perturbation == 0:
            return self.brightness(image)   
        elif perturbation == 1:
            return self.gaussian_noise(image)
        elif perturbation == 2:
            return self.saturate(image)
        elif perturbation == 3:
            return self.pixelate(image)
        else:
            return image

    def brightness(self, x: Tensor) -> Tensor:
        # print("brightness")
        # print(x.dtype)
        c = [0.1, 0.2, 0.3, 0.4, 0.5][self.severity - 1]
        x = x.permute(1, 2, 0).numpy()  # Convert to HWC format and numpy array
        x = x / 255.0
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)
        x = torch.tensor(x).permute(2, 0, 1)  # Convert back to CHW format
        return (x * 255).clip(0, 255).to(torch.float32)

    def gaussian_noise(self, x: Tensor) -> Tensor:
        # print("gaussian_noise")
        c = [0.08, 0.12, 0.18, 0.26, 0.38][self.severity - 1]
        noise = torch.normal(mean=0.0, std=c, size=x.size()).to(x.dtype)
        x = x / 255.0
        x = (x + noise).clamp(0, 1) * 255
        return x.to(torch.float32)

    def saturate(self, x: Tensor) -> Tensor:
        # print("saturate")
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][self.severity - 1]
        x = x.permute(1, 2, 0).numpy()  # Convert to HWC format and numpy array
        x = x / 255.0
        x = sk.color.rgb2hsv(x)
        x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
        x = sk.color.hsv2rgb(x)
        x = torch.tensor(x).permute(2, 0, 1)  # Convert back to CHW format
        return (x * 255).clip(0, 255).to(torch.float32)

    def pixelate(self, x: Tensor) -> Tensor:
        # print("pixelate")
        c = [0.6, 0.5, 0.4, 0.3, 0.25][self.severity - 1]
        x = x.permute(1, 2, 0).numpy()  # Convert to HWC format and numpy array
        img = PILImage.fromarray(x.astype(np.uint8))
        img = img.resize((int(224 * c), int(224 * c)), PILImage.BOX)
        img = img.resize((224, 224), PILImage.BOX)
        x = torch.tensor(np.array(img)).permute(2, 0, 1)  # Convert back to CHW format
        return x.to(torch.float32)
