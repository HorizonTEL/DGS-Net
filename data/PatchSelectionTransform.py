import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import math

def spectral_entropy(patch):
    if patch.ndim == 3:
        patch_gray = np.dot(patch[..., :3], [0.299, 0.587, 0.114])
    else:
        patch_gray = patch
    if patch_gray.max() > 1:
        patch_gray = patch_gray / 255.0
    f = np.fft.fft2(patch_gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    power = magnitude ** 2
    power_flat = power.flatten()
    p_norm = power_flat / (np.sum(power_flat) + 1e-12)
    ent = -np.sum(p_norm * np.log2(p_norm + 1e-12))
    return ent


class PatchSelectionTransform:
    def __init__(self, patch_size=32, num_patches=49):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        image_np = np.array(img)
        height, width = image_np.shape[:2]
        height, width = height - height % self.patch_size, width - width % self.patch_size
        image_padded = image_np[:height, :width]

        patches = []
        for i in range(0, height, self.patch_size):
            for j in range(0, width, self.patch_size):
                patch = image_padded[i:i + self.patch_size, j:j + self.patch_size]
                patches.append((patch, (i, j)))

        entropies = []
        for patch, _ in patches:
            ent = spectral_entropy(patch)
            entropies.append(ent)

        entropies = np.array(entropies)
        total_patches = len(patches)
        half_low = self.num_patches // 2
        half_high = self.num_patches - half_low

        sorted_indices = np.argsort(entropies)
        low_indices = sorted_indices[:half_low]
        high_indices = sorted_indices[-half_high:]
        selected_indices = np.concatenate([low_indices, high_indices])

        selected_patches = [patches[i] for i in selected_indices]

        random.shuffle(selected_patches)

        stitched_image = np.zeros((224, 224, 3), dtype=np.uint8)
        grid_size = int(math.sqrt(self.num_patches))
        for idx, (patch, _) in enumerate(selected_patches):
            row = (idx // grid_size) * self.patch_size
            col = (idx % grid_size) * self.patch_size
            stitched_image[row:row + self.patch_size, col:col + self.patch_size] = patch

        return Image.fromarray(stitched_image)
