# ============================================
# DATASET CLASS - WITH SLICE SKIPPING & BLACK SLICE FILTERING
# ============================================

import torch
from torch.utils.data import Dataset
import nibabel as nib
import torchvision.transforms.v2 as torchvision
from pathlib import Path
import numpy as np
import random
import torchvision.transforms.functional as TF


class PETSliceDataset(Dataset):
    def __init__(self, exam_dirs: list, target_size=(256, 256), transform=False,
            ):
        """
        Args:
            exam_dirs: List of exam directories
            target_size: Target image size
            transform: Defines if Augmentation is done
         """
        self.resize = torchvision.Resize(target_size)
        self.target_size = target_size
        self.transform = transform

        self.volume_cache = {}
        # Store slices with their metadata
        self.slices = []

        print(f"📊 Loading PET Dataset")

        for exam_dir in exam_dirs:
            pet_dir = exam_dir / "PET"
            pet_files = list(pet_dir.glob("*_trimmed.nii.gz*"))
            if not pet_files:
                continue

            pet_path = pet_files[0]
            pet_img = nib.load(str(pet_path))
            pet_data = pet_img.get_fdata().astype(np.float32)

            self.volume_cache[str(pet_path)] = pet_data

            num_slices = pet_img.shape[2]

            for z in range(num_slices):

                self.slices.append({
                    "pet_path": str(pet_path),

                })
        print(f"Loaded {len(self.slices)} paired slices")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        path, z, max_intensity = self.slices[idx]

        img = nib.load(path)
        pet_slice = img.get_fdata()[:, :, z]

        # Crop to ROI
        mask = pet_slice > 0.01
        if mask.any():
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Make it SQUARE by extending the shorter side
            width = cmax - cmin
            height = rmax - rmin

            if width > height:
                # Add padding to top and bottom
                diff = width - height
                rmin = max(0, rmin - diff // 2)
                rmax = min(pet_slice.shape[0], rmax + diff - diff // 2)
            else:
                # Add padding to left and right
                diff = height - width
                cmin = max(0, cmin - diff // 2)
                cmax = min(pet_slice.shape[1], cmax + diff - diff // 2)

            # Add 2px padding
            rmin = max(0, rmin - 2)
            rmax = min(pet_slice.shape[0], rmax + 2)
            cmin = max(0, cmin - 2)
            cmax = min(pet_slice.shape[1], cmax + 2)

            pet_slice = pet_slice[rmin:rmax, cmin:cmax]

        # Convert to tensor and add channel dimension
        pet_tensor = torch.tensor(pet_slice, dtype=torch.float32).unsqueeze(0)

        # Resize to target size
        if pet_tensor.shape[1:] != self.target_size:
            pet_tensor = self.resize(pet_tensor)

        if self.transform:

            if random.random() < 0.5:
                pet_tensor = TF.hflip(pet_tensor)

            angle = random.uniform(-10, 10)

            pet_tensor = TF.rotate(pet_tensor, angle)

            tx = random.uniform(-0.02, 0.02) * pet_tensor.shape[-1]
            ty = random.uniform(-0.02, 0.02) * pet_tensor.shape[-2]

            pet_tensor = TF.affine(
                pet_tensor,
                angle=0,
                translate=(int(tx), int(ty)),
                scale=1.0,
                shear=0
            )
        return pet_tensor
