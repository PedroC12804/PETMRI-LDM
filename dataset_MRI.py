import torch
from torch.utils.data import Dataset
import nibabel as nib
import torchvision.transforms.v2 as torchvision
from pathlib import Path
import numpy as np


class PETMRISliceDataset(Dataset):

    def __init__(
        self,
        exam_dirs,
        target_size=(256, 256),

    ):
        self.volume_cache = {}
        self.target_size = target_size
        self.resize = torchvision.Resize(target_size)

        self.samples = []

        print("\nLoading PET-MRI paired dataset...")

        for exam_dir in exam_dirs:

            pet_dir = exam_dir / "PET"
            mri_dir = exam_dir / "MRI_trimmed"

            pet_files = list(pet_dir.glob("*_trimmed.nii.gz*"))
            mri_files = list(mri_dir.glob("*.nii.gz*"))

            if len(pet_files) == 0 or len(mri_files) == 0:
                continue

            pet_path = pet_files[0]
            mri_path = mri_files[0]

            # Load once only for metadata
            pet_img = nib.load(str(pet_path))
            mri_img = nib.load(str(mri_path))

            # print("\nExam:", exam_dir.name)
            # print("PET shape:", pet_img.shape, type(pet_img.shape))
            # print("MRI shape:", mri_img.shape, type(mri_img.shape))
            # print("PET ndim:", len(pet_img.shape))
            # print("MRI ndim:", len(mri_img.shape))

            pet_data = pet_img.get_fdata().astype(np.float32)
            mri_data = mri_img.get_fdata().astype(np.float32)

            self.volume_cache[str(pet_path)] = pet_data
            self.volume_cache[str(mri_path)] = mri_data

            # Safety check
            assert pet_img.shape == mri_img.shape, \
                f"Shape mismatch in {exam_dir.name}"

            num_slices = pet_img.shape[2]

            for z in range(num_slices):

                self.samples.append({
                    "pet_path": str(pet_path),
                    "mri_path": str(mri_path),
                    "z": z,
                })

        print(f"Loaded {len(self.samples)} paired slices")

    def __len__(self):
        return len(self.samples)

    def crop_from_pet_mask(self, pet_slice, mri_slice):

        mask = pet_slice > 0.01

        if not mask.any():
            return pet_slice, mri_slice

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # make square
        width = cmax - cmin
        height = rmax - rmin

        if width > height:
            diff = width - height
            rmin = max(0, rmin - diff // 2)
            rmax = min(pet_slice.shape[0], rmax + diff - diff // 2)

        else:
            diff = height - width
            cmin = max(0, cmin - diff // 2)
            cmax = min(pet_slice.shape[1], cmax + diff - diff // 2)

        # padding
        pad = 2

        rmin = max(0, rmin - pad)
        rmax = min(pet_slice.shape[0], rmax + pad)

        cmin = max(0, cmin - pad)
        cmax = min(pet_slice.shape[1], cmax + pad)

        pet_slice = pet_slice[rmin:rmax, cmin:cmax]
        mri_slice = mri_slice[rmin:rmax, cmin:cmax]

        return pet_slice, mri_slice

    def __getitem__(self, idx):

        sample = self.samples[idx]

        pet_volume = self.volume_cache[sample["pet_path"]]
        mri_volume = self.volume_cache[sample["mri_path"]]


        z = sample["z"]

        pet_slice = pet_volume[:, :, z]
        mri_slice = mri_volume[:, :, z]

        # Shared crop
        pet_slice, mri_slice = self.crop_from_pet_mask(
            pet_slice,
            mri_slice
        )

        # To tensor
        pet_tensor = torch.tensor(
            pet_slice,
            dtype=torch.float32
        ).unsqueeze(0)

        mri_tensor = torch.tensor(
            mri_slice,
            dtype=torch.float32
        ).unsqueeze(0)

        # Resize
        if pet_tensor.shape[1:] != self.target_size:
            pet_tensor = self.resize(pet_tensor)

        if mri_tensor.shape[1:] != self.target_size:
            mri_tensor = self.resize(mri_tensor)

        return pet_tensor, mri_tensor