import re
import torch
from torch.utils.data import Dataset
import nibabel as nib
import torchvision.transforms.v2 as torchvision
import numpy as np
import torchvision.transforms.functional as TF
import random


def reconcile_shape(data, ref_shape, max_diff=4):
    """
    Center-crop or zero-pad `data` so its shape matches `ref_shape`,
    as long as every axis differs by at most `max_diff` voxels.
    """

    diffs = [abs(d - r) for d, r in zip(data.shape, ref_shape)]

    if any(d > max_diff for d in diffs):
        raise ValueError(
            f"Shape mismatch too large to reconcile: {data.shape} vs {ref_shape}"
        )

    if data.shape == ref_shape:
        return data

    out = data
    # Crop any axis that's larger than the reference, centered.
    slices = []
    for size, ref_size in zip(out.shape, ref_shape):
        if size > ref_size:
            start = (size - ref_size) // 2
            slices.append(slice(start, start + ref_size))
        else:
            slices.append(slice(0, size))
    out = out[tuple(slices)]

    # Pad any axis that's still smaller than the reference, centered.
    if out.shape != ref_shape:
        pad_widths = []
        for size, ref_size in zip(out.shape, ref_shape):
            total_pad = ref_size - size
            before = total_pad // 2
            after = total_pad - before
            pad_widths.append((before, after))
        out = np.pad(out, pad_widths, mode="constant", constant_values=0)

    return out


class SliceDataset(Dataset):

    def __init__(
        self,
        exam_dirs,
        target_size=(256, 256),
        transform=False,
        shape_tolerance=4,
        dose_levels = None
    ):
        self.volume_cache = {}
        self.target_size = target_size
        self.resize = torchvision.Resize(target_size)
        self.transform = transform
        self.shape_tolerance = shape_tolerance
        self.dose_levels = set(dose_levels) if dose_levels is not None else None


        self.samples = []

        dose_msg = "all dose levels" if self.dose_levels is None else f"dose levels {sorted(self.dose_levels)}"


        print(f"\nLoading PET-MRI-LDPET paired dataset... {dose_msg}")

        for exam_dir in exam_dirs:

            pet_dir = exam_dir / "PET"
            mri_dir = exam_dir / "MRI_trimmed"
            ld_dir = exam_dir / "PET_LD"
            pet_files = list(pet_dir.glob("*_trimmed.nii.gz*"))
            mri_files = list(mri_dir.glob("*.nii.gz*"))
            ld_files = list(ld_dir.glob("*.nii.gz*"))

            if len(pet_files) == 0 or len(mri_files) == 0 or len(ld_files) == 0:
                continue

            pet_path = pet_files[0]
            mri_path = mri_files[0]

            pet_img = nib.as_closest_canonical(nib.load(str(pet_path)))
            mri_img = nib.as_closest_canonical(nib.load(str(mri_path)))

            pet_data = pet_img.get_fdata().astype(np.float32)
            mri_data = mri_img.get_fdata().astype(np.float32)

            if pet_data.shape != mri_data.shape:
                try:
                    mri_data = reconcile_shape(
                        mri_data, pet_data.shape, max_diff=self.shape_tolerance
                    )
                    print(
                        f"NOTE: {exam_dir.name} MRI shape "
                        f"{mri_img.shape} reconciled to PET shape {pet_img.shape}"
                    )
                except ValueError as e:
                    print(f"WARNING: skipping {exam_dir.name}, MRI/PET {e}")
                    continue

            self.volume_cache[str(pet_path)] = pet_data
            self.volume_cache[str(mri_path)] = mri_data

            # ------------------------------------------------------
            # Parse dose level from each LD-PET filename and cache
            # ------------------------------------------------------

            dose_pattern = re.compile(r"dose(\d+)")


            ld_entries = []

            for ld_path in ld_files:
                match = dose_pattern.search(ld_path.name)
                if match is None:
                    print(f"WARNING: could not parse dose from {ld_path.name}, skipping")
                    continue

                dose_level = int(match.group(1))

                if self.dose_levels is not None and dose_level not in self.dose_levels:
                    continue
                    
                ld_img = nib.as_closest_canonical(nib.load(str(ld_path)))
                ld_data = ld_img.get_fdata().astype(np.float32)

                if ld_data.shape != pet_data.shape:
                    try:
                        ld_data = reconcile_shape(
                            ld_data, pet_data.shape, max_diff=self.shape_tolerance
                        )
                        print(
                            f"NOTE: {exam_dir.name} LD-PET ({ld_path.name}) shape "
                            f"{ld_img.shape} reconciled to PET shape {pet_img.shape}"
                        )
                    except ValueError as e:
                        print(
                            f"WARNING: skipping {ld_path.name} in {exam_dir.name}, {e}"
                        )
                        continue

                self.volume_cache[str(ld_path)] = ld_data

                ld_entries.append((str(ld_path), dose_level))

            if len(ld_entries) == 0:
                continue

            num_slices = pet_img.shape[2]

            # ------------------------------------------------------
            # Flatten: one sample per (slice, dose_level) combination
            # ------------------------------------------------------
            for z in range(num_slices):
                for ld_path_str, dose_level in ld_entries:
                    self.samples.append({
                        "pet_path": str(pet_path),
                        "mri_path": str(mri_path),
                        "ld_path": ld_path_str,
                        "dose_level": dose_level,
                        "z": z,
                    })

        print(f"Loaded {len(self.samples)} paired slices (patient x slice x dose)")

    def __len__(self):
        return len(self.samples)

    def crop_from_pet_mask(self, pet_slice, mri_slice, ld_slice):
        """
        Crop region is computed from the full-dose PET mask only,
        then applied identically to MRI and LD-PET, so all three
        stay in exact spatial correspondence.
        """

        mask = pet_slice > 0.01

        if not mask.any():
            return pet_slice, mri_slice, ld_slice

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

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

        pad = 2
        rmin = max(0, rmin - pad)
        rmax = min(pet_slice.shape[0], rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(pet_slice.shape[1], cmax + pad)

        pet_slice = pet_slice[rmin:rmax, cmin:cmax]
        mri_slice = mri_slice[rmin:rmax, cmin:cmax]
        ld_slice = ld_slice[rmin:rmax, cmin:cmax]

        return pet_slice, mri_slice, ld_slice

    def __getitem__(self, idx):

        sample = self.samples[idx]

        pet_volume = self.volume_cache[sample["pet_path"]]
        mri_volume = self.volume_cache[sample["mri_path"]]
        ld_volume = self.volume_cache[sample["ld_path"]]

        z = sample["z"]
        dose_level = sample["dose_level"]

        pet_slice = pet_volume[:, :, z]
        mri_slice = mri_volume[:, :, z]
        ld_slice = ld_volume[:, :, z]

        # Shared crop, same region applied to all three
        pet_slice, mri_slice, ld_slice = self.crop_from_pet_mask(
            pet_slice, mri_slice, ld_slice
        )

        pet_tensor = torch.tensor(pet_slice, dtype=torch.float32).unsqueeze(0)
        mri_tensor = torch.tensor(mri_slice, dtype=torch.float32).unsqueeze(0)
        ld_tensor = torch.tensor(ld_slice, dtype=torch.float32).unsqueeze(0)

        if pet_tensor.shape[1:] != self.target_size:
            pet_tensor = self.resize(pet_tensor)

        if mri_tensor.shape[1:] != self.target_size:
            mri_tensor = self.resize(mri_tensor)

        if ld_tensor.shape[1:] != self.target_size:
            ld_tensor = self.resize(ld_tensor)

        if self.transform:

            # Generate augmentation parameters once, apply identically
            # to all three tensors so spatial correspondence is preserved.
            do_flip = random.random() < 0.5
            angle = random.uniform(-10, 10)
            tx = random.uniform(-0.02, 0.02) * pet_tensor.shape[-1]
            ty = random.uniform(-0.02, 0.02) * pet_tensor.shape[-2]

            tensors = [pet_tensor, mri_tensor, ld_tensor]

            for i in range(len(tensors)):
                t = tensors[i]

                if do_flip:
                    t = TF.hflip(t)

                t = TF.rotate(t, angle)

                t = TF.affine(
                    t,
                    angle=0,
                    translate=(int(tx), int(ty)),
                    scale=1.0,
                    shear=0
                )

                tensors[i] = t

            pet_tensor, mri_tensor, ld_tensor = tensors

        dose_tensor = torch.tensor(float(dose_level), dtype=torch.float32)

        return pet_tensor, mri_tensor, ld_tensor, dose_tensor