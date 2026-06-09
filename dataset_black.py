# ============================================
# DATASET CLASS - WITH SLICE SKIPPING & BLACK SLICE FILTERING
# ============================================

import torch
from torch.utils.data import Dataset
import nibabel as nib
import torchvision.transforms.v2 as torchvision
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class PETSliceDataset(Dataset):
    def __init__(self, exam_dirs: list, target_size=(256, 256), skip_first=0, transform=None,
                 filter_black_slices=False, black_intensity_threshold=0.01):
        """
        Args:
            exam_dirs: List of exam directories
            target_size: Target image size
            skip_first: Number of slices to skip from the beginning of each exam
            transform: Optional transforms
            filter_black_slices: Whether to filter out black/empty slices
            black_intensity_threshold: Max intensity below which slice is considered black
        """
        self.resize = torchvision.Resize(target_size)
        self.target_size = target_size
        self.filter_black_slices = filter_black_slices
        self.black_threshold = black_intensity_threshold

        # Store slices with their metadata
        self.slices = []  # (path, z, max_intensity)

        # Statistics
        self.stats = {
            'total_considered': 0,
            'black_filtered': 0,
            'kept': 0,
            'black_by_exam': {}
        }

        print(f"\n{'=' * 60}")
        print(f"📊 Loading PET Dataset")
        print(f"{'=' * 60}")
        print(f"  Filter black slices: {'ON' if filter_black_slices else 'OFF'}")
        print(f"  Black threshold: max intensity < {black_intensity_threshold}")
        print(f"  Skip first {skip_first} slices per exam")

        for exam_dir in exam_dirs:
            exam_name = exam_dir.name
            pet_dir = exam_dir / "PET"
            pet_files = list(pet_dir.glob("*_trimmed.nii.gz*"))
            if not pet_files:
                continue

            pet_path = pet_files[0]
            img = nib.load(str(pet_path))
            num_slices = img.shape[2]

            exam_black_count = 0

            # Skip first 'skip_first' slices
            for z in range(skip_first, num_slices):
                self.stats['total_considered'] += 1

                # Quick check if slice is black without full preprocessing
                slice_2d = img.get_fdata()[:, :, z]
                max_intensity = float(slice_2d.max())

                is_black = max_intensity < self.black_threshold

                if filter_black_slices and is_black:
                    self.stats['black_filtered'] += 1
                    exam_black_count += 1
                else:
                    self.slices.append((str(pet_path), z, max_intensity))
                    self.stats['kept'] += 1

            if exam_black_count > 0:
                self.stats['black_by_exam'][exam_name] = exam_black_count

        # Print summary
        self._print_summary()

        # Show warning if significant black slices found
        black_percentage = (self.stats['black_filtered'] / self.stats['total_considered'] * 100) if self.stats[
                                                                                                        'total_considered'] > 0 else 0
        if black_percentage > 1.0:
            print(f"\n⚠️  WARNING: {black_percentage:.1f}% of slices are black!")
            print(f"   These have been filtered out. Consider checking your data or skip_first parameter.")
        elif black_percentage > 0:
            print(f"\n✓ Removed {black_percentage:.1f}% black slices")

    def _print_summary(self):
        """Print loading summary"""
        print(f"\n📈 Dataset Summary:")
        print(f"  Total slices considered: {self.stats['total_considered']:,}")
        print(f"  Kept slices: {self.stats['kept']:,}")

        if self.filter_black_slices and self.stats['black_filtered'] > 0:
            black_pct = self.stats['black_filtered'] / self.stats['total_considered'] * 100
            print(f"  Filtered out (black): {self.stats['black_filtered']:,} ({black_pct:.2f}%)")

        if self.stats['black_by_exam']:
            print(f"\n  Exams with black slices:")
            for exam, count in list(self.stats['black_by_exam'].items())[:5]:
                print(f"    - {exam}: {count} black slices")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        path, z, max_intensity = self.slices[idx]

        img = nib.load(path)
        slice_2d = img.get_fdata()[:, :, z]

        # Crop to ROI
        mask = slice_2d > 0.01
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
                rmax = min(slice_2d.shape[0], rmax + diff - diff // 2)
            else:
                # Add padding to left and right
                diff = height - width
                cmin = max(0, cmin - diff // 2)
                cmax = min(slice_2d.shape[1], cmax + diff - diff // 2)

            # Add 2px padding
            rmin = max(0, rmin - 2)
            rmax = min(slice_2d.shape[0], rmax + 2)
            cmin = max(0, cmin - 2)
            cmax = min(slice_2d.shape[1], cmax + 2)

            slice_2d = slice_2d[rmin:rmax, cmin:cmax]

        # Convert to tensor and add channel dimension
        slice_tensor = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0)

        # Resize to target size
        if slice_tensor.shape[1:] != self.target_size:
            slice_tensor = self.resize(slice_tensor)

        return slice_tensor


def visualize_dataset_samples(dataset, num_samples=8, save_path=None):
    """Simple visualization of dataset samples"""
    if len(dataset) == 0:
        print("Dataset is empty!")
        return

    num_samples = min(num_samples, len(dataset))
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(num_samples):
        sample = dataset[i]
        img = sample[0].numpy()

        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Sample {i + 1}')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Dataset Samples (Total: {len(dataset)} slices)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def compare_filtering_visualization(exam_dirs, target_size=(256, 256), skip_first=45):
    """Compare dataset with and without black slice filtering"""
    print("\n" + "=" * 60)
    print("COMPARING FILTERING STRATEGIES")
    print("=" * 60)

    # Dataset without filtering
    print("\n🔴 WITHOUT filtering:")
    dataset_unfiltered = PETSliceDataset(
        exam_dirs,
        target_size=target_size,
        skip_first=skip_first,
        filter_black_slices=False
    )

    # Dataset with filtering
    print("\n🟢 WITH filtering:")
    dataset_filtered = PETSliceDataset(
        exam_dirs,
        target_size=target_size,
        skip_first=skip_first,
        filter_black_slices=True,
        black_intensity_threshold=0.01
    )

    # Visualize both
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Show unfiltered samples (including potential black slices)
    for i in range(min(4, len(dataset_unfiltered))):
        sample = dataset_unfiltered[i]
        img = sample[0].numpy()
        axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Unfiltered {i + 1}')
        axes[0, i].axis('off')

    # Show filtered samples
    for i in range(min(4, len(dataset_filtered))):
        sample = dataset_filtered[i]
        img = sample[0].numpy()
        axes[1, i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Filtered {i + 1}')
        axes[1, i].axis('off')

    # If filtered dataset is smaller, show message
    if len(dataset_filtered) < 4:
        for i in range(len(dataset_filtered), 4):
            axes[1, i].text(0.5, 0.5, f'Only {len(dataset_filtered)} slices\nkept after filtering',
                            ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].axis('off')

    plt.suptitle(
        f'Comparison: Unfiltered ({len(dataset_unfiltered)} slices) vs Filtered ({len(dataset_filtered)} slices)\n'
        f'Removed {len(dataset_unfiltered) - len(dataset_filtered)} black slices ({100 * (len(dataset_unfiltered) - len(dataset_filtered)) / len(dataset_unfiltered):.1f}%)',
        fontsize=12)
    plt.tight_layout()
    plt.savefig("filtering_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

    return dataset_unfiltered, dataset_filtered


# ============================================
# TESTING CODE
# ============================================

if __name__ == "__main__":
    # Setup
    root = Path("/home/pedrocarreiro/Desktop/Dataset_Normalized2/")
    all_exams = sorted([d for d in root.iterdir() if d.is_dir()])

    # Test with a single exam first
    exam_index = 170
    selected_exams = [all_exams[exam_index]]
    skip_value = 45

    print(f"\n🔍 Testing exam: {selected_exams[0].name}")
    print(f"Skipping first {skip_value} slices")

    # Create dataset WITH filtering (recommended)
    dataset = PETSliceDataset(
        selected_exams,
        target_size=(256, 256),
        skip_first=skip_value,
        filter_black_slices=True,  # Filter out black slices
        black_intensity_threshold=0.01
    )

    # Visualize samples
    visualize_dataset_samples(dataset, num_samples=8, save_path="dataset_samples.png")

    # Optional: Compare filtering strategies
    print("\n" + "=" * 60)
    compare = input("\nWant to compare filtering vs no filtering? (y/n): ").lower()
    if compare == 'y':
        compare_filtering_visualization(selected_exams, target_size=(256, 256), skip_first=skip_value)

    # Show slice statistics if requested
    if len(dataset) > 0:
        print(f"\n📊 Final dataset: {len(dataset)} slices ready for training")
        print(f"   All black slices (max intensity < 0.01) have been removed")