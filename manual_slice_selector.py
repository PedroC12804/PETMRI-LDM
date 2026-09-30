import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


class PETSliceSelector:
    def __init__(self, nii_path, output_suffix="_trimmed.nii.gz"):
        """
        Args:
            nii_path: Path to the PET .nii.gz file
            output_suffix: Suffix for saved trimmed file (e.g., "_trimmed.nii.gz")
        """
        self.nii_path = Path(nii_path)
        self.img = nib.load(str(self.nii_path))
        self.data = self.img.get_fdata()  # shape (x, y, z)
        self.affine = self.img.affine
        self.header = self.img.header
        self.num_slices = self.data.shape[2]

        # Decision array: None = undecided, True = keep, False = discard
        self.decisions = [None] * self.num_slices
        self.current_idx = 0

        self.output_path = self.nii_path.parent / f"{self.nii_path.stem}{output_suffix}"
        self.meta_path = self.output_path.with_suffix(".json")

        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_display()
        plt.title(
            f"Exam: {self.nii_path.parent.parent.name} / {self.nii_path.parent.name}\nUse ← → | k=keep | d=discard | u=undecided | n=next exam | s=save & exit | q=quit")
        plt.show()

    def update_display(self):
        """Show current slice with color coding and info"""
        slice_data = self.data[:, :, self.current_idx]
        self.ax.clear()
        self.ax.imshow(slice_data, cmap='gray', vmin=0, vmax=np.percentile(self.data, 99))

        # Border color based on decision
        decision = self.decisions[self.current_idx]
        if decision is True:
            color = 'lime'
            status = "KEPT"
        elif decision is False:
            color = 'red'
            status = "DISCARDED"
        else:
            color = 'yellow'
            status = "UNDECIDED"

        # Draw border
        for spine in self.ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(5)

        # Add text info
        kept = sum(1 for d in self.decisions if d is True)
        discarded = sum(1 for d in self.decisions if d is False)
        undecided = self.num_slices - kept - discarded
        info = (f"Slice {self.current_idx + 1}/{self.num_slices} | {status}\n"
                f"Kept: {kept} | Discarded: {discarded} | Undecided: {undecided}")
        self.ax.set_xlabel(info, fontsize=10, color=color)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.current_idx = min(self.current_idx + 1, self.num_slices - 1)
            self.update_display()
        elif event.key == 'left':
            self.current_idx = max(self.current_idx - 1, 0)
            self.update_display()
        elif event.key == 'k':
            self.decisions[self.current_idx] = True
            # Auto-advance to next slice
            if self.current_idx < self.num_slices - 1:
                self.current_idx += 1
            self.update_display()
        elif event.key == 'd':
            self.decisions[self.current_idx] = False
            if self.current_idx < self.num_slices - 1:
                self.current_idx += 1
            self.update_display()
        elif event.key == 'u':
            self.decisions[self.current_idx] = None
            self.update_display()
        elif event.key == 'n':  # next exam
            self.save_trimmed()
            plt.close()
        elif event.key == 's':  # save and exit (stop all)
            self.save_trimmed()
            plt.close()
            raise StopIteration("User requested stop")
        elif event.key == 'q':  # quit without saving this exam
            plt.close()
            raise StopIteration("User quit")

    def save_trimmed(self):
        """Create new NIfTI with only kept slices"""
        kept_indices = [i for i, dec in enumerate(self.decisions) if dec is True]
        if not kept_indices:
            print(f"⚠️ No slices kept for {self.nii_path}. Skipping saving.")
            return

        # Stack kept slices along z-axis
        kept_data = self.data[:, :, kept_indices]

        # Create new NIfTI image
        new_img = nib.Nifti1Image(kept_data, self.affine, self.header)
        nib.save(new_img, str(self.output_path))

        # Save decision metadata as JSON
        meta = {
            "original_file": str(self.nii_path),
            "kept_slices": kept_indices,
            "discarded_slices": [i for i, dec in enumerate(self.decisions) if dec is False],
            "num_slices_original": self.num_slices,
            "num_slices_kept": len(kept_indices)
        }
        with open(self.meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"✓ Saved trimmed PET: {self.output_path}")
        print(f"  Kept {len(kept_indices)} / {self.num_slices} slices")


def process_all_exams(root_dir, output_suffix="_trimmed.nii.gz", exam_subset=None):
    """
    Iterate over all exam directories, run interactive selector for each PET file.

    Args:
        root_dir: Path to dataset root (contains exam folders)
        output_suffix: Suffix for saved trimmed files
        exam_subset: Optional list of exam names to process (None = all)
    """
    root = Path(root_dir)
    exam_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if exam_subset:
        exam_dirs = [d for d in exam_dirs if d.name in exam_subset]

    print(f"Found {len(exam_dirs)} exams. Starting interactive selection...")
    print(
        "Controls:\n  ← → : navigate slices\n  k : keep current slice\n  d : discard current slice\n  u : undecided (reset)\n  n : next exam (save)\n  s : save current exam and quit\n  q : quit without saving current exam")

    for i, exam_dir in enumerate(exam_dirs):
        pet_dir = exam_dir / "PET"
        if not pet_dir.exists():
            print(f"⚠️ No PET folder in {exam_dir.name}, skipping.")
            continue

        pet_files = list(pet_dir.glob("*.nii.gz"))
        # Exclude already trimmed files to avoid reprocessing
        pet_files = [f for f in pet_files if output_suffix not in f.name]
        if not pet_files:
            print(f"⚠️ No PET .nii.gz in {pet_dir}, skipping.")
            continue

        pet_path = pet_files[0]  # assume one PET per exam
        print(f"\n{'=' * 60}\nProcessing exam {i + 1}/{len(exam_dirs)}: {exam_dir.name}\nFile: {pet_path}")

        try:
            selector = PETSliceSelector(pet_path, output_suffix=output_suffix)
        except StopIteration:
            print("User interrupted. Exiting.")
            break
        except Exception as e:
            print(f"Error processing {exam_dir.name}: {e}")
            continue


if __name__ == "__main__":
    # Example usage
    root_dir = Path("/home/pedrocarreiro/Desktop/Dataset_2/")
    # To process all exams:
    #process_all_exams(root_dir)
    # To process only specific exams:
    process_all_exams(root_dir, exam_subset=["5974235"])