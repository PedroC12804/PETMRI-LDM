import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class OverlayViewer:
    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)

        # ------------------------------------------------------------
        # Find exams
        # ------------------------------------------------------------
        self.exams = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir()
        ])

        self.exam_idx = 0
        self.slice_idx = 0

        # ------------------------------------------------------------
        # Setup figure
        # ------------------------------------------------------------
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key
        )

        self.load_exam()
        plt.show()

    def load_exam(self):

        exam_dir = self.exams[self.exam_idx]

        print("\n" + "=" * 70)
        print(f"Exam {self.exam_idx+1}/{len(self.exams)}")
        print(exam_dir.name)

        # ------------------------------------------------------------
        # Find PET
        # ------------------------------------------------------------
        pet_dir = exam_dir / "PET"

        pet_files = [
            f for f in pet_dir.glob("*.nii.gz")
            if "trimmed" in f.name
        ]

        if len(pet_files) == 0:
            print("⚠️ No PET found")
            return

        self.pet_path = pet_files[0]

        # ------------------------------------------------------------
        # Find registered MRI
        # ------------------------------------------------------------
        reg_dir = exam_dir / "MRI_trimmed"

        reg_files = list(
            reg_dir.glob("*.nii.gz")
        )

        if len(reg_files) == 0:
            print("⚠️ No registered MRI found")
            return

        self.mri_path = reg_files[0]

        # ------------------------------------------------------------
        # Load volumes
        # ------------------------------------------------------------
        self.pet = nib.load(
            str(self.pet_path)
        ).get_fdata()

        self.mri = nib.load(
            str(self.mri_path)
        ).get_fdata()

        print("PET shape:", self.pet.shape)
        print("MRI shape:", self.mri.shape)

        if self.pet.shape != self.mri.shape:
            print("⚠️ Shape mismatch")
            return

        # Start in middle slice
        self.slice_idx = self.pet.shape[2] // 2

        self.update_display()

    def update_display(self):

        pet_slice = self.pet[:, :, self.slice_idx]
        mri_slice = self.mri[:, :, self.slice_idx]

        # ------------------------------------------------------------
        # Normalize
        # ------------------------------------------------------------
        pet_slice = pet_slice.astype(np.float32)
        mri_slice = mri_slice.astype(np.float32)

        pet_slice /= np.max(pet_slice) + 1e-8
        mri_slice /= np.max(mri_slice) + 1e-8

        # ------------------------------------------------------------
        # Clear
        # ------------------------------------------------------------
        for ax in self.axes:
            ax.clear()

        # MRI
        self.axes[0].imshow(
            mri_slice.T,
            cmap="gray",
            origin="lower"
        )

        self.axes[0].set_title("MRI")
        self.axes[0].axis("off")

        # PET
        self.axes[1].imshow(
            pet_slice.T,
            cmap="hot",
            origin="lower"
        )

        self.axes[1].set_title("PET")
        self.axes[1].axis("off")

        # Overlay
        self.axes[2].imshow(
            mri_slice.T,
            cmap="gray",
            origin="lower"
        )

        self.axes[2].imshow(
            pet_slice.T,
            cmap="hot",
            alpha=0.4,
            origin="lower"
        )

        self.axes[2].set_title(
            f"Overlay | Slice {self.slice_idx}"
        )

        self.axes[2].axis("off")

        exam_name = self.exams[self.exam_idx].name

        self.fig.suptitle(
            f"{exam_name}\n"
            f"← → : slices | n : next exam | b : previous exam | q : quit",
            fontsize=12
        )

        self.fig.canvas.draw()

    def on_key(self, event):

        # ------------------------------------------------------------
        # Slice navigation
        # ------------------------------------------------------------
        if event.key == "right":

            self.slice_idx = min(
                self.slice_idx + 1,
                self.pet.shape[2] - 1
            )

            self.update_display()

        elif event.key == "left":

            self.slice_idx = max(
                self.slice_idx - 1,
                0
            )

            self.update_display()

        # ------------------------------------------------------------
        # Next exam
        # ------------------------------------------------------------
        elif event.key == "n":

            self.exam_idx = min(
                self.exam_idx + 1,
                len(self.exams) - 1
            )

            self.load_exam()

        # ------------------------------------------------------------
        # Previous exam
        # ------------------------------------------------------------
        elif event.key == "b":

            self.exam_idx = max(
                self.exam_idx - 1,
                0
            )

            self.load_exam()

        # ------------------------------------------------------------
        # Quit
        # ------------------------------------------------------------
        elif event.key == "q":

            plt.close()


if __name__ == "__main__":

    root_dir = "/home/pedrocarreiro/Desktop/Dataset_Normalized3"

    OverlayViewer(root_dir)