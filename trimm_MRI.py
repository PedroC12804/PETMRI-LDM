import nibabel as nib
import numpy as np
from pathlib import Path
import json


def trim_registered_mri(
    registered_mri_path,
    json_path,
    output_path
):
    """
    Trim registered MRI using PET kept slice indices.
    """

    print("\n" + "=" * 60)
    print("Processing:")
    print(registered_mri_path)

    # ------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------
    with open(json_path, "r") as f:
        meta = json.load(f)

    kept_indices = meta["kept_slices"]

    print(f"Keeping {len(kept_indices)} slices")

    # ------------------------------------------------------------
    # Load MRI
    # ------------------------------------------------------------
    mri_img = nib.load(str(registered_mri_path))

    mri_data = mri_img.get_fdata()

    print("MRI shape:", mri_data.shape)

    # ------------------------------------------------------------
    # Trim MRI
    # ------------------------------------------------------------
    trimmed_mri = mri_data[:, :, kept_indices]

    print("Trimmed MRI shape:", trimmed_mri.shape)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    trimmed_img = nib.Nifti1Image(
        trimmed_mri,
        mri_img.affine,
        mri_img.header
    )

    nib.save(
        trimmed_img,
        str(output_path)
    )

    print("✓ Saved:")
    print(output_path)


def process_dataset(root_dir):

    root_dir = Path(root_dir)

    exam_dirs = sorted([
        d for d in root_dir.iterdir()
        if d.is_dir()
    ])

    print(f"Found {len(exam_dirs)} exams")

    for i, exam_dir in enumerate(exam_dirs):

        print("\n" + "#" * 70)
        print(f"Exam {i+1}/{len(exam_dirs)}")
        print(exam_dir.name)

        # --------------------------------------------------------
        # Registered MRI
        # --------------------------------------------------------
        reg_dir = exam_dir / "MRI_registered2"

        reg_files = list(
            reg_dir.glob("*.nii.gz")
        )

        if len(reg_files) == 0:
            print("⚠️ No registered MRI found")
            continue

        registered_mri_path = reg_files[0]

        # --------------------------------------------------------
        # JSON metadata
        # --------------------------------------------------------
        pet_dir = exam_dir / "PET"

        json_files = list(
            pet_dir.glob("*nii.json")
        )

        if len(json_files) == 0:
            print("⚠️ No JSON metadata found")
            continue

        json_path = json_files[0]

        # --------------------------------------------------------
        # Output
        # --------------------------------------------------------
        output_dir = exam_dir / "MRI_trimmed2"
        output_dir.mkdir(exist_ok=True)

        output_path = (
            output_dir /
            "t1_registered_trimmed.nii.gz"
        )

        try:

            trim_registered_mri(
                registered_mri_path,
                json_path,
                output_path
            )

        except Exception as e:

            print(f"❌ Error processing {exam_dir.name}")
            print(e)


if __name__ == "__main__":

    root_dir = "/home/pedrocarreiro/Desktop/Dataset_2"

    process_dataset(root_dir)