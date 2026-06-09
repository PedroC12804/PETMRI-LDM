import SimpleITK as sitk
from pathlib import Path


def rigid_register_mri_to_pet(
    pet_path,
    mri_path,
    output_path
):
    """
    Simple rigid MRI -> PET registration.

    Keeps registration intentionally conservative:
    - rigid only
    - no deformable transforms
    - minimal interpolation artifacts

    PET = fixed image
    MRI = moving image
    """

    print("\n" + "=" * 60)
    print("Loading images...")

    # ------------------------------------------------------------
    # Load images
    # ------------------------------------------------------------
    fixed_pet = sitk.ReadImage(
        str(pet_path),
        sitk.sitkFloat32
    )

    moving_mri = sitk.ReadImage(
        str(mri_path),
        sitk.sitkFloat32
    )

    print("PET size:", fixed_pet.GetSize())
    print("MRI size:", moving_mri.GetSize())

    # ------------------------------------------------------------
    # Initial alignment
    # ------------------------------------------------------------
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_pet,
        moving_mri,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # ------------------------------------------------------------
    # Registration setup
    # ------------------------------------------------------------
    registration = sitk.ImageRegistrationMethod()

    # Mutual information works well for PET/MRI
    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50
    )

    registration.SetMetricSamplingStrategy(
        registration.RANDOM
    )

    registration.SetMetricSamplingPercentage(0.1)

    registration.SetInterpolator(
        sitk.sitkLinear
    )

    # Conservative optimizer
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=100,
        relaxationFactor=0.5
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetInitialTransform(
        initial_transform,
        inPlace=False
    )

    # ------------------------------------------------------------
    # Run registration
    # ------------------------------------------------------------
    print("Running rigid registration...")

    final_transform = registration.Execute(
        fixed_pet,
        moving_mri
    )

    print("Done.")
    print("Final metric value:",
          registration.GetMetricValue())

    # ------------------------------------------------------------
    # Resample MRI into PET space
    # ------------------------------------------------------------
    print("Resampling MRI...")

    registered_mri = sitk.Resample(
        moving_mri,
        fixed_pet,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_mri.GetPixelID()
    )

    # ------------------------------------------------------------
    # Save registered MRI
    # ------------------------------------------------------------
    sitk.WriteImage(
        registered_mri,
        str(output_path)
    )

    print(f"✓ Saved:")
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
        # PET
        # --------------------------------------------------------
        pet_dir = exam_dir / "PET"

        pet_files = [
            f for f in pet_dir.glob("*.nii.gz")
            if "trimmed" not in f.name
        ]

        if len(pet_files) == 0:
            print("⚠️ No PET found")
            continue

        pet_path = pet_files[0]

        # --------------------------------------------------------
        # MRI
        # --------------------------------------------------------
        mri_dir = exam_dir / "T1"

        mri_files = list(
            mri_dir.glob("*.nii.gz")
        )

        if len(mri_files) == 0:
            print("⚠️ No MRI found")
            continue

        mri_path = mri_files[0]

        # --------------------------------------------------------
        # Output
        # --------------------------------------------------------
        output_dir = exam_dir / "MRI_registered"
        output_dir.mkdir(exist_ok=True)

        output_path = (
            output_dir /
            "t1_registered_to_pet.nii.gz"
        )

        try:

            rigid_register_mri_to_pet(
                pet_path,
                mri_path,
                output_path
            )

        except Exception as e:

            print(f"❌ Error processing {exam_dir.name}")
            print(e)


if __name__ == "__main__":

    root_dir = "/home/pedrocarreiro/Desktop/Dataset_Normalized3"

    process_dataset(root_dir)