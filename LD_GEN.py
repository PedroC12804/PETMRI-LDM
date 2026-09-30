"""
Automated low-dose PET simulation pipeline for one mMR patient.

Reuses every command validated by hand earlier in this project:
  forward_project -> calculate_attenuation_coefficients ->
  norm/randoms/scatter placeholders -> combine into noiseless "true"
  prompts -> for each dose level: poisson_noise + scaled additive +
  OSMAPOSL reconstruction.

Dose levels match the vendor-software (PET-TIME truncation) comparison:
  50%, 25%, 10%, 5%, 1%

Requires (already validated, reusable across all mMR patients):
  - a scanner template sinogram (template_mmr.hs)
  - the forward projector par file (forward_projector_proj_matrix_ray_tracing.par)
  - the reconstruction par file (OSMAPOSL_QP.par, with subsets/subiterations
    already corrected for this scanner's view count)

Per-patient inputs:
  - emission image (.hv/.v, converted from the patient's PET NIfTI/DICOM)
  - attenuation image (.hv/.v, resampled mu-map, already scaled to cm^-1)

Usage:
    python LD_GEN.py \\
        --patient-id PATIENT001 \\
        --emission-image patient001_emission.hv \\
        --atten-image patient001_atten.hv \\
        --template template_mmr.hs \\
        --projector-par forward_projector_proj_matrix_ray_tracing.par \\
        --recon-par OSMAPOSL_QP.par \\
        --output-dir /path/to/output/patient001
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DOSE_LEVELS = [0.50, 0.25,0.1,0.05,0.01]  # 50%, 25%, 10%, 5%, 1%
DEFAULT_SEED = 12345
MU_MAP_SCALE_FACTOR = 10000.0


def resample_z_spacing(image_sitk, target_z_spacing):
    """Resample an image so its z-spacing exactly matches target_z_spacing,
    keeping xy spacing/size and physical extent unchanged. STIR's
    projector requires emission image z-spacing = ring_spacing/integer;
    real clinical reconstructions vary in slice thickness across
    patients/protocols, so this must be enforced per-patient rather
    than assumed."""
    orig_spacing = image_sitk.GetSpacing()  # (x, y, z)
    orig_size = image_sitk.GetSize()  # (x, y, z)

    physical_extent_z = orig_spacing[2] * orig_size[2]
    new_size_z = int(round(physical_extent_z / target_z_spacing))

    new_spacing = (orig_spacing[0], orig_spacing[1], target_z_spacing)
    new_size = (orig_size[0], orig_size[1], new_size_z)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image_sitk)


def get_study_id_from_pet_folder(patient_dir):
    """Read StudyID directly from the actual PET_AC DICOM header --
    more robust than assuming the folder name matches, since folder
    names could in principle be renamed/copied by hand at some point."""
    import pydicom
    try:
        pet_folder = find_series_folder(patient_dir, "PET_AC")
    except FileNotFoundError:
        return None

    for path in pet_folder.iterdir():
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            study_id = str(getattr(ds, "StudyID", "")).strip()
            if study_id:
                return study_id
        except Exception:
            continue
    return None


def find_series_folder(patient_dir, keyword):
    """Find a subfolder whose name contains the given keyword
    (case-sensitive, matching reorganize_pacs_folders.py's output
    naming: 'PET_AC', 'MuMap')."""
    matches = [p for p in patient_dir.rglob("*") if p.is_dir() and keyword in p.name]
    if not matches:
        raise FileNotFoundError(
            f"No folder containing '{keyword}' found under {patient_dir}"
        )
    if len(matches) > 1:
        print(f"  WARNING: multiple '{keyword}' folders found, using first: "
              f"{[m.name for m in matches]}")
    return matches[0]


def read_dicom_series(directory):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(directory))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {directory}")
    filenames = reader.GetGDCMSeriesFileNames(str(directory), series_ids[0])
    reader.SetFileNames(filenames)
    return reader.Execute()


def write_interfile_from_sitk(image_sitk, out_basename, is_atten_image=False):
    """Write a SimpleITK image directly to STIR Interfile format
    (.hv/.v), using the confirmed real header syntax from earlier in
    this project. SimpleITK's GetArrayFromImage already returns data in
    (z,y,x) order, matching Interfile's expected matrix convention."""
    arr = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    vx, vy, vz = image_sitk.GetSpacing()  # SimpleITK spacing is (x,y,z)
    nz, ny, nx = arr.shape

    out_basename = Path(out_basename)
    v_path = out_basename.with_suffix(".v")
    hv_path = out_basename.with_suffix(".hv")

    arr.astype("<f4").tofile(v_path)

    type_of_data = "PET"  # confirmed required even for attenuation images
    header_text = f"""!INTERFILE :=
name of data file := {v_path.name}
!GENERAL DATA :=
!GENERAL IMAGE DATA :=
imagedata byte order := LITTLEENDIAN
!type of data := {type_of_data}
number format := float
!number of bytes per pixel := 4
number of dimensions := 3
matrix axis label [1] := x
!matrix size [1] := {nx}
scaling factor (mm/pixel) [1] := {vx:.6f}
matrix axis label [2] := y
!matrix size [2] := {ny}
scaling factor (mm/pixel) [2] := {vy:.6f}
matrix axis label [3] := z
!matrix size [3] := {nz}
scaling factor (mm/pixel) [3] := {vz:.6f}
number of time frames := 1
image scaling factor[1] := 1
data offset in bytes[1] := 0
quantification units := 1
!END OF INTERFILE :=
"""
    with open(hv_path, "w") as f:
        f.write(header_text)

    return hv_path


def prepare_interfile_images(patient_dir, output_dir, patient_id,
                              ring_spacing_mm, z_divisor):
    """Find PET_AC and MuMap DICOM series under patient_dir, correct the
    PET image's z-spacing to satisfy STIR's requirement
    (z-spacing = ring_spacing / integer), resample the mu-map onto that
    corrected grid, apply the confirmed /10000 scale factor, and write
    both as STIR Interfile images. Returns (emission_hv_path, atten_hv_path)."""
    print(f"\n[{patient_id}] Locating PET_AC and MuMap series...")
    pet_folder = find_series_folder(patient_dir, "PET_AC")
    mumap_folder = find_series_folder(patient_dir, "MuMap")
    print(f"  PET_AC:  {pet_folder}")
    print(f"  MuMap:   {mumap_folder}")

    print(f"[{patient_id}] Reading DICOM series...")
    pet_img = read_dicom_series(pet_folder)
    mumap_img = read_dicom_series(mumap_folder)
    print(f"  PET size: {pet_img.GetSize()}, spacing: {pet_img.GetSpacing()}")
    print(f"  MuMap size: {mumap_img.GetSize()}, spacing: {mumap_img.GetSpacing()}")

    target_z_spacing = ring_spacing_mm / z_divisor
    actual_z_spacing = pet_img.GetSpacing()[2]
    if abs(actual_z_spacing - target_z_spacing) > 1e-4:
        print(f"[{patient_id}] PET z-spacing ({actual_z_spacing:.5f} mm) does not "
              f"match required ring_spacing/{z_divisor} ({target_z_spacing:.5f} mm) "
              f"-- resampling to correct this (STIR requires this exact "
              f"relationship for its projector)...")
        pet_img = resample_z_spacing(pet_img, target_z_spacing)
        print(f"  corrected PET size: {pet_img.GetSize()}, spacing: {pet_img.GetSpacing()}")
    else:
        print(f"[{patient_id}] PET z-spacing already matches "
              f"ring_spacing/{z_divisor} -- no correction needed.")

    print(f"[{patient_id}] Resampling mu-map onto (corrected) PET grid...")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(pet_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    mumap_resampled = resampler.Execute(mumap_img)

    print(f"[{patient_id}] Applying confirmed scale factor "
          f"(pixel_value / {MU_MAP_SCALE_FACTOR:.0f} -> 1/cm)...")
    mumap_float = sitk.Cast(mumap_resampled, sitk.sitkFloat32)
    mumap_mu = mumap_float / MU_MAP_SCALE_FACTOR

    stats = sitk.StatisticsImageFilter()
    stats.Execute(mumap_mu)
    print(f"  resampled mu-map: min={stats.GetMinimum():.4f}, "
          f"max={stats.GetMaximum():.4f}, mean={stats.GetMean():.4f}")

    print(f"[{patient_id}] Writing Interfile images...")
    emission_hv = write_interfile_from_sitk(
        pet_img, output_dir / f"{patient_id}_emission"
    )
    atten_hv = write_interfile_from_sitk(
        mumap_mu, output_dir / f"{patient_id}_atten"
    )
    print(f"  emission image: {emission_hv}")
    print(f"  atten image:    {atten_hv}")

    return emission_hv, atten_hv


def run(cmd, cwd, log_name, env=None):
    """Run a command, logging stdout/stderr to a file, raising on
    failure with the log contents included in the error for easy
    debugging."""
    log_path = cwd / f"{log_name}.log"
    print(f"  Running: {' '.join(str(c) for c in cmd)}")
    with open(log_path, "w") as logf:
        result = subprocess.run(
            cmd, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT, env=env
        )
    if result.returncode != 0:
        log_contents = log_path.read_text(errors="ignore")
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}\n"
            f"--- log ({log_path}) ---\n{log_contents[-3000:]}"
        )


def build_ground_truth(patient_id, emission_image, atten_image, template,
                        projector_par, output_dir):
    """Steps 1-8: forward-project through the noiseless 'true' prompts
    sinogram. This is computed ONCE per patient and reused across all
    dose levels."""
    prefix = output_dir / patient_id

    print(f"\n[{patient_id}] Step 1: forward-project emission image...")
    line_integrals = f"{prefix}_line_integrals.hs"
    run(["forward_project", line_integrals, str(emission_image),
         str(template), str(projector_par)],
        output_dir, f"{patient_id}_01_forward_project")

    print(f"[{patient_id}] Step 2: attenuation correction factors...")
    acfs = f"{prefix}_acfs.hs"
    run(["calculate_attenuation_coefficients", "--ACF", acfs,
         str(atten_image), str(template), str(projector_par)],
        output_dir, f"{patient_id}_02_acfs")

    print(f"[{patient_id}] Step 3: normalization placeholder (constant 3.4)...")
    norm = f"{prefix}_norm.hs"
    run(["stir_math", "-s", "--including-first", "--times-scalar", "0",
         "--add-scalar", "3.4", norm, acfs],
        output_dir, f"{patient_id}_03_norm")

    print(f"[{patient_id}] Step 4: combine ACF x norm -> multfactors...")
    multfactors = f"{prefix}_multfactors.hs"
    run(["stir_math", "-s", "--mult", multfactors, norm, acfs],
        output_dir, f"{patient_id}_04_multfactors")

    print(f"[{patient_id}] Step 5: constant randoms background...")
    randoms = f"{prefix}_randoms.hs"
    run(["stir_math", "-s", "--including-first", "--times-scalar", "0",
         "--add-scalar", "10", randoms, line_integrals],
        output_dir, f"{patient_id}_05a_randoms")
    run(["stir_divide", "-s", "--accumulate", randoms, norm],
        output_dir, f"{patient_id}_05b_randoms_divide")

    print(f"[{patient_id}] Step 6: zero scatter (sanctioned shortcut)...")
    scatter = f"{prefix}_scatter.hs"
    run(["stir_math", "-s", "--including-first", "--times-scalar", "0",
         "--add-scalar", "0", scatter, line_integrals],
        output_dir, f"{patient_id}_06_scatter")

    print(f"[{patient_id}] Step 7: additive sinogram (randoms+scatter, corrected)...")
    additive = f"{prefix}_additive_sinogram.hs"
    run(["stir_math", "-s", additive, randoms, scatter],
        output_dir, f"{patient_id}_07a_additive")
    run(["stir_math", "-s", "--mult", "--accumulate", additive, multfactors],
        output_dir, f"{patient_id}_07b_additive_mult")

    print(f"[{patient_id}] Step 8: final noiseless 'true' prompts...")
    prompts = f"{prefix}_prompts.hs"
    run(["stir_math", "-s", prompts, line_integrals, additive],
        output_dir, f"{patient_id}_08a_prompts")
    run(["stir_divide", "-s", "--accumulate", prompts, multfactors],
        output_dir, f"{patient_id}_08b_prompts_divide")

    return {
        "prompts": prompts,
        "additive": additive,
        "multfactors": multfactors,
    }


def run_dose_level(patient_id, ground_truth, dose_fraction, recon_par,
                    output_dir, seed=DEFAULT_SEED):
    """Step 9: for one dose level, apply Poisson noise + scale additive,
    then reconstruct."""
    pct_label = f"{dose_fraction*100:.0f}pct"
    prefix = output_dir / f"{patient_id}_dose{pct_label}"

    print(f"\n[{patient_id}] Dose {pct_label}: applying Poisson noise "
          f"(scaling factor {dose_fraction})...")
    noisy_prompts_base = f"{prefix}_prompts_noisy"
    run(["poisson_noise", noisy_prompts_base, ground_truth["prompts"],
         str(dose_fraction), str(seed)],
        output_dir, f"{patient_id}_dose{pct_label}_09a_poisson")
    noisy_prompts = f"{noisy_prompts_base}.hs"

    print(f"[{patient_id}] Dose {pct_label}: scaling additive sinogram...")
    scaled_additive = f"{prefix}_additive.hs"
    run(["stir_math", "-s", "--times-scalar", str(dose_fraction),
         scaled_additive, ground_truth["additive"]],
        output_dir, f"{patient_id}_dose{pct_label}_09b_scale_additive")

    print(f"[{patient_id}] Dose {pct_label}: reconstructing...")
    recon_output = f"{prefix}_recon"
    env = os.environ.copy()
    env["INPUT"] = noisy_prompts
    env["MULTFACTORS"] = ground_truth["multfactors"]
    env["ADDSINO"] = scaled_additive
    env["OUTPUT"] = recon_output
    run(["OSMAPOSL", str(recon_par)], output_dir,
        f"{patient_id}_dose{pct_label}_10_reconstruct", env=env)

    print(f"[{patient_id}] Dose {pct_label}: DONE -> {recon_output}*.hv")
    return recon_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id", default=None,
                         help="Label used to prefix output files. If omitted, "
                              "defaults to the Study ID read directly from the "
                              "PET_AC DICOM header, falling back to the "
                              "--patient-dir folder name only if that's not found.")
    parser.add_argument("--patient-dir", required=True, type=Path,
                         help="Directory containing PET_AC and MuMap DICOM "
                              "series subfolders (as named by "
                              "reorganize_pacs_folders.py)")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--projector-par", required=True, type=Path)
    parser.add_argument("--recon-par", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ring-spacing-mm", type=float, default=4.0625,
                         help="Scanner ring spacing in mm (default: 4.0625, "
                              "Siemens mMR). Use 3.21843 for Biograph One.")
    parser.add_argument("--z-divisor", type=int, default=2,
                         help="Target emission image z-spacing = "
                              "ring_spacing_mm / z_divisor (default: 2)")
    args = parser.parse_args()

    # Resolve everything to absolute paths immediately -- since STIR
    # commands run with cwd set to output_dir, relative paths here would
    # otherwise be interpreted relative to output_dir, not wherever this
    # script was launched from. Resolving now means it doesn't matter
    # where you run this from.
    args.patient_dir = args.patient_dir.resolve()
    args.template = args.template.resolve()
    args.projector_par = args.projector_par.resolve()
    args.recon_par = args.recon_par.resolve()
    args.output_dir = args.output_dir.resolve()

    if args.patient_id is None:
        study_id = get_study_id_from_pet_folder(args.patient_dir)
        if study_id:
            args.patient_id = study_id
            print(f"(no --patient-id given, using StudyID read from DICOM: "
                  f"'{args.patient_id}')")
        else:
            args.patient_id = args.patient_dir.name
            print(f"(no --patient-id given and no StudyID found in DICOM, "
                  f"falling back to folder name: '{args.patient_id}')")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}\nPatient: {args.patient_id}\n{'='*70}")

    try:
        emission_image, atten_image = prepare_interfile_images(
            args.patient_dir, args.output_dir, args.patient_id,
            args.ring_spacing_mm, args.z_divisor
        )
    except Exception as e:
        print(f"\nERROR preparing Interfile images for {args.patient_id}:\n{e}")
        sys.exit(1)

    try:
        ground_truth = build_ground_truth(
            args.patient_id, emission_image, atten_image,
            args.template, args.projector_par, args.output_dir
        )
    except RuntimeError as e:
        print(f"\nERROR building ground truth for {args.patient_id}:\n{e}")
        sys.exit(1)

    results = {}
    for dose in DOSE_LEVELS:
        try:
            recon_path = run_dose_level(
                args.patient_id, ground_truth, dose, args.recon_par,
                args.output_dir, seed=args.seed
            )
            results[dose] = recon_path
        except RuntimeError as e:
            print(f"\nERROR at dose {dose*100:.0f}% for {args.patient_id}:\n{e}")
            results[dose] = None

    print(f"\n{'='*70}\nSUMMARY for {args.patient_id}\n{'='*70}")
    for dose, path in results.items():
        status = path if path else "FAILED"
        print(f"  {dose*100:>5.0f}% dose -> {status}")


if __name__ == "__main__":
    main()