"""
Run the full low-dose PET simulation pipeline across every qualifying
patient in a dataset:
  1. LD_GEN.py (forward-project -> attenuation -> dose levels -> reconstruct)
  2. convert_recon_to_nifti.py
  3. cleanup_intermediate_files.py --apply

A patient "qualifies" if their folder contains both a PET_AC and a MuMap
series (as named by reorganize_pacs_folders.py).

Skips any patient whose output already has all 5 expected
final NIfTI files -- safe to stop and re-run this script at any time
without redoing completed work.

FAILURE-ISOLATED: if any step fails for one patient, it's logged and
the batch continues with the next patient, rather than aborting
everything. A summary of successes/failures is printed at the end and
written to a log file.

Usage:
    python run_batch_pipeline.py \\
        --dataset-root /path/to/reorganized_dataset \\
        --output-root /path/to/separate/output/location \\
        --template /path/to/template_mmr.hs \\
        --projector-par /path/to/forward_projector_proj_matrix_ray_tracing.par \\
        --recon-par /path/to/OSMAPOSL_QP.par \\
        --pipeline-script /path/to/LD_GEN.py
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

EXPECTED_DOSE_LEVELS = 5  # 50%, 25%, 10%, 5%, 1%


def find_qualifying_patients(dataset_root):
    """A patient folder qualifies if it directly contains subfolders
    matching both PET_AC and MuMap (substring match, same convention
    used throughout this project)."""
    patients = []
    for folder in dataset_root.iterdir():
        if not folder.is_dir():
            continue
        has_pet_ac = any("PET_AC" in c.name for c in folder.iterdir() if c.is_dir())
        has_mumap = any("MuMap" in c.name for c in folder.iterdir() if c.is_dir())
        if has_pet_ac and has_mumap:
            patients.append(folder)
    return sorted(patients)


def is_already_done(output_dir):

    if not output_dir.exists():
        return False
    nifti_count = len(list(output_dir.glob("*_recon_42.nii.gz")))
    return nifti_count >= EXPECTED_DOSE_LEVELS


def run_step(cmd, log_path):
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return result.returncode == 0


def process_one_patient(patient_dir, output_root, args, summary_log):
    patient_id = patient_dir.name
    output_dir = output_root / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_already_done(output_dir):
        print(f"[{patient_id}] Already completed -- skipping.")
        summary_log.write(f"{patient_id}\tSKIPPED (already done)\n")
        return "skipped"

    print(f"\n{'='*70}\n[{patient_id}] Starting pipeline\n{'='*70}")

    print(f"[{patient_id}] Step 1/3: running pipeline (forward-project -> reconstruct)...")
    ok = run_step(
        [sys.executable, str(args.pipeline_script),
         "--patient-dir", str(patient_dir),
         "--template", str(args.template),
         "--projector-par", str(args.projector_par),
         "--recon-par", str(args.recon_par),
         "--output-dir", str(output_dir)],
        output_dir / "batch_step1_pipeline.log"
    )
    if not ok:
        print(f"[{patient_id}] FAILED at pipeline generation step. "
              f"See {output_dir / 'batch_step1_pipeline.log'}")
        summary_log.write(f"{patient_id}\tFAILED (step 1: pipeline)\n")
        return "failed"

    print(f"[{patient_id}] Step 2/3: converting to NIfTI...")
    ok = run_step(
        [sys.executable, str(args.nifti_script),
         "--input-dir", str(output_dir)],
        output_dir / "batch_step2_nifti.log"
    )
    if not ok:
        print(f"[{patient_id}] FAILED at NIfTI conversion step. "
              f"See {output_dir / 'batch_step2_nifti.log'}")
        summary_log.write(f"{patient_id}\tFAILED (step 2: nifti conversion)\n")
        return "failed"

    print(f"[{patient_id}] Step 3/3: cleaning up intermediate files...")
    ok = run_step(
        [sys.executable, str(args.cleanup_script),
         str(output_dir), "--apply"],
        output_dir / "batch_step3_cleanup.log"
    )
    if not ok:
        print(f"[{patient_id}] WARNING: cleanup step failed, but generation "
              f"succeeded -- data is safe, just not cleaned up. "
              f"See {output_dir / 'batch_step3_cleanup.log'}")
        summary_log.write(f"{patient_id}\tPARTIAL (step 3 cleanup failed, "
                          f"but recon+nifti succeeded)\n")
        return "partial"

    print(f"[{patient_id}] COMPLETE.")
    summary_log.write(f"{patient_id}\tSUCCESS\n")
    return "success"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--projector-par", required=True, type=Path)
    parser.add_argument("--recon-par", required=True, type=Path)
    parser.add_argument("--pipeline-script", required=True, type=Path)
    parser.add_argument("--nifti-script", type=Path,
                         default=Path(__file__).parent / "convert_recon_nifti.py")
    parser.add_argument("--cleanup-script", type=Path,
                         default=Path(__file__).parent / "clean_intermediate_files.py")
    args = parser.parse_args()

    for attr in ["dataset_root", "output_root", "template", "projector_par",
                 "recon_par", "pipeline_script", "nifti_script", "cleanup_script"]:
        setattr(args, attr, getattr(args, attr).resolve())

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.dataset_root} for qualifying patients "
          f"(need both PET_AC and MuMap)...")
    patients = find_qualifying_patients(args.dataset_root)
    print(f"Found {len(patients)} qualifying patient(s).\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = args.output_root / f"batch_summary_{timestamp}.txt"

    counts = {"success": 0, "failed": 0, "skipped": 0, "partial": 0}

    with open(summary_path, "w") as summary_log:
        summary_log.write(f"Batch run started: {datetime.now()}\n")
        summary_log.write(f"Dataset root: {args.dataset_root}\n")
        summary_log.write(f"Total qualifying patients: {len(patients)}\n\n")

        for idx, patient_dir in enumerate(patients, 1):
            print(f"\n### Patient {idx}/{len(patients)} ###")
            status = process_one_patient(patient_dir, args.output_root, args, summary_log)
            counts[status] += 1
            summary_log.flush()

        summary_log.write(f"\nBatch run finished: {datetime.now()}\n")
        summary_log.write(f"Success: {counts['success']}, Failed: {counts['failed']}, "
                          f"Skipped: {counts['skipped']}, Partial: {counts['partial']}\n")

    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Success: {counts['success']}")
    print(f"  Failed:  {counts['failed']}")
    print(f"  Skipped (already done): {counts['skipped']}")
    print(f"  Partial (cleanup only failed): {counts['partial']}")
    print(f"\nFull summary written to: {summary_path}")
    print(f"If interrupted, just re-run this same command -- completed "
          f"patients will be skipped automatically.")


if __name__ == "__main__":
    main()