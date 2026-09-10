"""CLI entrypoint to execute the BANKING77 dataset audit and reproducible split pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATASET_NAME,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REPORT_PATH,
    VAL_SIZE,
    get_environment_metadata,
)
from src.loader import load_banking77
from src.audit import audit_dataset
from src.splitter import create_reproducible_split, save_processed_splits
from src.reporter import generate_markdown_report, print_console_summary


def run_pipeline(
    dataset_name: str = DATASET_NAME,
    val_size: float = VAL_SIZE,
    random_seed: int = RANDOM_SEED,
    output_data_dir: Path = PROCESSED_DATA_DIR,
    report_output_path: Path = REPORT_PATH,
) -> int:
    """Run the audit, split, verification, and reporting pipeline."""
    print("[1/5] Gathering environment metadata...")
    env_metadata = get_environment_metadata()

    print(f"[2/5] Loading official dataset '{dataset_name}'...")
    loaded = load_banking77(dataset_name=dataset_name)
    print(f"      Official train samples: {len(loaded.train_df):,}")
    print(f"      Official test samples : {len(loaded.test_df):,}")
    print(f"      Unique intent classes : {len(loaded.label_names)}")

    print("[3/5] Auditing dataset (lengths, duplicates, quality diagnostics)...")
    audit_res = audit_dataset(
        train_df=loaded.train_df,
        test_df=loaded.test_df,
        intent_names=loaded.label_names,
        dataset_name=dataset_name,
        environment_metadata=env_metadata,
    )

    print(f"[4/5] Creating reproducible stratified split (seed={random_seed}, val_ratio={val_size})...")
    split_res = create_reproducible_split(
        train_df=loaded.train_df,
        test_df=loaded.test_df,
        val_size=val_size,
        random_seed=random_seed,
    )

    print(f"      Train sub-split: {len(split_res.train_df):,} examples")
    print(f"      Val sub-split  : {len(split_res.val_df):,} examples")
    print(f"      Test split     : {len(split_res.test_df):,} examples (untouched)")
    print(f"      Integrity check: {'PASSED' if split_res.verification.passed else 'FAILED'}")

    print("[5/5] Saving split artifacts and generating audit report...")
    saved_paths = save_processed_splits(split_res, output_dir=output_data_dir)
    print(f"      Saved processed splits to: {output_data_dir}")

    generate_markdown_report(
        audit=audit_res,
        split_res=split_res,
        output_path=report_output_path,
    )
    print(f"      Saved audit report to: {report_output_path}")

    print_console_summary(audit_res, split_res)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Audit and create reproducible stratified split for BANKING77."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAME,
        help="Hugging Face dataset identifier (default: banking77)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=VAL_SIZE,
        help="Validation split ratio from official train (default: 0.20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory to save processed split files",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_PATH,
        help="Path for saving markdown audit report",
    )

    args = parser.parse_args()
    sys.exit(
        run_pipeline(
            dataset_name=args.dataset,
            val_size=args.val_size,
            random_seed=args.seed,
            output_data_dir=args.output_dir,
            report_output_path=args.report_path,
        )
    )


if __name__ == "__main__":
    main()
