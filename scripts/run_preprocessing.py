from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow running this script directly from the repository root, e.g.
# python scripts/run_preprocessing.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing import preprocess_asas_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the ASAS/MRSi-AI preprocessing pipeline and save an "
            "interval-level CSV file."
        )
    )

    parser.add_argument(
        "--cohort-data",
        required=True,
        type=Path,
        help="Path to the cohort-level input file. Supported formats: .csv, .sas7bdat.",
    )
    parser.add_argument(
        "--glucose-data",
        required=True,
        type=Path,
        help="Path to the glucose/insulin measurement Excel file.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
        help="Path where the preprocessed CSV file will be saved.",
    )
    parser.add_argument(
        "--scale-income",
        action="store_true",
        help=(
            "Divide V1_INCOME-V10_INCOME by 10,000 before preprocessing. "
            "Use this only when the raw income variables have not already been scaled."
        ),
    )
    parser.add_argument(
        "--no-impute",
        action="store_true",
        help=(
            "Do not impute missing values in the final interval-level dataframe. "
            "Use this if imputation will be performed after train/test/validation splitting."
        ),
    )
    parser.add_argument(
        "--keep-drug-missing-baseline",
        action="store_true",
        help=(
            "Do not drop medication-outcome rows with missing baseline variables "
            "used in the original preprocessing workflow."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )

    return parser.parse_args()


def main() -> None:
    """Run preprocessing from command-line arguments."""
    args = parse_args()

    preprocessed_df = preprocess_asas_data(
        cohort_data=args.cohort_data,
        glucose_data=args.glucose_data,
        output_csv=args.output_csv,
        scale_income=args.scale_income,
        impute_missing=not args.no_impute,
        drop_drug_missing_baseline=not args.keep_drug_missing_baseline,
        show_progress=not args.no_progress,
    )

    print("Preprocessing completed.")
    print(f"Output file: {args.output_csv}")
    print(f"Output shape: {preprocessed_df.shape[0]} rows x {preprocessed_df.shape[1]} columns")


if __name__ == "__main__":
    main()
