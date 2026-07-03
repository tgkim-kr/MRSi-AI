import argparse
import sys
from pathlib import Path

# Add project root to Python path so that data_preprocessing can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing import run_full_preprocessing


def main():
    """
    Command-line entry point for running the full preprocessing pipeline.

    Usage example
    -------------
    python scripts/run_preprocessing.py \
        --sas-path data_cohort.sas7bdat \
        --glu-path data_glucose.xlsx \
        --output preprocessed_data.csv

    Alternative argument names are also supported:
    python scripts/run_preprocessing.py \
        --input-sas data_cohort.sas7bdat \
        --input-glucose data_glucose.xlsx \
        --output-csv preprocessed_data.csv
    """
    parser = argparse.ArgumentParser(
        description="MRSi-AI cohort preprocessing pipeline"
    )

    parser.add_argument(
        "--sas-path",
        "--input-sas",
        dest="sas_path",
        type=str,
        required=True,
        help="Path to .sas7bdat file, e.g. data_cohort.sas7bdat",
    )

    parser.add_argument(
        "--glu-path",
        "--input-glucose",
        dest="glu_path",
        type=str,
        required=True,
        help="Path to glucose .xlsx file, e.g. data_glucose.xlsx",
    )

    parser.add_argument(
        "--output",
        "--output-csv",
        dest="output",
        type=str,
        default="preprocessed_data.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    run_full_preprocessing(
        sas_data_path=args.sas_path,
        glu_data_path=args.glu_path,
        output_csv_path=args.output,
    )


if __name__ == "__main__":
    main()
