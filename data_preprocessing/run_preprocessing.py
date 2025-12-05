#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
from data_processing import run_full_preprocessing


def main():
    """
    Command-line entry point for running the full preprocessing pipeline.

    Usage example
    -------------
    asas-preprocess \
        --sas-path data_cohort.sas7bdat \
        --glu-path data_glucose.xlsx \
        --output ASAS_preprocessed_data_test.csv
    """
    parser = argparse.ArgumentParser(
        description="ASAS cohort preprocessing pipeline"
    )
    parser.add_argument(
        "--sas-path",
        type=str,
        required=True,
        help="Path to .sas7bdat file (e.g. data_cohort.sas7bdat)",
    )
    parser.add_argument(
        "--glu-path",
        type=str,
        required=True,
        help="Path to glucose .xlsx file (e.g. data_glucose.xlsx)",
    )
    parser.add_argument(
        "--output",
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

