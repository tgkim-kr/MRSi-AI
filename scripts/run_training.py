import argparse
from model_learning import run_experiment


def main():
    """
    Command-line entry point for running model training experiments.
    """
    parser = argparse.ArgumentParser(
        description="Run model training experiments with multiple feature sets, PCA modes and models."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to the preprocessed CSV file.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="result_summary.json",
        help="Path to the JSON summary output file.",
    )
    parser.add_argument(
        "--result-root",
        type=str,
        default="result",
        help="Base directory to store model artifacts.",
    )
    args = parser.parse_args()

    run_experiment(
        csv_path=args.csv_path,
        output_json_path=args.output_json,
        result_root=args.result_root,
    )


if __name__ == "__main__":
    main()
