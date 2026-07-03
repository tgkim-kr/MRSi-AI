import argparse
import sys
from pathlib import Path

# Add project root to Python path so that model_learning can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model_learning import run_experiment


def main():
    """
    Command-line entry point for running model training experiments.

    Usage example
    -------------
    python scripts/run_training.py \
        --input-csv outputs/preprocessed_data.csv \
        --sampling over \
        --features all \
        --pca plus \
        --models XGB LR RF LGBM ANN RNN \
        --output-dir result

    Equivalent legacy-style example
    -------------------------------
    python scripts/run_training.py \
        --csv-path outputs/preprocessed_data.csv \
        --output-json result/result_summary.json \
        --result-root result
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run model training experiments with selected feature sets, "
            "PCA modes, sampling strategies, and models."
        )
    )

    parser.add_argument(
        "--csv-path",
        "--input-csv",
        dest="csv_path",
        type=str,
        required=True,
        help="Path to the preprocessed CSV file.",
    )

    parser.add_argument(
        "--sampling",
        type=str,
        default="over",
        choices=["over", "none"],
        help="Sampling strategy. Use 'over' for oversampling or 'none' for no resampling.",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        choices=["all", "bio", "physical", "life", "Non_invasive"],
        help=(
            "Feature set(s) to use. "
            "If omitted, all predefined feature sets are evaluated."
        ),
    )

    parser.add_argument(
        "--pca",
        nargs="+",
        default=None,
        choices=["plus", "none", "only"],
        help=(
            "PCA mode(s) to use: "
            "'plus' = raw features + PCA features, "
            "'none' = raw features only, "
            "'only' = PCA features only. "
            "If omitted, all PCA modes are evaluated."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=["LR", "RF", "XGB", "LGBM", "ANN", "RNN"],
        help=(
            "Model(s) to train. "
            "If omitted, all predefined models are evaluated."
        ),
    )

    parser.add_argument(
        "--output-dir",
        "--result-root",
        dest="result_root",
        type=str,
        default="result",
        help="Base directory to store model artifacts.",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "Path to the JSON summary output file. "
            "If omitted, it will be saved as <output-dir>/result_summary.json."
        ),
    )

    args = parser.parse_args()

    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)

    if args.output_json is None:
        output_json_path = result_root / "result_summary.json"
    else:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

    run_experiment(
        csv_path=args.csv_path,
        output_json_path=str(output_json_path),
        result_root=str(result_root),
        feature_sets=args.features,
        pca_modes=args.pca,
        models_to_run=args.models,
        sampling=args.sampling,
    )


if __name__ == "__main__":
    main()
