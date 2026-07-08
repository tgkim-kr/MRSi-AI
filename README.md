# MRSi-AI (Multi-domain Risk Simulation-AI)

This repository provides the source code for a machine learning framework for predicting **incident type 2 diabetes (T2D)** using longitudinal cohort data.

The framework jointly models:

- baseline lifestyle, anthropometric, and biochemical variables, and
- **2-year interval changes** in these variables,

to capture dynamic risk trajectories rather than static snapshots.

The pipeline trains and compares multiple models:

- Logistic Regression (LR)
- Random Forest (RF)
- XGBoost (XGB)
- LightGBM (LGBM)
- Artificial Neural Network (ANN)
- SimpleRNN-based model (RNN)
- Ensemble model based on averaged predicted probabilities

Principal component analysis (PCA) is used as an optional feature transformation, and SHAP is used for model interpretability for supported tree-based models.

---

## 1. Background (summary of the study)

Most T2D prediction models rely on **single baseline measurements** and often use diagnostic glycemic markers such as fasting glucose or HbA1c. This creates two limitations:

1. They cannot fully capture **temporal evolution of risk**, and
2. Including diagnostic markers can lead to **circular reasoning** by detecting early disease rather than predicting future onset.

In this framework:

- Static baseline features are combined with **interval changes** in lifestyle, anthropometric, and biochemical markers.
- Diagnostic glycemic markers and related insulin/glucose variables are excluded from model inputs during the model-learning step.
- Tree-based models, logistic regression, and neural networks are compared under a unified experimental design.
- SHAP analysis can be used to inspect feature contributions for supported tree-based models.

By explicitly modeling **dynamic changes**, the framework aims to predict future T2D risk and support preventive intervention before diagnostic thresholds are reached.

---

## 2. Repository structure

```text
MRSi-AI/
├─ data_preprocessing/
│  ├─ __init__.py
│  └─ preprocessing.py
├─ model_learning/
│  ├─ __init__.py
│  ├─ feature_engineering.py
│  ├─ utils.py
│  ├─ models.py
│  └─ experiment.py
├─ scripts/
│  ├─ run_preprocessing.py
│  └─ run_training.py
├─ requirements.txt
└─ README.md
```

Module roles:

- `data_preprocessing/preprocessing.py`: converts raw longitudinal cohort files into interval-level examples.
- `model_learning/feature_engineering.py`: feature-set selection, PCA feature construction, and participant-level train/test/validation splitting.
- `model_learning/utils.py`: shared metrics, plotting, JSON serialization, and filesystem utilities.
- `model_learning/models.py`: LR, RF, XGB, LGBM, ANN, and RNN training functions.
- `model_learning/experiment.py`: experiment orchestration across feature groups, PCA modes, models, and ensemble evaluation.
- `scripts/run_preprocessing.py`: command-line wrapper for preprocessing.
- `scripts/run_training.py`: command-line wrapper for model training and evaluation.

---

## 3. Data availability and reproducibility notice

The original cohort data used in this study are **not included** in this repository due to data-use restrictions and privacy considerations.

The repository does not distribute the original individual-level data, dummy data, original variable names, or detailed input schema. Full end-to-end execution of the preprocessing and model-learning pipeline therefore requires authorized access to the original study data and the corresponding data specification.

This repository is intended to document and archive the source code used for preprocessing, feature engineering, data splitting, model training, and model evaluation. The public repository alone is not a fully executable reproduction package for users without access to the restricted study data.

---

## 4. Learning pipeline

The workflow is:

- **Preprocessing**
  - Load cohort-level longitudinal data and glucose/insulin measurement data.
  - Merge input files by participant ID.
  - Normalize column names.
  - Carry forward visit-level missing values encoded with the study-specific missing sentinel.
  - Generate 2-year interval examples.
  - Construct baseline and interval-change features.
  - Generate `diag` and `drug` analysis rows.
  - Optionally perform final statistical imputation.

- **Model learning**
  - Load the preprocessed interval-level CSV.
  - Optionally filter by `data_type`, such as `diag` or `drug`.
  - Drop diagnostic glycemic markers and related insulin/glucose variables by default.
  - Apply selected feature group and PCA mode.
  - Split data into train/test/validation sets at the participant level using `rid`.
  - Train LR, RF, XGB, LGBM, ANN, and/or RNN models.
  - Add an ensemble model based on averaged validation probabilities.
  - Save metrics, plots, selected hyperparameters, model artifacts, and summary files.

<img width="5005" height="5310" alt="flow" src="https://github.com/user-attachments/assets/3131755e-eb8b-4be8-bcbd-0e40b3fe1bce" />

---

## 5. Requirements

### 5.1. Python

- Tested on **Python 3.10**

The package versions in `requirements.txt` should be treated as the tested environment. Do not downgrade or upgrade pinned packages unless the full pipeline is re-tested.

### 5.2. System dependency: Graphviz

To export XGBoost tree visualizations, the **Graphviz** system binary (`dot`) is required in addition to the Python `graphviz` package.

On Windows:

1. Download Graphviz from: https://graphviz.org/download/
2. During installation, enable “Add Graphviz to the system PATH” or the equivalent option.
3. Confirm installation:

```bash
dot -V
```

If `dot -V` is not recognized, add the Graphviz `bin` directory to the system `PATH` manually.

### 5.3. Python packages

All Python dependencies are listed in `requirements.txt`.

Core libraries include:

- `numpy`, `pandas`, `scipy`
- `scikit-learn`, `joblib`
- `xgboost`, `lightgbm`
- `tensorflow` for ANN/RNN models
- `shap` for model interpretability
- `matplotlib`, `seaborn`, `tqdm`
- `sas7bdat`, `openpyxl`
- `graphviz`

Install with:

```bash
pip install -r requirements.txt
```

---

## 6. Installation

Clone the repository:

```bash
git clone https://github.com/tgkim-kr/MRSi-AI.git
cd MRSi-AI
```

Create or activate a Python 3.10 environment, then install dependencies:

```bash
pip install -r requirements.txt
```

---

## 7. Data preprocessing

The preprocessing step converts the original cohort data into an interval-level model-learning dataset combining baseline values and longitudinal changes.

### 7.1. Inputs

The preprocessing script expects authorized access to:

- Main cohort file: `.csv` or `.sas7bdat`
- Glucose/insulin measurement file: `.xlsx`

The original data and detailed schema are not distributed with this repository.

### 7.2. Example command

```bash
python scripts/run_preprocessing.py \
  --cohort-data input_data/cohort_data_example.csv \
  --glucose-data input_data/glucose_data_example.xlsx \
  --output-csv outputs/preprocessed_data.csv
```

The `input_data/` directory contains synthetic example input files for testing the pipeline:

- `input_data/cohort_data_example.csv`
- `input_data/glucose_data_example.xlsx`

These files are dummy data and do not contain real participant-level information. They are provided only to demonstrate the expected input format and to allow users to test whether the preprocessing and model-learning scripts run correctly.

The original cohort data used in the study are not publicly distributed due to data-use restrictions and privacy considerations.

### 7.3. Optional preprocessing arguments

```bash
--scale-income
```

Divide raw `V1_INCOME`-`V10_INCOME` values by 10,000 before preprocessing. Use this only when the raw income variables have not already been scaled.

```bash
--no-impute
```

Disable final imputation in the interval-level dataframe. This is useful if imputation will be performed after train/test/validation splitting.

```bash
--keep-drug-missing-baseline
```

Keep medication-outcome rows with missing baseline variables that are otherwise dropped in the original preprocessing workflow.

```bash
--no-progress
```

Disable progress bars.

### 7.4. Preprocessing output

The preprocessing script saves an interval-level CSV file containing rows with a `data_type` column. The main values are:

- `diag`: diagnosis-based outcome rows
- `drug`: medication-based outcome rows

Example output:

```text
outputs/preprocessed_data.csv
```

---

## 8. Model training

The training pipeline evaluates multiple model families with consistent feature selection, PCA construction, participant-level splitting, and validation evaluation.

### 8.1. Supported models

- Logistic Regression: `LR`
- Random Forest: `RF`
- XGBoost: `XGB`
- LightGBM: `LGBM`
- Artificial Neural Network: `ANN`
- SimpleRNN-based model: `RNN`
- Ensemble: automatically added after selected models are trained

ANN and RNN are executed sequentially in the main process. The remaining models can be executed in parallel through `joblib`.

### 8.2. Feature groups

Use the `--feature-sets` argument with one or more of the following values:

- `all`
- `bio` — biochemical markers
- `physical` — anthropometric and blood-pressure variables
- `life` — lifestyle and socioeconomic variables
- `Non_invasive` — physical + lifestyle features, excluding biochemical markers

### 8.3. PCA modes

Use the `--pca-modes` argument with one or more of the following values:

- `none` — no PCA
- `plus` — add PCA components in addition to raw features
- `only` — replace selected feature groups with PCA components

### 8.4. Sampling options

Use the `--sampling` argument:

- `over` — oversample positive cases within each split
- `none` — no resampling

The split is performed at the participant level using the `rid` column to prevent the same participant from appearing in more than one split. The default split ratio is 80% train, 10% test, and 10% validation.

### 8.5. Example command

```bash
python scripts/run_training.py \
  --input-csv outputs/preprocessed_data.csv \
  --output-dir result \
  --data-type diag \
  --sampling over \
  --feature-sets all bio physical life Non_invasive \
  --pca-modes plus none only \
  --models LR RF XGB LGBM ANN RNN \
  --n-jobs 3
```

To run only classical machine-learning models:

```bash
python scripts/run_training.py \
  --input-csv outputs/preprocessed_data.csv \
  --output-dir result \
  --models LR RF XGB LGBM \
  --n-jobs 3
```

### 8.6. Additional training arguments

```bash
--data-type diag
```

Use rows where `data_type == "diag"`. This is the default.

```bash
--data-type drug
```

Use medication-based rows.

```bash
--no-data-type-filter
```

Use all rows without filtering by `data_type`.

```bash
--drop-contains ins glu hba1c
```

Drop columns whose names contain the listed substrings before model training. This is the default and is used to remove diagnostic glycemic markers and related insulin/glucose variables from model inputs.

To disable this removal, pass the flag with no values:

```bash
python scripts/run_training.py \
  --input-csv outputs/preprocessed_data.csv \
  --drop-contains
```

```bash
--no-shap
```

Disable SHAP analysis.

```bash
--no-plots
```

Disable ROC and feature-importance plot generation.

```bash
--save-xgb-trees
```

Save XGBoost tree visualizations. This can be slow and requires both the Python `graphviz` package and the system Graphviz binary.

```bash
--result-json path/to/result.json
```

Set a custom path for the nested result JSON file. If omitted, the default is:

```text
result/result.json
```

```bash
--summary-csv path/to/auc_summary.csv
```

Set a custom path for the validation-AUC summary CSV file. If omitted, the default is:

```text
result/auc_summary.csv
```

---

## 9. Outputs

For each combination of feature group, PCA mode, and model, a structured directory is created under `result/`.

Example layout:

```text
result/
└─ all/
   └─ plus/
      ├─ LR/
      │  ├─ lr_model.pkl
      │  ├─ parameter_search.csv
      │  └─ ROC_curve.jpg
      ├─ RF/
      │  ├─ rf_model.pkl
      │  ├─ parameter_search.csv
      │  ├─ ROC_curve.jpg
      │  └─ feature_importance.jpg
      ├─ XGB/
      │  ├─ xgb_model.pkl
      │  ├─ parameter_search.csv
      │  ├─ ROC_curve.jpg
      │  ├─ feature_importance.jpg
      │  ├─ shap_summary_plot.png
      │  ├─ trees/
      │  │  ├─ all__plus__XGB_tree_0.png
      │  │  └─ ...
      │  └─ trees_txt/
      │     ├─ all__plus__XGB_tree_0.txt
      │     └─ ...
      ├─ LGBM/
      │  ├─ lgbm_model.pkl
      │  ├─ parameter_search.csv
      │  ├─ ROC_curve.jpg
      │  ├─ feature_importance.jpg
      │  ├─ lgbm_tree.png
      │  └─ shap_summary_plot.png
      ├─ ANN/
      │  ├─ ann_model.keras
      │  ├─ parameter_search.csv
      │  └─ ROC_curve.jpg
      ├─ RNN/
      │  ├─ rnn_model.keras
      │  ├─ parameter_search.csv
      │  └─ ROC_curve.jpg
      └─ Ensemble/
          └─ ROC_curve.jpg
```

The training script also saves summary files:

```text
result/
├─ result.json
└─ auc_summary.csv
```

`result.json` contains the nested model results, including:

- Test AUROC
- Validation AUROC
- F1-optimized cutoff and associated metrics
- Feature importance, where available
- Selected hyperparameters

`auc_summary.csv` contains a flattened validation-AUC ranking across feature groups, PCA modes, and models.

---

## 10. Python API usage

The command-line scripts are thin wrappers around importable Python functions.

Preprocessing:

```python
from data_preprocessing import preprocess_asas_data

preprocessed_df = preprocess_asas_data(
    cohort_data="path/to/cohort_data.sas7bdat",
    glucose_data="path/to/glucose_data.xlsx",
    output_csv="outputs/preprocessed_data.csv",
)
```

Model learning:

```python
from model_learning import run_experiment

result = run_experiment(
    input_csv="outputs/preprocessed_data.csv",
    output_dir="result",
    data_type="diag",
    sampling_method="over",
    feature_sets=("all", "bio", "physical", "life", "Non_invasive"),
    pca_modes=("plus", "none", "only"),
    cpu_models=("LR", "RF", "XGB", "LGBM"),
    gpu_models=("ANN", "RNN"),
    n_jobs=3,
)
```

---

## 11. Notes and limitations

- The original cohort data are not distributed with this repository.
- Full end-to-end reproduction requires authorized access to the original data and data specification.
- The default preprocessing behavior reproduces the original notebook-style missing-value handling. For leakage-free imputation workflows, use `--no-impute` and perform imputation after participant-level splitting.
- The model-learning pipeline removes columns containing `ins`, `glu`, or `hba1c` by default.
- XGBoost tree visualization can be slow and is disabled by default. Use `--save-xgb-trees` only when tree images are required.
- SHAP analysis can increase runtime. Use `--no-shap` to disable it.
