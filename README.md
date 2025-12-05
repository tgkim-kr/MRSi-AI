# MRSi-AI

This repository provides a machine learning framework for predicting **incident type 2 diabetes (T2D)** using longitudinal cohort data.  
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
- (Optional) ensemble of the above

Principal component analysis (PCA) is used to address multicollinearity, and SHAP is used for model interpretability.

---

## 1. Background (summary of the study)

Most T2D prediction models rely on **single baseline measurements** and often use diagnostic glycemic markers such as fasting glucose or HbA1c.  
This creates two limitations:

1. They cannot fully capture **temporal evolution of risk**, and  
2. Including diagnostic markers can lead to **circular reasoning** (detecting early disease rather than predicting future onset).

In this framework:

- We combine static baseline features with **interval changes** in lifestyle, anthropometric and biochemical markers.
- Diagnostic glycemic markers (e.g., fasting glucose, HbA1c) are **excluded** from model inputs.
- Tree-based models (XGBoost, LGBM, Random Forest), logistic regression, and neural networks are compared under a unified experimental design.
- SHAP analysis shows that change variables such as ▲CRP and ▲BMI, together with baseline triglycerides (TG), are among the most important predictors.

By explicitly modeling **dynamic changes**, the framework aims to predict true future risk and support preventive intervention before diagnostic thresholds are reached.

---

## 2. Learning–Inference pipeline

The workflow is:

- **Learning phase**
  - Preprocessing
    - Feature generation  
    - Data imputation  
    - Feature clustering / grouping  
    - PCA (optional)
  - Model training
    - LR, RF, XGB, LGBM, ANN, RNN
    - Ensemble of base models
  - Selection of the **best-performing model**

- **Inference phase**
  - Apply the trained model to real-world population data
  - Use lifestyle, anthropometric, and biochemical features
  - Obtain diabetes risk predictions and select the **best simulation scenario**

<img width="5005" height="5310" alt="flow" src="https://github.com/user-attachments/assets/3131755e-eb8b-4be8-bcbd-0e40b3fe1bce" />

---

## 3. Requirements

### 3.1. Python

- Tested on **Python 3.11**

### 3.2. System dependency: Graphviz

To export decision trees from XGBoost / LightGBM as images, the **Graphviz** system binary (`dot`) is required.


- **Windows**

  1. Download Graphviz from: https://graphviz.org/download/  
  2. During installation, make sure “Add Graphviz to the system PATH” (or equivalent) is enabled.  
  3. Confirm installation:


### 3.3. Python packages

All Python dependencies are listed in `requirements.txt`.

Core libraries include:

- `numpy`, `pandas`, `scipy`  
- `scikit-learn`  
- `xgboost`, `lightgbm`  
- `tensorflow` (for ANN / RNN)  
- `shap`  
- `matplotlib`, `seaborn`, `tqdm`  
- `sas7bdat`

Install with:

```bash
pip install -r requirements.txt
```

---

## 4. Installation

### 4.1. Clone the repository

```bash
git clone https://github.com/tgkim-kr/MRSi-AI.git
cd MRSi-AI
```


## 5. Data preprocessing

The preprocessing step converts the original cohort data into a model learning data combining baseline values and longitudinal changes.

**Inputs**

- Main cohort file (e.g. `*.sas7bdat`)  
- Glucose / insulin measurements (e.g. `*.xlsx`)

**Example**

```bash
python scripts/run_preprocessing.py \
  --input-sas path/to/cohort_data.sas7bdat \
  --input-glucose path/to/glucose_data.xlsx \
  --output-csv outputs/preprocessed_data.csv
```

This script:

1. Loads SAS and Excel files.  
2. Merges them by participant ID.  
3. Normalizes column names and data types.  
4. Generates 2-year interval change features for lifestyle, anthropometric and biochemical markers.  
5. Performs longitudinal forward filling and statistical imputation for missing values.  
6. Drops diagnostic glycemic markers (fasting glucose, HbA1c, related insulin/glucose variables).  
7. Saves the final dataset to `outputs/preprocessed_data.csv`.

---

## 6. Model training

The training pipeline evaluates multiple model families with consistent preprocessing and splitting.

**Supported models**

- Logistic Regression (**LR**)  
- Random Forest (**RF**)  
- XGBoost (**XGB**)  
- LightGBM (**LGBM**)  
- Artificial Neural Network (**ANN**)  
- Recurrent Neural Network (**RNN**)

**Feature groups**

- `all`  
- `bio` (biochemical markers)  
- `physical` (anthropometric measures)  
- `life` (lifestyle & socioeconomic)  
- `Non_invasive` (non-invasive subset)

**PCA options**

- `none` – no PCA  
- `plus` – add PCA components in addition to raw features  
- `only` – replace selected feature groups with PCA components only  

**Sampling options**

- `over` – oversample positive (label=1) cases  
- `none` – no resampling  

### 6.1. Example command

```bash
python scripts/run_training.py \
  --input-csv outputs/preprocessed_data.csv \
  --sampling over \
  --features all \
  --pca plus \
  --models XGB LR RF LGBM ANN RNN \
  --output-dir result
```

This script:

1. Applies the selected feature group and PCA setting.  
2. Splits the data into train / test / validation sets.  
3. Trains each specified model with predefined hyperparameter grids.  
4. Computes AUROC and an F1-optimized cutoff on the validation set.  
5. Saves metrics, plots, and model artifacts under `result/`.

---

## 7. Outputs

For each combination of feature group, PCA setting, and model, a structured directory is created under `result/`.

**Example layout**

```text
result/
└─ all/
   └─ plus/
      ├─ XGB/
      │  ├─ ROC_curve.jpg
      │  ├─ feature_importance.jpg
      │  ├─ shap_summary_plot.png
      │  ├─ trees/
      │  │  ├─ tree_0.png
      │  │  └─ ...
      │  └─ trees_txt/
      │     ├─ tree_0.txt
      │     └─ ...
      ├─ LGBM/
      │  ├─ ROC_curve.jpg
      │  ├─ feature_importance.jpg
      │  └─ shap_summary_plot.png
      ├─ RF/
      │  ├─ ROC_curve.jpg
      │  └─ feature_importance.jpg
      ├─ LR/
      │  └─ ROC_curve.jpg
      ├─ ANN/
      │  └─ ROC_curve.jpg
      └─ RNN/
         └─ ROC_curve.jpg
```

You may also include a summary JSON file such as:

```text
result/
└─ metrics_summary.json
```

Each entry can contain:

- Test AUROC  
- Validation AUROC  
- Best cutoff and associated metrics (precision, recall, F1, accuracy)  
- Feature importance (if applicable)  
- Selected hyperparameters  


