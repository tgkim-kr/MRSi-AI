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


```md

