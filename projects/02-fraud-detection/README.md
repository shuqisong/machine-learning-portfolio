# Credit Card Fraud Detection

Built a fraud detection classifier on the **Credit Card Fraud Detection**
dataset (284,807 transactions, 0.17% fraud) — a classic, highly imbalanced
binary classification problem. Demonstrates the full ML workflow for imbalanced
data: EDA, resampling strategies, model comparison, threshold tuning, and
metrics that actually matter when the positive class is rare (PR-AUC,
precision/recall — not accuracy).

**Tech stack:** Python · pandas · scikit-learn · imbalanced-learn · NumPy · Matplotlib · Seaborn

---

## Project structure

```
projects/02-fraud-detection/
├── notebooks/                         # run in order
│   ├── 01_exploration.ipynb           # EDA — class imbalance, feature distributions
│   ├── 02_baseline.ipynb              # logistic regression baseline + metrics
│   ├── 03_resampling.ipynb            # SMOTE / undersampling / class weights
│   ├── 04_tree_models.ipynb           # random forest + histogram gradient boosting
│   └── 05_evaluation.ipynb            # threshold tuning + final comparison
├── src/
│   └── download_data.py               # fetches raw data from Kaggle
├── data/
│   └── raw/                           # CSV downloaded here (git-ignored)
└── reports/figures/                   # 11 result charts
```

---

## Roadmap

| Stage | Topic | Notebook | Status |
|-------|-------|----------|--------|
| 1 | Exploratory data analysis | `01_exploration.ipynb` | ✅ |
| 2 | Baseline model (logistic regression) | `02_baseline.ipynb` | ✅ |
| 3 | Handling class imbalance (SMOTE, class weights) | `03_resampling.ipynb` | ✅ |
| 4 | Tree-based models (Random Forest, Gradient Boosting) | `04_tree_models.ipynb` | ✅ |
| 5 | Threshold tuning & final evaluation | `05_evaluation.ipynb` | ✅ |

Each notebook explains *why*, not just *how* — written for someone learning the concepts from scratch.

---

## Setup

### 1. Activate the workspace environment

```bash
# from the workspace root
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the data (one-time)

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

**Create a Kaggle API token** (skip if already set up for another project):
1. Sign in at [kaggle.com](https://www.kaggle.com) → profile picture → **Settings**
2. Scroll to **API** → click **Create New Token** → downloads `kaggle.json`
3. Move it into place:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Download the dataset:**

```bash
python projects/02-fraud-detection/src/download_data.py
```

This populates `data/raw/` with `creditcard.csv`.

### 3. Launch Jupyter

```bash
jupyter lab
```

Open the notebooks under `projects/02-fraud-detection/notebooks/`, select the
**"Python (ML Workspace)"** kernel, and run them in order 01 → 05.

---

## Results

### Dataset

284,807 credit card transactions, 492 fraud cases (0.17%). 28 PCA-anonymized
features (V1–V28) plus `Time` and `Amount`. No missing values; 1,081 exact
duplicate rows removed before modeling.

### Model progression

| Stage | Model | PR-AUC ↑ | Recall | False alarms |
|---|---|---|---|---|
| 2 | Logistic regression (baseline) | 0.6935 | 0.59 | 10 |
| 3 | LR + SMOTE/undersampling | 0.6769 | 0.87 | ~1,400 |
| 4 | Random Forest + SMOTE | 0.8135 | 0.76 | 7 |
| **5** | **RF + SMOTE, threshold tuned (t=0.41)** | **0.8135** | **0.79** | **11** |

### Key findings

**Non-linear models are essential.** Switching from logistic regression to
Random Forest raised PR-AUC from 0.69 to 0.81 — an 18% relative improvement.
The PCA-anonymized V features have complex non-linear structure that a linear
hyperplane cannot capture.

**SMOTE pays off more with tree models.** With logistic regression, SMOTE
produced no meaningful gain over class weighting. With Random Forest, it
pushed PR-AUC from 0.79 (default RF) to 0.81. The richer non-linear decision
boundary fully exploits the synthetic minority samples.

**Threshold choice is a business decision.** The default 0.5 threshold already
happens to be near-optimal for F1 (0.828). Lowering to 0.3 catches 5 more
fraud cases (77 vs 72) but adds 16 more false alarms. Under a cost model where
missed fraud costs 25× a false alarm, the optimal threshold drops below 0.2 —
capturing ~84% of fraud is worth absorbing the extra false alarms.

**V14 dominates feature importance** (~19% of split decisions), consistent
with Stage 1 EDA where V14 showed the largest class mean separation.

### Skills demonstrated

- **Imbalanced classification** — accuracy paradox, PR-AUC vs ROC-AUC, SMOTE, undersampling, class weights
- **Model development** — logistic regression baseline, Random Forest, Histogram Gradient Boosting
- **Threshold tuning** — sweeping thresholds, cost-based optimisation, precision-recall tradeoff
- **Feature importance** — tree model importances validated against EDA findings
- **Data leakage prevention** — stratified splits, fit-on-train-only scaling, SMOTE applied to training only

### What's next

- Try a calibrated probability output (Platt scaling / isotonic regression) for the HistGBM model
- Add cross-validation to confirm test-set results aren't lucky splits
- Build a Streamlit demo with an adjustable fraud threshold slider
