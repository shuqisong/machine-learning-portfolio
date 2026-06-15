# Movie Recommendation System

Built a movie recommendation system on the **MovieLens** dataset (100K ratings,
610 users, 9,742 movies), progressing through three modeling approaches and a
full evaluation pipeline. Demonstrates the end-to-end ML workflow: exploratory
analysis, feature engineering, model development, hyperparameter tuning, and
metric-driven comparison.

**Tech stack:** Python · pandas · scikit-learn · scikit-surprise · NumPy · Matplotlib · Seaborn

---

## Project structure

```
projects/01-movie-recommender/
├── notebooks/                         # run in order
│   ├── 01_exploration.ipynb           # EDA — distributions, sparsity, long-tail
│   ├── 02_baseline.ipynb              # popularity baseline + RMSE / Precision@K
│   ├── 03_content_based.ipynb         # genre-vector similarity recommender
│   ├── 04_collaborative_filtering.ipynb  # SVD matrix factorization
│   └── 05_evaluation.ipynb            # grid search tuning + final comparison
├── src/
│   └── download_data.py               # fetches raw data from Kaggle
├── data/
│   └── raw/                           # CSVs downloaded here (git-ignored)
└── reports/figures/                   # result charts
```

---

## Notebooks (what we built, step by step)

| Stage | Topic | Notebook |
|-------|-------|----------|
| 1 | Exploratory data analysis | `01_exploration.ipynb` |
| 2 | Popularity baseline | `02_baseline.ipynb` |
| 3 | Content-based filtering (genres) | `03_content_based.ipynb` |
| 4 | Collaborative filtering (SVD) | `04_collaborative_filtering.ipynb` |
| 5 | Hyperparameter tuning & final evaluation | `05_evaluation.ipynb` |

Each notebook explains *why*, not just *how* — written for someone learning the concepts from scratch.

---

## Setup

### 1. Create and activate a virtual environment

```bash
# from the workspace root (machine-learning-portfolio/)
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn scikit-surprise matplotlib seaborn jupyterlab kaggle
```

### 3. Get the data (one-time)

This project uses the [MovieLens dataset](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset) from Kaggle.

**Create a Kaggle API token:**
1. Sign in at [kaggle.com](https://www.kaggle.com) → profile picture → **Settings**
2. Scroll to **API** → click **Create New Token** → downloads `kaggle.json`
3. Move it into place:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json   # keep it private
```

**Download the dataset:**

```bash
python projects/01-movie-recommender/src/download_data.py
```

This populates `data/raw/` with `movies.csv`, `ratings.csv`, `tags.csv`, and `links.csv`.

### 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name ml-workspace --display-name "Python (ML Workspace)"
```

### 5. Launch Jupyter

```bash
jupyter lab
```

Open the notebooks under `projects/01-movie-recommender/notebooks/`, select the
**"Python (ML Workspace)"** kernel, and run them in order 01 → 05.

---

## Results

### Dataset

100,836 ratings from 610 users across 9,742 movies (MovieLens small, latest).
Ratings run 0.5–5.0 stars; the data is **98.3% sparse** — the central challenge
every recommender must overcome.

### Model comparison

| Model | RMSE ↓ | Precision@10 ↑ | Personalized |
|---|---|---|---|
| Popularity baseline (global mean) | 1.0488 | 0.1220 | No |
| Content-based (genre vectors) | — | 0.0053 | Yes |
| SVD collaborative filtering (default) | 0.8831 | 0.0479 | Yes |
| **SVD collaborative filtering (tuned)** | **0.8794** | 0.0411 | Yes |

*Best SVD hyperparameters (grid search, 3-fold CV): n_factors=50, reg_all=0.05*

### Key findings

**SVD cuts RMSE by 16.2%** over the global-mean baseline — it learns each user's
tendency to rate high or low, and each movie's tendency to receive high or low
scores, producing far more accurate rating predictions.

**Precision@10 tells a more nuanced story.** The popularity baseline scores highest
(0.122) because blockbusters appear in almost every user's test set by definition.
SVD recommends personalized titles a user is likely to love, but niche personal
recommendations are harder to "hit" in an offline test set — this is a known
limitation of offline Precision@K evaluation; online A/B tests would tell a
different story.

**Content-based (genre-only) underperforms.** Genres are too coarse — hundreds of
movies share identical genre combinations, so cosine similarity can't distinguish
between them. Richer features (plot keywords, cast, decade) would close this gap.

**Cold-start is real.** SVD's prediction error is highest for movies with fewer than
20 training ratings (MAE 0.66–0.76 vs 0.63 for popular movies). A hybrid model
combining SVD with content features would address this.

### Skills demonstrated

- **Exploratory data analysis** — distribution analysis, long-tail patterns, sparsity calculation
- **Feature engineering** — genre binarization with `MultiLabelBinarizer`, cosine similarity scoring, user taste profiles
- **Model development** — popularity baseline, content-based filtering, SVD matrix factorization
- **Hyperparameter tuning** — grid search with 3-fold cross-validation via `scikit-surprise`
- **Evaluation** — RMSE, Precision@K, cross-validation, error analysis by item popularity
- **Visualization** — result charts with Matplotlib/Seaborn

### What's next

- Add plot-keyword and cast features to the content-based model
- Try a hybrid SVD + content approach for cold-start users
- Implement online evaluation (simulated A/B test) for a fairer Precision@K
- Deploy a Streamlit demo that lets you pick a user and see live recommendations
