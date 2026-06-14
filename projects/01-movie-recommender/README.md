# Movie Recommendation System

Built a movie recommendation system on the **MovieLens** dataset (100K ratings,
610 users, 9,742 movies), progressing through three modeling approaches and a
full evaluation pipeline. Demonstrates the end-to-end ML workflow: exploratory
analysis, feature engineering, model development, hyperparameter tuning, and
metric-driven comparison.

**Tech stack:** Python · pandas · scikit-learn · scikit-surprise · NumPy · Matplotlib · Seaborn

---

## Notebooks (what we built, step by step)

| Stage | What you'll learn | Notebook | Status |
|-------|-------------------|----------|--------|
| 1. Exploration (EDA) | Understand the data: who rates what, how ratings are distributed | `01_exploration.ipynb` | ✅ Done |
| 2. Popularity baseline | The simplest recommender — "most popular movies" — as a benchmark | `02_baseline.ipynb` | ✅ Done |
| 3. Content-based filtering | Recommend movies similar to ones you liked (by genre/tags) | `03_content_based.ipynb` | ✅ Done |
| 4. Collaborative filtering | "Users like you also liked..." using matrix factorization (SVD) | `04_collaborative_filtering.ipynb` | ✅ Done |
| 5. Evaluation | Measure quality with RMSE, precision@k, and discuss trade-offs | `05_evaluation.ipynb` | ✅ Done |

Each notebook explains *why*, not just *how* — written for someone learning the concepts from scratch.

---

## Get the data

This project pulls data from Kaggle using the Kaggle API. **One-time setup:**

### 1. Create a Kaggle API token
1. Sign in (or sign up) at [kaggle.com](https://www.kaggle.com).
2. Click your profile picture → **Settings**.
3. Scroll to the **API** section → click **Create New Token**.
4. This downloads a file called `kaggle.json` to your Downloads folder.

### 2. Install the token where the CLI expects it
Run this in your terminal (it moves the token and locks down its permissions):

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

> ⚠️ Never commit `kaggle.json` to git — it's your private key. Our `.gitignore`
> already blocks it, but keep it out of project folders to be safe.

### 3. Download the dataset
From the workspace root, with the venv active:

```bash
source .venv/bin/activate
python projects/01-movie-recommender/src/download_data.py
```

You should end up with `movies.csv`, `ratings.csv`, `tags.csv`, and `links.csv`
inside `data/raw/`.

---

## How to run

```bash
# from the workspace root
source .venv/bin/activate
jupyter lab
```

Open any notebook under `projects/01-movie-recommender/notebooks/` and select
the **"Python (ML Workspace)"** kernel (top-right in Jupyter). Run them in order:
`01_exploration` → `02_baseline` → `03_content_based` → `04_collaborative_filtering` → `05_evaluation`.

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

*Best SVD hyperparameters found by grid search: n_factors=50, reg_all=0.05*

### Key findings

**SVD cuts RMSE by 16.2%** over the global-mean baseline — it learns each user's
tendency to rate high or low, and each movie's tendency to receive high or low
scores, producing far more accurate rating predictions.

**Precision@10 tells a more nuanced story.** The popularity baseline scores highest
(0.122) because the most-rated blockbusters appear in almost every user's test set
by definition. SVD recommends *personalized* titles a user is likely to love, but
since the test set only contains movies the user happened to rate during evaluation,
niche personal recommendations are harder to "hit" in this metric. This is a known
limitation of offline Precision@K evaluation — online A/B tests would tell a
different story.

**Content-based (genre-only) underperforms.** Genres are too coarse — hundreds of
movies share identical genre combinations, so cosine similarity can't separate them.
Richer features (plot keywords, cast, decade) would close this gap significantly.

**Cold-start is real.** SVD's prediction error is highest for movies with fewer than
20 training ratings (mean absolute error 0.66–0.76 vs 0.63 for popular movies).
A hybrid model combining SVD with content features would address this.

### Skills demonstrated

- **Exploratory data analysis** — distribution analysis, long-tail patterns, sparsity calculation
- **Feature engineering** — genre binarization with `MultiLabelBinarizer`, cosine similarity scoring, user taste profiles
- **Model development** — popularity baseline, content-based filtering, SVD matrix factorization
- **Hyperparameter tuning** — grid search with 3-fold cross-validation via `scikit-surprise`
- **Evaluation** — RMSE, Precision@K, cross-validation, error analysis by item popularity
- **Visualization** — publication-quality charts with Matplotlib/Seaborn; automated PowerPoint report

### What's next (if continuing this project)

- Add plot-keyword and cast features to the content-based model
- Try a hybrid SVD + content approach for cold-start users
- Implement online evaluation (simulated A/B test) for a fairer Precision@K
- Deploy a simple Streamlit demo that lets you pick a user and see live recommendations
