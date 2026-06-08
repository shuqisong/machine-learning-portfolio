# 🎬 Movie Recommender System

A recommendation system built on the **MovieLens** dataset — the classic
dataset for learning how recommenders work. The goal: given what a user has
watched and rated, suggest movies they're likely to enjoy.

This is a **resume-worthy** project because recommender systems power Netflix,
Spotify, Amazon, and YouTube — and this project demonstrates the full workflow:
data cleaning, exploratory analysis, multiple modeling approaches, and evaluation.

---

## The plan (what we'll build, step by step)

| Stage | What you'll learn | Notebook |
|-------|-------------------|----------|
| 1. Exploration (EDA) | Understand the data: who rates what, how ratings are distributed | `01_exploration.ipynb` |
| 2. Popularity baseline | The simplest recommender — "most popular movies" — as a benchmark | *(coming next)* |
| 3. Content-based filtering | Recommend movies similar to ones you liked (by genre/tags) | *(coming next)* |
| 4. Collaborative filtering | "Users like you also liked..." using matrix factorization (SVD) | *(coming next)* |
| 5. Evaluation | Measure quality with RMSE, precision@k, and discuss trade-offs | *(coming next)* |

We'll build these one at a time — you're a beginner, so each stage comes with
explanations of *why*, not just *how*.

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

Open `projects/01-movie-recommender/notebooks/01_exploration.ipynb` and select
the **"Python (ML Workspace)"** kernel (top-right in Jupyter).

---

## Results

*(Filled in as we go — this is where you'll summarize what worked, with charts.
Recruiters read this section first, so we'll make it tell a clear story.)*
