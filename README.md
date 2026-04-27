# SVD vs. SVD + Louvain Re-ranking for Popularity-Diverse Movie Recommendations

**Course:** Data Mining and Applications 
**Dataset:** MovieLens 10M &nbsp;|&nbsp; **Python:** 3.8 &nbsp;|&nbsp; **Author:** Pranavi Pathakota

Recommender systems trained on popularity-skewed data tend to recommend the same blockbusters to everyone. This project asks: can we break that loop using graph structure? We build two pipelines on MovieLens 10M — a bias-corrected SVD baseline and an SVD + Louvain re-ranker that enforces community diversity — and measure the diversity–accuracy trade-off head-to-head.

---

## Quick Start

**[👉 Open main_notebook.ipynb](main_notebook.ipynb)** 
**[🎥 Watch on YouTube](https://www.youtube.com/watch?v=INDIs76RLnA)**

---

## Research Question

> Does applying Louvain community structure as a diversity re-ranker on top of SVD produce more popularity-diverse recommendations than SVD alone — and at what cost to ranking accuracy?

### Hypotheses

| | Prediction | Result |
|---|---|---|
| H-A | SVD+Louvain reduces `head_rec_frac` | ✅ Confirmed |
| H-B | Small drop in P@10 / nDCG@10 | ✅ Confirmed |
| H-C | SVD+Louvain improves tail P@10 | ✅ Marginal |
| H-D | Lower Gini exposure coefficient | ✅ Confirmed |

---

## Results Summary

SVD+Louvain **measurably reduces head-movie dominance** (H-A) while incurring only a **small precision cost** (H-B). The trade-off curve shows diminishing diversity returns past ~3 community slots — meaning most of the gain comes cheaply, and `max_per_comm=2` is a practical sweet spot for real applications. Full figures and quantitative table are in `main_notebook.ipynb`.

---

## Methods

| | Method A: SVD | Method B: SVD + Louvain |
|---|---|---|
| Rating model | Bias-corrected `scipy svds k=25` | **Identical** |
| Candidate pool | Top-100 by predicted score | Top-100 by predicted score |
| Final top-10 | Top-10 by score | Greedy community-diversity filter |
| Extra variable | — | Louvain community membership |

**Key insight:** SVD scores are never modified. The re-ranker only controls *which* candidates make the final list — walking down SVD-ranked candidates and picking at most one movie per Louvain community per pass.

### Greedy Re-ranking Algorithm

```python
def rerank_with_communities(svd_candidates, movie_community, k=10):
    selected, seen_comms, fallback = [], set(), []
    for movie in svd_candidates:
        comm = movie_community.get(movie)
        if comm is None or comm not in seen_comms:
            selected.append(movie)
            if comm is not None:
                seen_comms.add(comm)
        else:
            fallback.append(movie)
        if len(selected) == k:
            break
    for movie in fallback:
        if len(selected) == k:
            break
        selected.append(movie)
    return selected[:k]
```

---

## Repository Structure

```
.
├── main_notebook.ipynb              ← Final curated notebook (start here)
├── requirements.txt                 ← Python dependencies
├── README.md
├── .gitignore
├── checkpoints/
│   ├── checkpoint_1_EDA.ipynb       ← CP1: Exploratory Data Analysis
│   ├── checkpoint_1_analysis.ipynb  ← CP1: Extended analysis
│   └── checkpoint_2.ipynb           ← CP2: Research Question Formation
├── notebooks/
│   ├── svd_bias_analysis.ipynb      ← λ & k hyperparameter sweep
│   └── louvain_community_analysis.ipynb  ← Community diagnostics
├── assets/                          ← All generated figures (*.png)
└── data/
    └── ml-10M100K/                  ← MovieLens 10M (not committed — see below)
```

---

## Data

**Dataset:** [MovieLens 10M](https://grouplens.org/datasets/movielens/10m/) — 10M ratings by 71,567 users on 10,681 movies (Harper & Konstan, 2015).

| Stat | Value |
|------|-------|
| Ratings | ~10 million |
| Users | 71,567 |
| Movies | 10,681 |
| Sparsity | 99.96% |
| Mean rating | 3.51 |

The `data/` directory is not committed (too large). To reproduce:

```bash
# Download and unzip into the correct location
curl -O https://files.grouplens.org/datasets/movielens/ml-10m.zip
unzip ml-10m.zip -d data/
```

---

## How to Reproduce

This project runs locally on CPU — no GPU required.

```bash
# 1. Create environment
conda create -n dma python=3.8
conda activate dma

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data (see above)

# 4. Run the main notebook
jupyter notebook main_notebook.ipynb
```

Run notebooks in this order if starting from scratch:
1. `checkpoints/checkpoint_1_EDA.ipynb` — EDA
2. `checkpoints/checkpoint_2.ipynb` — RQ formation
3. `main_notebook.ipynb` — Full pipeline & results

> **Note:** Cell 4.2 (co-rating graph projection) takes 5–20 min on CPU. All other cells are fast.

---

## EDA Key Findings

- **99.96% sparsity** — most user–movie pairs are unobserved
- **Top 10% of movies receive 58% of all ratings** — strong popularity skew that motivates this project
- **Mean rating: 3.51** on a 0.5–5.0 scale
- **~90% of users have ≤ 5 ratings** — severe cold-start challenge

---

## Key Dependencies

| Package | Version |
|---------|---------|
| Python | 3.8.18 |
| numpy | 1.24.4 |
| pandas | 1.4.2 |
| scipy | 1.10.1 |
| scikit-learn | 1.1.1 |
| networkx | 2.8.8 |
| python-louvain | 0.16 |
| matplotlib | 3.7.5 |
| seaborn | 0.13.2 |

Full list: [`requirements.txt`](requirements.txt)

---

*Harper, F.M. & Konstan, J.A. (2015). The MovieLens Datasets. ACM TIIS.*
*Blondel et al. (2008). Fast unfolding of communities in large networks. J. Stat. Mech.*
