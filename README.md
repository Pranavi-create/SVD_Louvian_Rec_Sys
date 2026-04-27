# SVD vs SVD + Louvain Re-ranking for Popularity-Diverse Movie Recommendations

**Course:** Data Mining and Applications (DMA) — Stanford
**Dataset:** MovieLens 10M

Does applying Louvain community structure as a diversity re-ranker on top of SVD produce more popularity-diverse recommendations than SVD alone — and at what cost to ranking quality? This project builds two recommendation pipelines on MovieLens 10M, compares them head-to-head across popularity-diversity and accuracy metrics, and delivers a data-driven answer to that trade-off.

---

## 👉 Start here: [main_notebook.ipynb](main_notebook.ipynb)

---

## 🎥 Project Video

[YOUTUBE_LINK](https://www.youtube.com/watch?v=INDIs76RLnA)



## Research Questions

1. **RQ:** Does SVD + Louvain re-ranking reduce head-movie dominance (head_rec_frac) compared to SVD alone?
2. **Sub-questions:** What is the precision/nDCG trade-off of the diversity gain? Does SVD+Louvain surface more tail-movie communities without sacrificing too much relevance?

---

## Results Summary

SVD+Louvain **reduces head-movie fraction** measurably (confirming H-A) while incurring a **small but real drop in P@10 and nDCG@10** (confirming H-B). The trade-off curve shows diminishing diversity gains past ~8 community slots, suggesting the greedy re-ranker is effective but not free. Full analysis and figures are in `main_notebook.ipynb`.

---

## Repository Structure

```
.
├── main_notebook.ipynb          ← Final curated notebook (start here)
├── requirements.txt             ← Python dependencies
├── README.md
├── .gitignore
├── checkpoints/
│   ├── checkpoint_1_EDA.ipynb   ← CP1: Exploratory Data Analysis
│   └── checkpoint_2.ipynb       ← CP2: Research Question Formation
├── notebooks/
│   ├── svd_bias_analysis.ipynb  ← Method A: SVD Analysis 
│   ├── louvain_community_analysis.ipynb           
├── assets/                      ← Method B: Louvain Analysis
│   └── *.png
└── data/
    └── ml-10M100K/              ← MovieLens 10M (see Data section below)
```

---

## Data

**Dataset:** [MovieLens 10M](https://grouplens.org/datasets/movielens/10m/)
**Files used:** `ratings.dat`, `movies.dat`, `tags.dat`
**Size:** ~63 MB compressed; ~10M ratings, 71K users, 10K movies

The data directory is not committed to this repo (too large). To reproduce:

1. Download from https://grouplens.org/datasets/movielens/10m/
2. Unzip into `data/ml-10M100K/`

---

## How to Reproduce

This project runs locally with Python 3.8  — not Colab.

```bash
# 1. Create environment
conda create -n dma python=3.8
conda activate dma

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data (see Data section above)

# 4. Run the main notebook
jupyter notebook main_notebook.ipynb
```

Notebooks run in order:
1. `checkpoints/checkpoint_1_EDA.ipynb` — EDA
2. `checkpoints/checkpoint_2.ipynb` — RQ formation
3. `main_notebook.ipynb` — Full implementation & results

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

Full dependency list: [`requirements.txt`](requirements.txt)

---

## Methods

| | Method A: SVD | Method B: SVD + Louvain |
|---|---|---|
| Rating prediction | `scipy svds k=35` | Same SVD model |
| Candidate pool | Top-10 by SVD score | Top-100 by SVD score |
| Re-ranking | None | Greedy community filter → Top-10 |
| New variable | — | Louvain community membership |

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

## EDA Key Findings (CP1)

- 99.96% sparsity in the user-movie matrix
- Top 10% of movies receive 58% of all ratings (strong popularity bias)
- Mean rating: 3.51
- ~90% of users have ≤ 5 ratings (cold-start challenge)

---

*Harper, F.M. & Konstan, J.A. (2015). The MovieLens Datasets. ACM TIIS.*
