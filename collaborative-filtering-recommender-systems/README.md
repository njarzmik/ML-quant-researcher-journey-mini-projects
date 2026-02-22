# Collaborative Filtering — Movie Recommender System

## Overview

A collaborative filtering algorithm that receives a 2D rating matrix **(num_movies × num_users)** where each entry represents a rating given to a movie by a user (0.5–5, in 0.5 steps). A value of 0 means the movie has not been rated yet. The goal is to estimate those missing opinions.

The algorithm learns two sets of vectors:
- **Movie vectors** — a latent description of each movie
- **User vectors** — a representation of each user's movie taste

The dot product of a movie vector and a user vector, plus a bias term, produces an estimated rating.

## Key Problem

We only have raw ratings — no explicit movie features and no explicit user profile data. We can't model users in isolation, nor movies in isolation. The algorithm must look at the whole picture: how every user rates every movie simultaneously.

## Solution

For each user a **preference vector** `(n_features, 1)` and for each movie a **description vector** `(n_features, 1)` are created. Using **gradient descent**, two linear regression models are combined into one that simultaneously optimises both sets of vectors so that:

```
rating_estimate(i, j) = X[i] · W[j] + b[j]
```

Because gradient descent operates on a non-convex landscape (many local minima and saddle points), it naturally escapes saddle points, and the local minima encountered are empirically all equally good.

## Why Linear Regression?

Movie ratings reflect human opinion, which moves in a roughly linear fashion. A bias term captures systematic tendencies — e.g. an optimistic user who consistently rates higher, or a pessimistic one who rates lower — while the latent vectors capture their actual preferences.

## Regularisation

L2 (squared) regularisation is applied to both the movie and user vectors to reduce overfitting. This pushes the vectors toward smaller magnitudes, preventing any single latent feature from dominating.

## Data & Architecture

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `Y` | (4778, 443) | Rating matrix |
| `R` | (4778, 443) | Binary mask — 1 if rated, 0 otherwise |
| `X` | (4778, 100) | Movie latent vectors |
| `W` | (443, 100) | User latent vectors |
| `b` | (1, 443) | Per-user bias terms |

- **4 778 movies**, **443 users**, **100 latent features**
- Optimiser: **Adam** (`lr = 0.1`)
- Trained for **200 iterations**

## Normalisation

Ratings are mean-normalised per movie before training. This ensures that new users with no ratings receive the global average as their default prediction (their parameters start at zero, and the mean is added back at inference time).

## Files

| File | Description |
|------|-------------|
| `move-ratings.ipynb` | Main notebook — model implementation and training |
| `recsys_utils.py` | Data loading utilities |
| `data/` | Rating dataset |
