# Plateau statistics: true plateau vs. false spike

Synthetic 5x5 grid (lookback x threshold), radius-1 neighbourhood, seed 0.

| Statistic | A: true plateau | B: false spike |
| --- | ---: | ---: |
| best_params (lookback, threshold) | (30, 0.03) | (20, 0.04) |
| plateau_best_params (lookback, threshold) | (30, 0.03) | (10, 0.05) |
| best_metric | 1.384 | 2.000 |
| plateau_best_score | 1.056 | 0.642 |
| metric_gap | 0.000 | 1.934 |
| grid_distance | 0 | 1 |
| best_neighborhood_mean | 1.056 | 0.039 |
| best_neighborhood_std | 0.164 | 0.224 |
| best_neighborhood_size | 8 | 8 |
| spikiness | 1.997 | 8.752 |
