# Roadmap

GAIA is under active development. This tracks what's done and what's next,
in rough priority order.

## Done

- [x] GNN training pipeline
- [x] MLflow experiment tracking
- [x] Freshwater and terrestrial data ingestion (was marine-heavy initially)

## In progress / next up

### Validation (highest priority — proves the model actually predicts something)
- [ ] Backtest cascade predictions against known historical events
  - [ ] Select 3–5 historical hypoxia events as test cases
  - [ ] Compare predicted cascade timing and probability against what actually happened
- [ ] Build evaluation metrics for cascade prediction accuracy
- [ ] Add confidence intervals to cascade predictions, not just point probabilities

### Robustness
- [ ] Handle missing/gappy data from ingestion sources gracefully
  - [ ] Decide on interpolation vs. flagging per source
  - [ ] Add fallback behavior when a feed is down
- [ ] Write tests for the graph builder

### Serving
- [ ] Build FastAPI endpoints for serving predictions
- [ ] Build the D3 cascade visualization on the frontend
- [ ] Set up Docker so the whole thing runs with one command

### Documentation
- [ ] Document causal edge sourcing so it's clear which edges come from which paper
- [ ] Investigate whether propagation lags should be learned rather than fixed priors
- [ ] Write up a proper paper once results are solid