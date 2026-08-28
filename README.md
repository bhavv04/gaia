# Gaia

Cross-domain cascade failure prediction in ecological systems.

Most environmental monitoring tools tell you what's wrong right now. GAIA tries to tell you what breaks next.

When a hypoxic dead zone forms in the Gulf of Mexico, fish populations displace, predators shift how they forage, algae blooms amplify, and oxygen levels drop even further. None of that happens in isolation, it's a cascade. GAIA models that chain with a graph neural network, where nodes are ecological indicators and edges are causal relationships pulled from the literature.

Environmental AI mostly works in silos right now. Atmospheric, marine, and terrestrial models don't talk to each other, so researchers piece cross domain cascades together by hand, after the fact. GAIA treats the ecosystem as what it actually is, a graph, and propagates disturbance signals forward through time across those domain boundaries.

## How it works

Real time data comes in from NOAA, OpenAQ, NASA Earthdata, iNaturalist, and SoilGrids, then gets normalized into ecological indicator nodes (hypoxia, fish biomass, algae, soil health, and so on) connected by directed causal edges with weights and propagation lags grounded in prior research. A graph neural network learns to propagate disturbance signals across that graph, refining the literature derived edge weights using observed historical data.

End result looks something like: "hypoxic event detected, fish displacement in 72h (84%), bloom amplification in 14d (63%)."

## To do

- [x] Finish the GNN training pipeline
- [x] Wire up MLflow experiment tracking properly
- [x] Pull in real historical data to validate edge weights, not just literature priors
- [ ] Backtest cascade predictions against known past events (e.g. past Gulf hypoxia seasons)
  - [ ] Pick 3-5 historical hypoxia events as test cases
  - [ ] Compare predicted cascade timing and probability against what actually happened
- [ ] Build out the evaluation metrics for cascade prediction accuracy
- [x] Add freshwater and terrestrial data ingestion (currently marine heavy)
- [ ] Handle missing/gappy data from the ingestion sources gracefully
  - [ ] Decide on interpolation vs. flagging per source
  - [ ] Add fallback behavior when a feed is down
- [ ] Build the FastAPI endpoints for serving predictions
- [ ] Build the D3 cascade visualization on the frontend
- [ ] Add confidence intervals to cascade predictions, not just point probabilities
- [ ] Write tests for the graph builder
- [ ] Document the causal edge sourcing so it's clear which edges come from which paper
- [ ] Look into whether propagation lags should be learned rather than fixed priors
- [ ] Set up Docker so the whole thing runs with one command
- [ ] Write up a proper writeup/paper once results are solid