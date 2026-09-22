# GAIA

Cross-domain cascade failure prediction in ecological systems.

Most environmental monitoring tools tell you what's wrong right now. GAIA tries to tell you what breaks next.

When a hypoxic dead zone forms in the Gulf of Mexico, fish populations displace, predators shift how they forage, algae blooms amplify, and oxygen levels drop even further. None of that happens in isolation - it's a cascade. GAIA models that chain with a graph neural network, where nodes are ecological indicators and edges are causal relationships pulled from the literature.

Environmental AI mostly works in silos right now: atmospheric, marine, and terrestrial models don't talk to each other, so researchers piece cross-domain cascades together by hand, after the fact. GAIA treats the ecosystem as what it actually is, a graph, and propagates disturbance signals forward through time across those domain boundaries.

## How it works

Real-time data comes in from NOAA, OpenAQ, NASA Earthdata, iNaturalist, and SoilGrids, then gets normalized into ecological indicator nodes (hypoxia, fish biomass, algae, soil health, and so on) connected by directed causal edges with weights and propagation lags grounded in prior research. A graph neural network learns to propagate disturbance signals across that graph, refining the literature-derived edge weights using observed historical data.

End result looks something like: *"hypoxic event detected, fish displacement in 72h (84%), bloom amplification in 14d (63%)."*

## Status

This is an active research project, not a finished product. Core training and ingestion pipelines are working; the serving layer, visualization, and historical validation are still in progress. See [ROADMAP.md](./ROADMAP.md) for the full breakdown of what's done and what's next.

## Project structure

```
api/              # prediction-serving API (in progress)
data/ingestion/   # pulls and normalizes data from NOAA, OpenAQ, NASA Earthdata, iNaturalist, SoilGrids
frontend/         # visualization layer
graph/            # graph construction — indicator nodes, causal edges, weights
model/            # GNN training and inference
```

## Setup

```bash
git clone https://github.com/bhavv04/gaia.git
cd gaia
cp .env.example .env   # fill in your API keys/tokens
pip install -r requirements.txt
```

<!-- Once docker-compose is finished end-to-end, replace/extend this with:
docker compose up
-->

## License

GPL-3.0