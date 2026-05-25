# GAIA

**Cross-domain cascade failure prediction in ecological systems.**

Most environmental monitoring tools tell you what's wrong right now. GAIA tells you what breaks next.

When a hypoxic dead zone forms in the Gulf of Mexico, fish populations displace, predator foraging patterns shift, algae blooms amplify, and oxygen levels drop further. These aren't isolated events — they're a cascade. GAIA models that chain using a graph neural network where nodes are ecological indicators and edges are empirically-derived causal relationships between them.

---

## The problem

Current environmental AI operates in silos. Atmospheric models don't talk to marine models. Marine models don't talk to terrestrial ones. Researchers piece together cross-domain cascades manually, after the fact.

GAIA treats the ecosystem as what it actually is — a graph — and propagates disturbance signals forward in time across domain boundaries.

---

## How it works

```
Real-time data feeds (NOAA, OpenAQ, NASA Earthdata, iNaturalist, SoilGrids)
        ↓
Normalised ecological indicator nodes (hypoxia, fish biomass, algae, soil health...)
        ↓
Directed causal edges with empirically-derived weights and propagation lags
        ↓
Graph Neural Network — learns to propagate disturbance signals across the graph
        ↓
Cascade prediction: "hypoxic event detected → fish displacement in 72h (84%) → bloom amplification in 14d (63%)"
```

The graph is built from literature-grounded prior knowledge — causal relationships sourced from Rabalais et al., Breitburg et al., and Diaz & Rosenberg. The GNN refines edge weights from observed historical data.

---

## Structure

```
gaia/
├── data/
│   └── ingestion/          # Per-source API fetchers (NOAA, OpenAQ, NASA, iNaturalist)
├── graph/
│   ├── schema.py           # Node types, edge types, domain taxonomy
│   ├── edges.py            # Literature-derived causal edge definitions
│   └── builder.py          # Assembles EcoNode instances into a NetworkX DiGraph
├── model/
│   ├── gnn.py              # PyTorch Geometric graph neural network
│   ├── train.py            # Training loop with MLflow experiment tracking
│   └── evaluate.py         # Cascade prediction evaluation metrics
├── api/
│   └── main.py             # FastAPI — serves cascade predictions as JSON
├── frontend/
│   └── src/
│       └── components/
│           └── CascadeGraph.tsx   # D3 graph propagation visualisation
└── notebooks/              # EDA, graph construction, model experiments
```

---

## Domains modelled

| Domain | Indicators |
|---|---|
| Marine | Hypoxic zones, fish populations, algae blooms, benthic habitat, SST |
| Freshwater | Nutrient runoff, water temperature |
| Atmospheric | Air quality, CO₂ concentration, precipitation |
| Terrestrial | Soil health, vegetation index, predator populations |

Cross-domain edges are the core research contribution — the causal chains that existing tools don't model end-to-end.

---

## Stack

- **Graph modelling** — NetworkX, PyTorch Geometric
- **Data pipeline** — Python, Polars
- **Experiment tracking** — MLflow
- **API** — FastAPI
- **Visualisation** — Next.js, D3.js
- **Infrastructure** — Docker

---

## Status

Active development. The graph layer (schema, edge definitions, builder) is complete. GNN training pipeline in progress.

---

## Background

Built on prior work modelling Gulf of Mexico hypoxic zones using ERA5 and NOAA reanalysis data. GAIA extends that into a multi-domain cascade framework, the dead zone is one node in a larger system.  
Ecological references: Rabalais et al. (2002), Breitburg et al. (2018), Diaz & Rosenberg (2008), Scheffer et al. (2001).