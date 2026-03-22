"""
astar — Modular toolkit for the Astar Island competition (NM i AI 2026).

Modules
-------
api        Live API session (LiveSession) + REST helpers — REAL network calls
session    SimulatedSession — offline fake API from historical GT, no network
priors     Dirichlet prior construction (hand-tuned and data-driven)
inference  Query planning, observation collection, prediction building
dirichlet  DirichletLookup — method-of-moments fitting from historical data
features   Spatial feature extraction (terrain codes, distance bins)
scoring    Competition scoring: entropy-weighted KL divergence
evaluation Leave-one-round-out cross-validation utilities
visualise  Colour maps, terrain/prediction renderers, matplotlib helpers
data       Data I/O: parsing round data, saving observations, GT fetching
"""
