"""
Live API session and client functions for the Astar Island competition.

WARNING
-------
Importing and using LiveSession or any function in this module makes REAL
API calls that consume your query budget for the active round.

For offline experimentation use astar.session.SimulatedSession instead —
it has the SAME interface but draws from historical ground truth data and
never touches the network.
"""

from __future__ import annotations

import time

import numpy as np
import requests

API_BASE = "https://api.ainm.no/astar-island"


# ---------------------------------------------------------------------------
# Low-level rate-limited HTTP session
# ---------------------------------------------------------------------------

class RateLimitedSession:
    """requests.Session with separate rate limits for simulate vs submit."""

    def __init__(self, token: str):
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self._last: dict[str, float] = {"simulate": 0.0, "submit": 0.0, "default": 0.0}
        self._min_interval: dict[str, float] = {
            "simulate": 1.0 / 3.0,   # API limit 5/s, stay at 3/s
            "submit":   1.0 / 1.0,   # API limit 2/s, stay at 1/s
            "default":  0.2,
        }

    def _wait(self, kind: str = "default") -> None:
        key = kind if kind in self._min_interval else "default"
        elapsed = time.time() - self._last[key]
        gap = self._min_interval[key]
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last[key] = time.time()

    def get(self, url: str, **kwargs) -> dict:
        self._wait("default")
        resp = self.session.get(url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def post(self, url: str, **kwargs) -> dict:
        kind = "submit" if url.endswith("/submit") else "simulate"
        backoff = 2.0
        for attempt in range(6):
            self._wait(kind)
            resp = self.session.post(url, **kwargs)
            if resp.status_code == 429:
                wait = backoff * (2 ** attempt)
                print(f"    [rate limit] 429 — waiting {wait:.0f}s before retry {attempt+1}/5")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()  # re-raise after all retries exhausted
        return resp.json()


# ---------------------------------------------------------------------------
# Unified LiveSession — same interface as SimulatedSession
# ---------------------------------------------------------------------------

class LiveSession:
    """High-level session for a single active round.

    Wraps RateLimitedSession and exposes the same .simulate() / .submit()
    interface as SimulatedSession so inference code works with both.

    Args:
        token:    JWT token from AINM_TOKEN environment variable.
        round_id: ID of the active round to query.
    """

    def __init__(self, token: str, round_id: str):
        self._http = RateLimitedSession(token)
        self.round_id = round_id
        self.queries_used: int = 0
        self.queries_max: int = 50

    def simulate(
        self,
        seed_idx: int,
        x: int, y: int,
        w: int = 15, h: int = 15,
    ) -> dict:
        """Query one viewport for one seed. Returns same dict format as SimulatedSession."""
        result = self._http.post(f"{API_BASE}/simulate", json={
            "round_id":    self.round_id,
            "seed_index":  seed_idx,
            "viewport_x":  x,
            "viewport_y":  y,
            "viewport_w":  w,
            "viewport_h":  h,
        })
        self.queries_used = result.get("queries_used", self.queries_used + 1)
        self.queries_max  = result.get("queries_max",  self.queries_max)
        return result

    def submit(self, seed_idx: int, prediction: np.ndarray) -> dict:
        """Submit a (H, W, 6) probability tensor for one seed."""
        return self._http.post(f"{API_BASE}/submit", json={
            "round_id":   self.round_id,
            "seed_index": seed_idx,
            "prediction": prediction.tolist(),
        })

    @property
    def budget_remaining(self) -> int:
        return self.queries_max - self.queries_used


# ---------------------------------------------------------------------------
# Standalone REST helpers (used by historical_data.ipynb and evaluate.py)
# ---------------------------------------------------------------------------

def get_rounds(session: RateLimitedSession) -> list:
    return session.get(f"{API_BASE}/rounds")


def get_round_details(session: RateLimitedSession, round_id: str) -> dict:
    return session.get(f"{API_BASE}/rounds/{round_id}")


def get_budget(session: RateLimitedSession, round_id: str) -> dict:
    return session.get(f"{API_BASE}/budget")


def get_my_rounds(session: RateLimitedSession) -> list:
    return session.get(f"{API_BASE}/my-rounds")


def get_analysis(session: RateLimitedSession, round_id: str, seed_index: int) -> dict:
    return session.get(f"{API_BASE}/analysis/{round_id}/{seed_index}")
