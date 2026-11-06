from __future__ import annotations
import numpy as np

N, K = 40, 6


class Agent:
    name = "base"
    description = ""

    def scores(self, history: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return 40 scores (higher = more preferred). Uses only past draws."""
        raise NotImplementedError

    def select(self, history: np.ndarray, rng: np.random.Generator) -> list[int]:
        s = self.scores(history, rng)
        s = s + rng.random(N) * 1e-9  # random tie-break
        return sorted((np.argsort(-s)[:K] + 1).tolist())

    def explain(self, history: np.ndarray, picks: list[int]) -> str:
        return self.description
