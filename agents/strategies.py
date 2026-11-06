"""Selection strategies. None of these can beat chance on a fair lottery;
the backtester exists to demonstrate that rigorously."""
from __future__ import annotations
import numpy as np
from .base import Agent, N, K


class RandomAgent(Agent):
    name = "random"
    description = "Uniform random pick of 6 distinct numbers (the control)."

    def scores(self, history, rng):
        return rng.random(N)


class HotAgent(Agent):
    name = "hot"

    def __init__(self, window: int = 50):
        self.window = window
        self.description = f"Picks the 6 numbers drawn most often in the last {window} draws."

    def scores(self, history, rng):
        return history[-self.window:].sum(axis=0).astype(float) if len(history) else np.zeros(N)


class OverdueAgent(Agent):
    name = "overdue"
    description = "Picks the 6 numbers that have gone longest without appearing."

    def scores(self, history, rng):
        T = len(history)
        if T == 0:
            return np.zeros(N)
        return np.where(history.any(axis=0), np.argmax(history[::-1], axis=0), T).astype(float)


class ThompsonAgent(Agent):
    name = "bayes_thompson"

    def __init__(self, prior_strength: float = 40.0, window: int | None = None):
        self.prior_strength = prior_strength
        self.window = window
        self.description = ("Beta-Bernoulli model of each number's appearance rate (prior centred on 0.15); "
                            "samples a rate per number and picks the top 6 (Thompson sampling).")

    def scores(self, history, rng):
        h = history if self.window is None else history[-self.window:]
        T = len(h)
        a0 = 0.15 * self.prior_strength
        b0 = 0.85 * self.prior_strength
        hits = h.sum(axis=0) if T else np.zeros(N)
        return rng.beta(a0 + hits, b0 + (T - hits))

    def posterior_mean(self, history):
        h = history if self.window is None else history[-self.window:]
        T = len(h)
        a0, b0 = 0.15 * self.prior_strength, 0.85 * self.prior_strength
        hits = h.sum(axis=0) if T else np.zeros(N)
        return (a0 + hits) / (a0 + b0 + T)


class LowOverlapAgent(Agent):
    """Avoid combinations other players likely choose (birthdays, small numbers).
    Does NOT change hit probability; aims to reduce jackpot sharing."""
    name = "low_overlap"
    description = ("Favours numbers above 31 and less 'popular' choices so that, if you did win, "
                   "you'd be less likely to split the jackpot. Does not improve your odds of winning.")

    def popularity(self) -> np.ndarray:
        n = np.arange(1, N + 1)
        pop = np.ones(N)
        pop[n <= 31] *= 1.6      # birthday days
        pop[n <= 12] *= 1.25     # also month numbers
        pop[np.isin(n, [7, 3, 11, 13, 21])] *= 1.2  # commonly 'lucky' numbers
        return pop

    def scores(self, history, rng):
        w = 1.0 / self.popularity()
        # weighted sampling w/o replacement via Gumbel-top-k
        return np.log(w) + rng.gumbel(size=N)


def default_agents():
    return [RandomAgent(), HotAgent(50), OverdueAgent(), ThompsonAgent(), LowOverlapAgent()]
