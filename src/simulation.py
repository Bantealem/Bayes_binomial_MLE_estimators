"""
## Project 202633: Bayes binomial vs MLE estimators

Monte Carlo comparison of the MSE of two estimators of p from a
Bernoulli(p) sample of size n:
  MLE:   X_bar
  Bayes: (sum(X) + sqrt(n/4)) / (n + sqrt(n))

Run from the repository root:  python src/simulation.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 2026
M = 10_000                                # Monte Carlo replications
N_VALUES = [4, 16, 50, 200]               # sample sizes
P_GRID = np.linspace(0.01, 0.99, 50)      # true values of p
OUT_DIR = Path(__file__).resolve().parent.parent / "results"


def mle(S, n):
    return S / n


def bayes(S, n):
    return (S + np.sqrt(n / 4)) / (n + np.sqrt(n))


def mse_mle_theory(p, n):
    return p * (1 - p) / n


def mse_bayes_theory(p, n):
    return np.full_like(p, 1 / (4 * (1 + np.sqrt(n)) ** 2))


def crossover_halfwidth(n):
    """Bayes beats MLE exactly when |p - 1/2| < this value."""
    return np.sqrt(0.25 - n / (4 * (1 + np.sqrt(n)) ** 2))


def simulate(rng):
    results = {}
    for n in N_VALUES:
        sim_mle, sim_bayes = [], []
        for p in P_GRID:
            S = rng.binomial(n, p, size=M)   # sum of n Bernoulli(p) draws
            sim_mle.append(np.mean((mle(S, n) - p) ** 2))
            sim_bayes.append(np.mean((bayes(S, n) - p) ** 2))
        results[n] = (np.array(sim_mle), np.array(sim_bayes))
    return results


def plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, n in zip(axes.ravel(), N_VALUES):
        sim_mle, sim_bayes = results[n]
        ax.plot(P_GRID, mse_mle_theory(P_GRID, n), "b-", label="MLE (theory)")
        ax.plot(P_GRID, mse_bayes_theory(P_GRID, n), "r-", label="Bayes (theory)")
        ax.plot(P_GRID, sim_mle, "b.", label="MLE (simulated)")
        ax.plot(P_GRID, sim_bayes, "r.", label="Bayes (simulated)")
        ax.set_title(f"n = {n}")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("true p")
    axes[0, 0].legend()
    fig.suptitle(f"MSE of MLE vs Bayes estimator (M = {M:,} replications)")
    fig.tight_layout()
    return fig


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = simulate(rng)
    plot(results).savefig(OUT_DIR / "mse_comparison.png", dpi=150)

    lines = ["n   | theory: Bayes wins for p in | simulated: Bayes wins for p in"]
    for n in N_VALUES:
        hw = crossover_halfwidth(n)
        sim_mle, sim_bayes = results[n]
        wins = P_GRID[sim_bayes < sim_mle]
        lines.append(f"{n:<3} | [{0.5-hw:.3f}, {0.5+hw:.3f}]            | "
                     f"[{wins.min():.2f}, {wins.max():.2f}]")
    table = "\n".join(lines)
    (OUT_DIR / "crossover_table.txt").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
