import numpy as np

def mse_at(p, n, M, seed):
    # simulated MSE of both estimators at one value of p
    rng = np.random.default_rng(seed)
    S = rng.binomial(n, p, size=M)
    est_mle = S / n
    est_bayes = (S + np.sqrt(n / 4)) / (n + np.sqrt(n))
    return np.mean((est_mle - p) ** 2), np.mean((est_bayes - p) ** 2)

p = 0.3

# 1. different sample sizes
print("different n")
for n in [2, 10, 30, 100, 1000]:
    print(n, mse_at(p, n, 10000, 123))

# 2. different seeds
print("different seeds")
for seed in [1, 42, 123, 2026]:
    print(seed, mse_at(p, 10, 10000, seed))

# 3. different M
print("different M")
for M in [100, 1000, 10000, 100000]:
    print(M, mse_at(p, 10, M, 123))