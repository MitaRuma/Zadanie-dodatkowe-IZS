import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(0)

# ZADANIE 1

M = 10000
N_tab = [10, 50, 200, 1000]

# rozklad 0 1
x = np.random.uniform(0, 1, 100000)

plt.figure()
plt.hist(x, bins=50, density=True)
plt.plot([0, 1], [1, 1])
plt.title("Histogram U(0,1)")
plt.xlabel("x")
plt.ylabel("gęstość")
plt.show()

for N in N_tab:
    samples = np.random.uniform(0, 1, (M, N))
    means = np.mean(samples, axis=1)

    mu = 0.5
    sigma = np.sqrt(1 / (12 * N))
    xs = np.linspace(min(means), max(means), 300)

    plt.figure()
    plt.hist(means, bins=50, density=True)
    plt.plot(xs, norm.pdf(xs, mu, sigma))
    plt.title(f"Rozkład średnich U(0,1), N={N}")
    plt.xlabel("średnia")
    plt.ylabel("gestość")
    plt.show()

# rozklad cauchyego
x = np.random.standard_cauchy(100000)

plt.figure()
plt.hist(x, bins=200, density=True, range=(-10, 10))
plt.title("Histogram rozkładu Cauchy’ego")
plt.xlabel("x")
plt.ylabel("gęstość")
plt.show()

for N in N_tab:
    samples = np.random.standard_cauchy((M, N))
    means = np.mean(samples, axis=1)

    plt.figure()
    plt.hist(means, bins=200, density=True, range=(-10, 10))
    plt.title(f"Rozkład średnich Cauchy’ego, N={N}")
    plt.xlabel("średnia")
    plt.ylabel("gęstość")
    plt.show()

# ZADANIE 2

p = 0.11
N = 500
M = 5000

kroki = np.where(np.random.rand(M, N) < p, 1, -1)

trajektorie = np.cumsum(kroki, axis=1)
pos_koncowe = trajektorie[:, -1]

pos_srednie = np.mean(pos_koncowe)
pos_std = np.std(pos_koncowe, ddof=1)
ci = 1.96 * pos_std / np.sqrt(M)

print("Średnia pozycja:", pos_srednie)
print("95% przedział ufności:", pos_srednie - ci, pos_srednie + ci)

plt.figure()
plt.hist(pos_koncowe, bins=50, density=True)
plt.title("Histogram pozycji końcowych (p = 0.11)")
plt.xlabel("pozycja końcowa")
plt.ylabel("gęstość")
plt.show()

plt.figure()
for i in range(10):
    plt.plot(trajektorie[i])
plt.title("Przykładowe trajektorie wędrówki losowej")
plt.xlabel("krok")
plt.ylabel("pozycja")
plt.show()