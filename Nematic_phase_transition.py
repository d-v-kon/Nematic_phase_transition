import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

L = 10
N = L * L
epsilon = 1.0
steps_per_T = 30000
equilibration_steps = 1800
#temperatures = [0.1, 10]
temperatures = np.linspace(0.1, 2.5, 20)
delta_theta = np.pi / 8


def neighbors(i, j, L):
    return [((i - 1) % L, j), ((i + 1) % L, j), (i, (j - 1) % L), (i, (j + 1) % L)]


def local_energy(i, j, angles, epsilon):
    theta = angles[i, j]
    u = np.array([np.cos(theta), np.sin(theta)])
    E = 0.0
    for ni, nj in neighbors(i, j, L):
        theta_n = angles[ni, nj]
        v = np.array([np.cos(theta_n), np.sin(theta_n)])
        E += -epsilon * (np.dot(u, v) ** 2)
    return E


def total_energy(L, angles, epsilon):  # it's something suboptimal, since I calculate every bound twice, but not crucial
    E = 0.0
    for i in range(L):
        for j in range(L):
            E += local_energy(i, j, angles, epsilon)
    E = E / 2
    return E


def compute_nematic_tensor_and_S(angles):
    thetas = angles.flatten()
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)

    Q_xx = np.mean(2 * cos_theta**2 - 1)
    Q_yy = np.mean(2 * sin_theta**2 - 1)
    Q_xy = np.mean(2 * cos_theta * sin_theta)

    Q = np.array([[Q_xx, Q_xy],
                  [Q_xy, Q_yy]])

    eigvals, eigvecs = np.linalg.eigh(Q)
    max_idx = np.argmax(eigvals)
    S = eigvals[max_idx]
    return Q, S


def monte_carlo_step(angles, beta, epsilon):
    i, j = np.random.randint(0, L), np.random.randint(0, L)
    old_angle = angles[i, j]
    old_energy = local_energy(i, j, angles, epsilon)

    new_angle = old_angle + np.random.uniform(-delta_theta, delta_theta) % (2 * np.pi)
    angles[i, j] = new_angle
    new_energy = local_energy(i, j, angles, epsilon)

    dE = new_energy - old_energy
    if dE > 0 and np.random.rand() > np.exp(-beta * dE):
        angles[i, j] = old_angle


def autocorrelation(S_series, max_lag):
    S = np.array(S_series)
    N = len(S)
    S_mean = np.mean(S)
    var = np.var(S)
    acf = np.zeros(max_lag)
    for tau in range(max_lag):
        acf[tau] = np.mean((S[:N - tau] - S_mean) * (S[tau:] - S_mean)) / var
    return acf


def integrated_autocorrelation_time(acf):
    return 0.5 + np.sum(acf[1:])


"""T = temperatures[-1]
beta = 1 / T
angles = np.random.uniform(0, 2 * np.pi, (L, L))
energy_trace = []
for step in tqdm(range(10000)):
    monte_carlo_step(angles, beta, epsilon)
    if step % 100 == 0:
        E = total_energy(L, angles, epsilon)
        energy_trace.append(E)

plt.plot(range(100) * 100, energy_trace)
plt.xlabel('Monte Carlo step')
plt.ylabel('Total energy')
plt.title('Convergence of energy')
plt.grid(True)
plt.show()
# necessary equilibration steps ~ 1800"""

"""for T in [temperatures[0], temperatures[-1]]:
    print(f"Running simulation at T = {T}")
    beta = 1 / T
    angles = np.random.uniform(0, 2 * np.pi, (L, L))

    S_vals = []
    # Equilibration
    for step in range(equilibration_steps):
        monte_carlo_step(angles, beta, epsilon)

    # Production run
    for step in tqdm(range(100000)):
        monte_carlo_step(angles, beta, epsilon)
        if step % 100 == 0:
            _, S = compute_nematic_tensor_and_S(angles)
            S_vals.append(S)

    # Plot histogram
    plt.figure()
    plt.hist(S_vals, bins=40, density=True)
    plt.title(f"Histogram of S at T = {T}")
    plt.xlabel("S")
    plt.ylabel("Probability density")
    plt.grid(True)
    plt.show()

    # Autocorrelation and integrated autocorrelation time
    max_lag = 200
    acf = autocorrelation(S_vals, max_lag)
    tau_int = 0.5 + np.sum(acf[1:])

    plt.figure()
    plt.plot(acf, label="ACF")
    plt.plot(np.exp(-np.arange(max_lag) / tau_int), '--', label=f"Exp fit (tau = {tau_int:.2f})")
    plt.title(f"Autocorrelation of S at T = {T}")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Integrated autocorrelation time at T = {T}: {tau_int:.2f}\n")"""

"""avg_S = []
err_S = []
tau_int_list = []

for T in tqdm(temperatures):
    beta = 1.0 / T
    angles = np.random.uniform(0, 2 * np.pi, (L, L))
    S_values = []

    for step in range(equilibration_steps):
        monte_carlo_step(angles, beta, epsilon)

    for step in range(steps_per_T):
        monte_carlo_step(angles, beta, epsilon)
        if step % 100 == 0:
            _, S = compute_nematic_tensor_and_S(angles)
        S_values.append(S)

    S_values = np.array(S_values)
    acf = autocorrelation(S_values, 200)
    tau_int = integrated_autocorrelation_time(acf)
    tau_int_list.append(tau_int)

    eff_samples = len(S_values) / (2 * tau_int) if tau_int > 0 else len(S_values)
    std_err = np.std(S_values) / np.sqrt(eff_samples) if eff_samples > 0 else 0

    avg_S.append(np.mean(S_values))
    err_S.append(std_err)

# Plot results
plt.errorbar(temperatures, avg_S, yerr=err_S, fmt='-o', capsize=3, label='⟨S⟩ with error bars')
plt.xlabel("Temperature (kBT/ε)")
plt.ylabel("Nematic Order Parameter ⟨S⟩")
plt.title("Nematic-Isotropic Phase Transition")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()"""
