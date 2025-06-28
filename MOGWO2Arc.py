import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ===================== Objective Function ===================== #
def evaluate_truss(x):
    mass = np.sum(x)  # Dummy mass (sum of cross-sectional areas)
    compliance = np.sum(x ** 2)  # Dummy compliance
    return mass, compliance

# ===================== Constraint Handling ===================== #
def penalty_function(x, lam=1000):
    constraints = [0]  # Placeholder: Add real constraints here
    penalty = lam * sum(max(0, g) for g in constraints)
    return penalty

# ===================== Initialization ===================== #
def initialize_population(n_agents, dim, bounds):
    return np.random.uniform(bounds[0], bounds[1], (n_agents, dim))

# ===================== Non-Dominated Sorting ===================== #
def dominates(a, b):
    return all(ai <= bi for ai, bi in zip(a, b)) and any(ai < bi for ai, bi in zip(a, b))

def non_dominated_sort(pop_objs):
    n = len(pop_objs)
    domination_count = [0] * n
    dominated_solutions = [[] for _ in range(n)]
    front = []
    for i in range(n):
        for j in range(n):
            if dominates(pop_objs[i], pop_objs[j]):
                dominated_solutions[i].append(j)
            elif dominates(pop_objs[j], pop_objs[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            front.append(i)
    return front

# ===================== IMOGWO Position Update ===================== #
def imogwo_position_update(X1, X2, X3):
    p = np.random.rand()
    if p < 0.33:
        return X1
    elif p < 0.66:
        return (X1 + X2) / 2
    else:
        return (X1 + X2 + X3) / 3

# ===================== Two Archive Update ===================== #
def archive_update(pop, pop_objs, archive_size):
    front_indices = non_dominated_sort(pop_objs)
    return [pop[i] for i in front_indices[:archive_size]], [pop_objs[i] for i in front_indices[:archive_size]]

# ===================== Adaptive Probability Function ===================== #
def compute_Sp(t, t_max, Sp_s=0.1, Sp_f=0.9):
    c2 = (np.log(Sp_f) - np.log(Sp_s)) / (t_max - 1)
    return Sp_s * np.exp(c2 * t)

# ===================== Main MOGWO2Arc ===================== #
def mogwo2arc(n_agents=50, dim=10, bounds=(0.001, 0.021), max_iter=100, archive_size=30):
    X = initialize_population(n_agents, dim, bounds)
    Archive1, Archive2 = [], []
    Archive1_objs, Archive2_objs = [], []

    for t in range(max_iter):
        objs = [evaluate_truss(x) for x in X]
        penalties = [penalty_function(x) for x in X]
        fitness = [(f[0] + p, f[1] + p) for f, p in zip(objs, penalties)]

        Archive1, Archive1_objs = archive_update(X, fitness, archive_size)

        w1, w2 = np.random.rand(), 1 - np.random.rand()
        Archive2_objs = [
            (1 / (np.linalg.norm(np.subtract(f, other)) + 1e-6),
             w1 * f[0] + w2 * f[1])
            for f in fitness for other in fitness if not np.array_equal(f, other)
        ]
        Archive2 = X[:len(Archive2_objs)]

        Sp_t = compute_Sp(t, max_iter)
        use_archive1 = np.random.rand() < Sp_t
        leaders = Archive1 if use_archive1 and len(Archive1) >= 3 else initialize_population(3, dim, bounds)
        Alpha, Beta, Delta = leaders[:3]

        for i in range(n_agents):
            A = 2 * np.random.rand(dim) - 1
            C = 2 * np.random.rand(dim)

            X1 = Alpha - A * np.abs(C * Alpha - X[i])
            X2 = Beta - A * np.abs(C * Beta - X[i])
            X3 = Delta - A * np.abs(C * Delta - X[i])
            X[i] = imogwo_position_update(X1, X2, X3)
            X[i] = np.clip(X[i], bounds[0], bounds[1])

    final_objs = [evaluate_truss(x) for x in X]
    final_front = non_dominated_sort(final_objs)
    pareto_set = [X[i] for i in final_front]
    pareto_objs = [final_objs[i] for i in final_front]
    return pareto_set, pareto_objs

# ===================== Plot Pareto Front ===================== #
def plot_pareto(pareto_objs):
    mass = [f[0] for f in pareto_objs]
    compliance = [f[1] for f in pareto_objs]
    plt.figure(figsize=(8, 6))
    plt.scatter(mass, compliance, c='blue')
    plt.xlabel('Mass')
    plt.ylabel('Compliance')
    plt.title('Pareto Front')
    plt.grid(True)
    plt.show()

# ===================== Export to CSV ===================== #
def export_results(pareto_set, pareto_objs, filename="pareto_results.csv"):
    data = []
    for x, f in zip(pareto_set, pareto_objs):
        row = list(x) + list(f)
        data.append(row)
    
    columns = [f"x{i+1}" for i in range(len(pareto_set[0]))] + ["Mass", "Compliance"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    print(f"\n✅ Results saved to {filename}")

# ===================== Run Optimization ===================== #
if __name__ == "__main__":
    pareto_set, pareto_objs = mogwo2arc(max_iter=200)

    # Print Pareto-optimal solutions
    print("=== Pareto-Optimal Design Variables ===")
    for i, x in enumerate(pareto_set):
        print(f"Solution {i+1}: {x}")

    print("\n=== Pareto-Optimal Objective Values (Mass, Compliance) ===")
    for i, f in enumerate(pareto_objs):
        print(f"Solution {i+1}: Mass = {f[0]:.6f}, Compliance = {f[1]:.6f}")

    # Plot Pareto front
    plot_pareto(pareto_objs)

    # Save to CSV
    export_results(pareto_set, pareto_objs)