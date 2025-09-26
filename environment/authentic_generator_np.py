import numpy as np
import random



def random_integer_partition(v, b):
    """
    Randomly partition integer v into b nonnegative integers, according to Ding & Chou (2015).
    (https://www.sciencedirect.com/science/article/pii/S0377221715002660#sec0014a)
    """
    if b == 1:
        return [v]
    y = list(range(1, v + b))
    for i in range(b - 1):
        j = random.randint(i, v + b - 2)
        y[i], y[j] = y[j], y[i]
    y[:b - 1] = sorted(y[:b - 1])
    x = [y[0] - 1]
    for i in range(1, b - 1):
        x.append(y[i] - y[i - 1] - 1)
    x.append(v + b - 1 - y[b - 2])
    return x

def dirichlet_partition(v, b, alpha=1.0, weights=None):
    """
    Partition integer v into b nonnegative integers using a Dirichlet distribution.

    Parameters:
    - v: total integer to partition
    - b: number of bins
    - alpha: concentration parameter (small -> skewed, large -> uniform)
    - weights: optional prior weights per bin (length b)

    Returns:
    - list of length b with integer partition summing to v
    """
    if weights is None:
        alphas = np.full(b, alpha)
    else:
        alphas = np.array(weights) * alpha

    probs = np.random.dirichlet(alphas)
    partition = np.random.multinomial(v, probs)
    return partition.tolist()


def generate_authentic_matrix(P, C, target_utils,middle_leg=None,  loading_only=False,
                              sparsity=0.3, perturb=0.2,  dirichlet=False, alpha=1.0):
    """
    Generate an authentic OD matrix, similar to to Ding & Chou (2015).
    (https://www.sciencedirect.com/science/article/pii/S0377221715002660#sec0014a)

    Parameters:
    - P: total number of ports
    - C: bay capacity
    - target_utils: list of target utilization fractions
    - loading_only: if True, loading ports only ship to discharging ports
    - middle_leg: index separating loading/discharging ports
    - sparsity: probability of zeroing an OD pair
    - perturb: fraction for random perturbation

    Returns:
    - T: P x P numpy array
    """
    T = np.zeros((P, P), dtype=int)
    if middle_leg is None:
        middle_leg = P // 2

    n_loading = middle_leg if loading_only else P - 1

    if len(target_utils) != n_loading:
        raise ValueError(f"target_utils must have length equal to middle_leg ({middle_leg})")

    for i in range(n_loading):
        # Determine start of destinations
        dest_start = middle_leg if loading_only else i + 1
        b = P - dest_start
        if b <= 0:
            continue

        # Target containers for this row, subtract already assigned
        v = int(target_utils[i] * C) - np.sum(T[:i, dest_start:])
        v = max(v, 0)

        # Partition containers
        if dirichlet:
            partition = dirichlet_partition(v, b, alpha=alpha)
        else:
            partition = random_integer_partition(v, b)

        # Apply sparsity
        for idx in range(b):
            if random.random() < sparsity:
                partition[idx] = 0

        # Re-normalize to match target
        s = sum(partition)
        if s > 0:
            partition = [int(round(x * v / s)) for x in partition]
        else:
            partition[random.randint(0, b - 1)] = v


        # Apply perturbation
        for idx in range(b):
            if partition[idx] > 0:
                delta = int(round(partition[idx] * random.uniform(-perturb, perturb)))
                partition[idx] = max(partition[idx] + delta, 0)

        T[i, dest_start:] = partition

    return T


def generate_multicargo_matrix(P, C, target_utils,
                               middle_leg=None, loading_only=False,
                               sparsity=0.3, perturb=0.2,
                               dirichlet=True, alpha=0.5,
                               cargo_shares=None,
                               include_reefer=True):
    """
    Generate OD matrices for multiple cargo types.

    Parameters:
    - P: number of ports
    - C: bay capacity
    - target_utils: target utilization list
    - cargo_shares: dict or array defining shares across cargo types
                    If None, uniform over all types
    - include_reefer: whether to add reefer types

    Returns:
    - T: dict {cargo_type: OD matrix}
    - cargo_types: list of cargo type labels
    """
    # Define cargo categories
    sizes = ["20ft", "40ft"]
    weights = ["light", "medium", "heavy"]
    revenues = ["long", "spot"]

    cargo_types = [(s, w, r) for s in sizes for w in weights for r in revenues]

    if include_reefer:
        # Add reefer cargo in most common type (assume 40ft, medium)
        cargo_types += [("40ft", "medium", "long", "reefer"),
                        ("40ft", "medium", "spot", "reefer")]

    K = len(cargo_types)

    # If no cargo_shares given, assume uniform
    if cargo_shares is None:
        cargo_shares = np.ones(K) / K
    else:
        cargo_shares = np.array(cargo_shares)
        cargo_shares = cargo_shares / cargo_shares.sum()

    # Allocate matrices
    T_multi = {}

    for k, ctype in enumerate(cargo_types):
        # Scale capacity for this type
        C_k = int(C * cargo_shares[k])

        # Generate OD matrix for this cargo type
        T_multi[ctype] = generate_authentic_matrix(
            P, C_k, target_utils,
            middle_leg=middle_leg,
            loading_only=loading_only,
            sparsity=sparsity,
            perturb=perturb,
            dirichlet=dirichlet,
            alpha=alpha
        )

    return T_multi, cargo_types

def randomize_demand_matrix(T_multi, dist="poisson", n_scenarios=10,
                            dispersion=1.0, sigma=0.3, seed=None):
    """
    Take expected OD matrices (per cargo type) and randomize into scenarios.

    Parameters:
    - T_multi: dict {cargo_type: expected OD matrix}
    - dist: distribution type: "poisson", "neg_binomial", "lognormal", "normal", "uniform"
    - n_scenarios: number of random scenarios to generate
    - dispersion: for neg_binomial (larger -> closer to Poisson)
    - sigma: stdev factor for lognormal/normal/uniform
    - seed: random seed

    Returns:
    - scenarios: list of dicts {cargo_type: randomized OD matrix}
    """
    if seed is not None:
        np.random.seed(seed)

    scenarios = []

    for s in range(n_scenarios):
        scenario = {}
        for ctype, T_exp in T_multi.items():
            T_rand = np.zeros_like(T_exp)
            for i in range(T_exp.shape[0]):
                for j in range(T_exp.shape[1]):
                    mu = T_exp[i, j]
                    if mu <= 0:
                        continue

                    # todo: check all distributions and add CV option to adjust variance
                    if dist == "poisson":
                        val = np.random.poisson(mu)
                    elif dist == "neg_binomial":
                        # mean = mu, var = mu + mu^2/dispersion
                        p = dispersion / (dispersion + mu)
                        r = dispersion
                        val = np.random.negative_binomial(r, p)
                    elif dist == "lognormal":
                        # lognormal with mean ≈ mu
                        val = int(np.random.lognormal(np.log(mu + 1e-6), sigma))
                    elif dist == "normal":
                        val = int(max(0, np.random.normal(mu, sigma * mu)))
                    elif dist == "uniform":
                        low = mu * (1 - sigma)
                        high = mu * (1 + sigma)
                        val = int(np.random.uniform(low, high))
                    else:
                        raise ValueError(f"Unknown distribution: {dist}")

                    T_rand[i, j] = val

            scenario[ctype] = T_rand
        scenarios.append(scenario)

    return scenarios


# ===== Example usage =====
P = 6
C = 20000
sparsity = 0.3
perturb = 0.2
loading_only = False
middle_leg = P // 2
dirichlet = True
alpha = 0.2

if loading_only:
    target_utils = [0.6, 0.8, 1.0]
else:
    target_utils = [0.6, 0.8, 1.0, 0.8, 0.6]  # Adjusted for middle_leg = 3

# Add random perturbation to target utils
target_utils *= np.random.uniform(0.9, 1.1, size=len(target_utils))
print("----------------")
print("Target utilizations:", target_utils)

# Loading-only authentic matrix
T_auth = generate_authentic_matrix(P, C, target_utils, middle_leg=middle_leg, loading_only=loading_only,
                                   sparsity=sparsity, perturb=perturb, dirichlet=dirichlet, alpha=alpha)
print("Authentic loading OD matrix:\n", T_auth)

# Track onboard containers per leg
onboard = np.zeros((P, P), dtype=int)
total_onboard_per_leg = []
for leg in range(P - 1):
    onboard[:, leg] = 0
    onboard[leg, leg + 1:] = T_auth[leg, leg + 1:]
    total_onboard_per_leg.append(np.sum(onboard))

print("Total containers on board per leg:", total_onboard_per_leg)
print("Utilization rate per leg:", [total / C for total in total_onboard_per_leg])

# # Run this multiple times to count sparsity effect
# sparsity_counts = []
# relevant_elements = (P * (P - 1)) // 2 if not loading_only else middle_leg * (P - middle_leg)
# for _ in range(100):
#     T_test = generate_authentic_matrix(P, C, target_utils, middle_leg=middle_leg, loading_only=loading_only,
#                                    sparsity=sparsity, perturb=perturb, dirichlet=dirichlet, alpha=alpha)
#     non_zero_count = np.count_nonzero(T_test)
#     sparse_count = relevant_elements  - non_zero_count
#     sparsity_counts.append(sparse_count)
#
# print("Average sparse OD pairs over 100 runs:", np.mean(sparsity_counts) )
# print("Average (%) sparsity over 100 runs:", np.mean(sparsity_counts) / relevant_elements )

T_multi, cargo_types = generate_multicargo_matrix(
    P, C, target_utils, middle_leg=P//2,
    loading_only=False, sparsity=0.2, perturb=0.15,
    dirichlet=True, alpha=0.3, include_reefer=True
)

print("----------------")
print("Generated cargo types:", cargo_types)
print("OD matrix for (40ft, medium, spot):")
print(T_multi[("40ft", "medium", "spot")])

# Assume T_multi from generate_multicargo_matrix
scenarios = randomize_demand_matrix(T_multi, dist="neg_binomial", n_scenarios=5, sigma=0.4)

print("----------------")
print("Scenario 1, cargo type (40ft, medium, spot):")
print(scenarios[0][("40ft", "medium", "spot")])