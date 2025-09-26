import torch
import math
import random
from torch.distributions import Dirichlet, Multinomial, Poisson, NegativeBinomial, Normal, Uniform

# -----------------------
# Partitions (torch)
# -----------------------

def random_integer_partition_torch(v: int, b: int, device='cpu'):
    """
    Partition integer v into b non-negative integers using the Fisher-Yates style algorithm.
    Returns a torch.long 1D tensor of length b summing to v.
    """
    if b == 1:
        return torch.tensor([v], dtype=torch.long, device=device)
    n = v + b - 1
    # create y = [1..n]
    y = torch.arange(1, n + 1, dtype=torch.long, device=device)
    # shuffle indices and take first b-1 of y (emulating the algorithm)
    perm = torch.randperm(n, device=device)
    first_idx = perm[: (b - 1)]
    y_first = y[first_idx]
    y_first_sorted, _ = torch.sort(y_first)
    yvals = y_first_sorted.to(torch.long).tolist()
    x = []
    x.append(int(yvals[0] - 1))
    for i in range(1, b - 1):
        x.append(int(yvals[i] - yvals[i - 1] - 1))
    x.append(int((v + b - 1) - yvals[b - 2]))
    return torch.tensor(x, dtype=torch.long, device=device)


def dirichlet_partition_torch(v: int, b: int, alpha: float = 1.0, weights=None, device='cpu'):
    """
    Partition integer v into b nonnegative integers using Dirichlet -> Multinomial (torch).
    Returns a torch.long 1D tensor length b summing to v.
    """
    if b == 1:
        return torch.tensor([v], dtype=torch.long, device=device)
    device = torch.device(device)
    if weights is None:
        alphas = torch.full((b,), float(alpha), dtype=torch.float32, device=device)
    else:
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        alphas = w * float(alpha)
    probs = Dirichlet(alphas).sample()
    # Multinomial sample (vector)
    sample = Multinomial(total_count=v, probs=probs).sample()
    return sample.to(torch.long)


# -----------------------
# Core generator
# -----------------------

def generate_authentic_matrix(P: int,
                              C: int,
                              target_utils,
                              middle_leg: int = None,
                              loading_only: bool = False,
                              sparsity: float = 0.3,
                              perturb: float = 0.2,
                              dirichlet: bool = True,
                              alpha: float = 1.0,
                              device='cpu'):
    """
    Generate an authentic OD matrix using torch.
    - P: total ports
    - C: bay capacity (int)
    - target_utils: list-like of floats; length = n_loading
        - if loading_only: n_loading = middle_leg
        - else: n_loading = P-1 (rows 0..P-2)
    - loading_only: restrict origins to loading block
    - middle_leg: index separating loading/discharging
    - sparsity: probability each OD cell is zeroed (0..1)
    - perturb: fraction for +/- perturbation applied to cell values
    - dirichlet: use dirichlet partition (True) or integer partition (False)
    - alpha: Dirichlet concentration
    Returns: T (P x P) torch.LongTensor
    """
    device = torch.device(device)
    T = torch.zeros((P, P), dtype=torch.long, device=device)

    if middle_leg is None:
        middle_leg = P // 2

    n_loading = middle_leg if loading_only else (P - 1)
    if len(target_utils) != n_loading:
        raise ValueError(f"target_utils length {len(target_utils)} != expected n_loading {n_loading}")

    # copy target utils as floats
    tutils = [float(x) for x in target_utils]

    for i in range(n_loading):
        dest_start = middle_leg if loading_only else (i + 1)
        b = P - dest_start
        if b <= 0:
            continue

        # already_assigned: sum of previous rows into these dest columns
        if i > 0:
            already_assigned = int(torch.sum(T[:i, dest_start:]).item())
        else:
            already_assigned = 0

        v_target = int(round(tutils[i] * C))
        v = max(v_target - already_assigned, 0)
        if v == 0:
            continue

        # Partition v into b parts
        if dirichlet:
            part = dirichlet_partition_torch(v, b, alpha=alpha, device=device)
        else:
            part = random_integer_partition_torch(v, b, device=device)

        # apply sparsity (vectorized): draw bernoulli mask
        if sparsity > 0:
            mask = (torch.rand(b, device=device) >= sparsity).to(torch.long)  # 1 = keep, 0 = zero
            part = part * mask

        # if all zeroed, force one destination to get v
        if torch.sum(part).item() == 0:
            idx = random.randrange(0, b)
            part[idx] = v

        # apply perturbation: for non-zero cells, add small +/- delta
        if perturb > 0:
            # compute deltas as floats then convert to ints
            nonzero_mask = (part > 0)
            # uniform in [-perturb, +perturb]
            randf = torch.rand(b, device=device) * 2.0 - 1.0  # in [-1,1]
            deltas = (part.to(torch.float32) * randf * perturb).round().to(torch.long)
            deltas = deltas * nonzero_mask.to(torch.long)
            part = torch.clamp(part + deltas, min=0)

        # rescale partition to sum to v (handle rounding)
        s = int(torch.sum(part).item())
        if s == 0:
            # fallback: one bucket gets v
            final = torch.zeros(b, dtype=torch.long, device=device)
            final[random.randrange(0, b)] = v
        else:
            # scale with integer rounding
            # compute float scaling and round
            scaled = (part.to(torch.float32) * (float(v) / float(s)))
            final = torch.floor(scaled + 0.5).to(torch.long)  # round
            diff = v - int(torch.sum(final).item())
            # correct difference by distributing +1 or -1
            if diff != 0:
                # get indices sorted by fractional part to adjust
                frac = (scaled - torch.floor(scaled)).cpu().numpy()
                idxs = list(range(b))
                # if diff > 0, add to largest fractional parts; if diff < 0, remove from smallest fractional parts with positive final
                if diff > 0:
                    order = sorted(idxs, key=lambda k: -frac[k])
                    k = 0
                    while diff > 0 and k < b:
                        final[order[k]] += 1
                        diff -= 1
                        k += 1
                else:
                    order = sorted(idxs, key=lambda k: frac[k])
                    k = 0
                    while diff < 0 and k < b:
                        if final[order[k]] > 0:
                            final[order[k]] -= 1
                            diff += 1
                        k += 1
                    # if still diff not zero (rare), distribute remaining adjustments arbitrarily
                    k = 0
                    while diff < 0 and k < b:
                        if final[k] > 0:
                            final[k] -= 1
                            diff += 1
                        k += 1

        # assign into matrix block
        T[i, dest_start: dest_start + b] = final

    return T


# -----------------------
# multi-cargo and scenarios
# -----------------------

def generate_multicargo_matrix(P: int,
                               C: int,
                               target_utils,
                               middle_leg=None,
                               loading_only=False,
                               sparsity=0.3,
                               perturb=0.2,
                               dirichlet=True,
                               alpha=0.5,
                               cargo_shares=None,
                               include_reefer=True,
                               device='cpu'):
    """
    Generate OD matrices for multiple cargo types.
    Returns dict mapping cargo_type -> torch.LongTensor (P x P) and list of cargo types.
    cargo_types are tuples (size, weight, revenue) and optionally 'reefer' appended as fourth element.
    """
    # define cargo taxonomy
    sizes = ["20ft", "40ft"]
    weights = ["light", "medium", "heavy"]
    revenues = ["long", "spot"]
    cargo_types = [(s, w, r) for s in sizes for w in weights for r in revenues]
    if include_reefer:
        cargo_types += [("40ft", "medium", "long", "reefer"), ("40ft", "medium", "spot", "reefer")]

    K = len(cargo_types)
    if cargo_shares is None:
        shares = torch.ones(K, dtype=torch.float32) / float(K)
    else:
        shares = torch.tensor(cargo_shares, dtype=torch.float32)
        shares = shares / shares.sum()

    T_multi = {}
    for k, ctype in enumerate(cargo_types):
        Ck = int(round(float(C) * float(shares[k].item())))
        T_multi[ctype] = generate_authentic_matrix(
            P, Ck, target_utils,
            middle_leg=middle_leg,
            loading_only=loading_only,
            sparsity=sparsity,
            perturb=perturb,
            dirichlet=dirichlet,
            alpha=alpha,
            device=device
        )
    return T_multi, cargo_types


def randomize_demand_matrix(T_multi: dict,
                            dist: str = "poisson",
                            n_scenarios: int = 10,
                            dispersion: float = 1.0,
                            sigma: float = 0.3,
                            seed: int = None,
                            device='cpu'):
    """
    Randomize expected OD matrices into scenarios.
    - T_multi: dict {ctype: torch.LongTensor}
    - dist: 'poisson', 'neg_binomial', 'lognormal', 'normal', 'uniform'
    Returns: list of scenario dicts (same keys as T_multi) with randomized torch.LongTensor matrices.
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    device = torch.device(device)
    scenarios = []

    # todo: vectorize this!
    for s in range(n_scenarios):
        scenario = {}
        for ctype, T_exp in T_multi.items():
            T_exp = T_exp.to(device)
            P = T_exp.shape[0]
            T_rand = torch.zeros_like(T_exp, device=device)
            # vectorize by flattening indices
            flat = T_exp.view(-1).to(torch.float32)
            nz_mask = flat > 0
            mu_vals = flat.clone()

            # todo: check all distributions and add CV option to adjust variance
            if dist == "poisson":
                # Poisson per element
                # torch.poisson expects a tensor of rates
                draws = torch.poisson(mu_vals)
                T_rand = draws.view(P, P).to(torch.long)
            elif dist == "neg_binomial":
                # approximate using Gamma-Poisson mixture: sample lambda ~ Gamma(r, p/(1-p)), then Poisson(lambda)
                # we want Var = mu + mu^2/dispersion => r = dispersion, p = r/(r+mu)
                r = dispersion
                # For vectorized, compute p per element
                mu = mu_vals
                p = r / (r + mu)
                # PyTorch NegativeBinomial takes total_count (r) and probs=p
                # It returns NB draws directly (non-negative ints)
                nb = NegativeBinomial(total_count=torch.clamp(torch.tensor(r, device=device), min=1e-6),
                                      probs=torch.clamp(p, min=1e-6, max=1 - 1e-6))
                draws = nb.sample(mu.shape)
                T_rand = draws.view(P, P).to(torch.long)
            elif dist == "lognormal":
                # approximate: draw lognormal with mean approx mu, but must map params
                # Use Normal on log-space with sigma factor
                # avoid zeros by adding small eps
                eps = 1e-6
                mu = mu_vals + eps
                # We compute logmean so that exp(mean + 0.5*sigma^2) = mu -> mean = log(mu) - 0.5*sigma^2
                logmean = torch.log(mu) - 0.5 * (sigma ** 2)
                normal = Normal(logmean, sigma)
                draws = torch.exp(normal.sample()).round()
                T_rand = draws.view(P, P).to(torch.long)
            elif dist == "normal":
                mu = mu_vals
                draws = Normal(mu, sigma * torch.clamp(mu, min=1.0)).sample().clamp(min=0).round()
                T_rand = draws.view(P, P).to(torch.long)
            elif dist == "uniform":
                low = mu_vals * (1 - sigma)
                high = mu_vals * (1 + sigma)
                draws = Uniform(low, high).sample().round().clamp(min=0)
                T_rand = draws.view(P, P).to(torch.long)
            else:
                raise ValueError(f"Unknown dist: {dist}")

            scenario[ctype] = T_rand
        scenarios.append(scenario)

    return scenarios


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    P = 6
    C = 20000
    middle_leg = 3
    loading_only = False
    sparsity = 0.25
    perturb = 0.12
    dirichlet = True
    alpha = 0.3

    if loading_only:
        target_utils = [0.6, 0.8, 1.0]  # length == middle_leg
    else:
        target_utils = [0.6, 0.8, 1.0, 0.8, 0.6]

    # small randomization of targets
    target_utils = (torch.tensor(target_utils) * torch.empty(len(target_utils)).uniform_(0.95, 1.05)).tolist()
    print("----------------")
    print("Target utilizations:", target_utils)

    T_auth = generate_authentic_matrix(P, C, target_utils,
                                       middle_leg=middle_leg,
                                       loading_only=loading_only,
                                       sparsity=sparsity,
                                       perturb=perturb,
                                       dirichlet=dirichlet,
                                       alpha=alpha,
                                       device='cpu')

    print("Authentic loading OD matrix (torch -> numpy shown):")
    print(T_auth.cpu().numpy())

    # compute onboard per leg properly for loading_only block
    # onboard vector: positions correspond to destinations (ports). We'll track containers currently onboard after each leg.
    onboard = torch.zeros(P, dtype=torch.long)
    total_onboard_per_leg = []
    for leg in range(P - 1):
        # discharge containers destined to current port (leg)
        # containers destined to port 'leg' should be removed (they left when we arrived at that port)
        # ensure indices: onboard[pos] corresponds to containers with destination pos
        onboard[leg] = 0
        # if this is a loading port, load new containers from that origin (row leg)
        if leg < middle_leg:
            # add all containers loaded at this port destined to later ports
            onboard[leg + 1:] += T_auth[leg, leg + 1:]
        total_onboard_per_leg.append(int(onboard.sum().item()))

    print("Total containers on board per leg:", total_onboard_per_leg)
    print("Utilization rate per leg:", [x / C for x in total_onboard_per_leg])

    # Generate multicargo and scenarios demo
    T_multi, cargo_types = generate_multicargo_matrix(
        P, C, target_utils,
        middle_leg=middle_leg,
        loading_only=loading_only,
        sparsity=0.2,
        perturb=0.15,
        dirichlet=True,
        alpha=0.3,
        include_reefer=True,
        device='cpu'
    )

    print("Generated cargo types (sample):", cargo_types[:4])
    sample_type = cargo_types[0]
    print("Sample OD matrix for", sample_type)
    print(T_multi[sample_type].cpu().numpy())

    scenarios = randomize_demand_matrix(T_multi, dist="poisson", n_scenarios=3, seed=42)
    print("Scenario 0 for sample type (poisson):")
    print(scenarios[0][sample_type].cpu().numpy())
