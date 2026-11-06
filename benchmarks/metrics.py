"""Evaluation metrics: classifier two-sample test, conditional log density,
compatibility of supplied statements, and calibration of conditional posteriors.
"""

import math

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier


def conditional_log_prob(model, x, condition_mask, theta, n_steps=30,
                         v_stmt=None, sigma_stmt=None):
    """Conditional log-density log q(theta_unknown | x, theta_known) under the
    model's flow, via the instantaneous change of variables with EXACT divergence
    (the parameter dim is small, so the full trace is cheap). Computed in the
    model's native (transformed) space.

    This is a strictly proper score. Compared ACROSS models that share the same
    parameter transform, a higher value means a conditional closer to the true one
    (it equals the negative KL to the true conditional up to a model-independent
    constant). The absolute value carries that constant and is not meaningful alone.

    v_stmt/sigma_stmt (each (d,), soft models only): evaluate under statement
    conditioning "about v with tolerance sigma" instead of exact conditioning on
    theta_known. The known slots then follow sample(): the flow state holds the
    clamped statement value and the token carries the raw (v, sigma).

    x/condition_mask/theta may be single (1D) or batched (2D). Returns (B,) log-probs.
    """
    theta = theta.clone()
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)
    B, d = theta.shape
    if x.ndim == 1:
        x = x.unsqueeze(0).expand(B, -1)
    if condition_mask.ndim == 1:
        condition_mask = condition_mask.unsqueeze(0).expand(B, -1)
    device = theta.device
    x = x.to(device)
    condition_mask = condition_mask.to(device)
    unknown_mask = 1.0 - condition_mask

    transform = getattr(model, "param_transform", None)
    z = transform.forward(theta) if transform is not None else theta   # native space

    # For a soft-conditioning model the known tokens carry a statement (v, sigma),
    # not the value itself; exact conditioning is sigma = 0. Build the statement
    # from the known values in [0, 1], as sample() does.
    statements = None
    if getattr(model, "soft_conditioning", False):
        if v_stmt is not None:
            sig = (torch.zeros_like(v_stmt) if sigma_stmt is None
                   else sigma_stmt).to(device)
            vb = v_stmt.to(device).unsqueeze(0).expand(B, -1)
            sb = sig.unsqueeze(0).expand(B, -1)
            stmt = torch.stack([vb, sb], dim=-1)
            # known flow slots hold the clamped statement value, as in sample()
            theta = theta * unknown_mask + vb.clamp(0, 1) * condition_mask
            z = transform.forward(theta) if transform is not None else theta
        else:
            stmt = torch.stack([theta, torch.zeros_like(theta)], dim=-1)
        statements = stmt * condition_mask.unsqueeze(-1)

    dt = 1.0 / n_steps
    trace_int = torch.zeros(B, device=device)
    for step in range(n_steps):
        t_val = 1.0 - (step + 0.5) * dt        # integrate backward, t: 1 -> 0
        t = torch.full((B,), t_val, device=device)
        z = z.detach().requires_grad_(True)
        v = model.forward(z, t, x, condition_mask, statements=statements)
        tr = torch.zeros(B, device=device)
        if v.requires_grad:                    # a constant velocity has zero divergence
            for j in range(d):                 # exact trace of d v_u / d z_u
                gj = torch.autograd.grad(v[:, j].sum(), z, retain_graph=(j < d - 1),
                                         allow_unused=True)[0]
                if gj is not None:             # None == v_j independent of z (zero)
                    tr = tr + gj[:, j] * unknown_mask[:, j]
        trace_int = trace_int + dt * tr.detach()
        z = (z - dt * v * unknown_mask).detach()   # backward Euler step

    zu = z * unknown_mask
    n_u = unknown_mask.sum(1)
    log_base = -0.5 * (zu ** 2).sum(1) - 0.5 * n_u * math.log(2 * math.pi)
    return (log_base - trace_int).detach()


def c2st(samples_p, samples_q, n_folds=5, seed=42):
    """Classifier Two-Sample Test.

    Trains a classifier to distinguish samples from P vs Q.
    Returns accuracy in [0.5, 1.0]. 0.5 = indistinguishable (perfect).

    """
    if isinstance(samples_p, torch.Tensor):
        samples_p = samples_p.cpu().numpy()
    if isinstance(samples_q, torch.Tensor):
        samples_q = samples_q.cpu().numpy()

    n = min(len(samples_p), len(samples_q))
    samples_p = samples_p[:n]
    samples_q = samples_q[:n]

    X = np.concatenate([samples_p, samples_q], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accuracies = []

    for train_idx, test_idx in kf.split(X):
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=10,
        )
        clf.fit(X[train_idx], y[train_idx])
        accuracies.append(clf.score(X[test_idx], y[test_idx]))

    return float(np.mean(accuracies))


def statement_compatibility(model, x, condition_mask, v_stmt, sigma_stmt,
                            n_posterior=1000, n_steps=50, n_ref=256,
                            sigma_floor=0.01, device="cpu", seed=0):
    """How compatible is a soft-conditioning statement with the data?

    Soft conditioning is only trustworthy for statements the data does not
    flatly contradict (conflicting statements are low-density in the training
    joint, so the model extrapolates there). This diagnostic scores the
    statement's likelihood under the model's OWN full posterior p(theta_k | x),
    then calibrates it against statements generated FROM that posterior:

      score(v)   = log (1/N) sum_n prod_i N(v_i ; theta_k_i^(n), sigma_i^2)
      percentile = fraction of posterior-generated statements scoring <= score(v)

    Returns (log_score, percentile). Percentile near 0 = the hypothesis
    conflicts with the measurement -> flag rather than trust the conditional.
    sigma=0 (hard) coordinates use sigma_floor so the density is defined.
    """
    torch.manual_seed(seed)
    dim = condition_mask.shape[0]
    k = torch.where(condition_mask == 1)[0]
    sig = sigma_stmt[k].clamp_min(sigma_floor)

    # Full posterior p(theta | x): condition on nothing.
    full = model.sample(x.to(device), torch.zeros(dim), torch.zeros(dim),
                        n_samples=n_posterior, n_steps=n_steps).cpu()
    samp = full[:, k]                                          # (N, n_k)
    log_norm = -0.5 * math.log(2 * math.pi) * k.numel() - torch.log(sig).sum()

    def logscore(vv):                                          # vv: (..., n_k)
        ll = -0.5 * (((vv.unsqueeze(-2) - samp) / sig) ** 2).sum(-1) + log_norm
        return torch.logsumexp(ll, dim=-1) - math.log(samp.shape[0])

    s_act = logscore(v_stmt[k])
    idx = torch.randint(0, samp.shape[0], (n_ref,))
    v_ref = samp[idx] + sig * torch.randn(n_ref, k.numel())    # compatible statements
    s_ref = logscore(v_ref)
    percentile = float((s_ref <= s_act).float().mean())
    return float(s_act), percentile


def coverage_at_levels(ranks, n_posterior_samples, nominal_levels=None):
    """Empirical coverage from SBC ranks."""
    if nominal_levels is None:
        nominal_levels = np.array([0.5, 0.68, 0.9, 0.95, 0.99])

    n_test, n_params = ranks.shape
    coverage = np.zeros((len(nominal_levels), n_params))

    for j, level in enumerate(nominal_levels):
        lo = (1 - level) / 2
        hi = 1 - lo
        normalized = ranks / n_posterior_samples
        in_interval = (normalized > lo) & (normalized < hi)
        coverage[j] = in_interval.mean(0)

    return nominal_levels, coverage


def conditional_sbc(model, task, n_test=300, n_posterior=1000, n_steps=50,
                    device="cpu", nominal=(0.5, 0.68, 0.9, 0.95), seed=0,
                    testset=None, solver="midpoint"):
    """Simulation-based calibration for conditional posteriors, without a reference.

    For each test draw: take a (theta*, x) pair, fix a random subset of theta to
    its true values, sample the model's conditional posterior of the remaining
    parameters, and record (a) the rank of each true unknown parameter among the
    samples and (b) whether it falls inside each central credible interval. A
    calibrated model gives uniform ranks and empirical coverage equal to nominal.
    Needs only true (theta*, x) pairs, no reference posterior, so it works on any
    task including SAXS.

    Source of the (theta*, x) pairs:
      - ``testset=(theta, x)``: draw from a fixed held-out set (used for SAXS),
      - otherwise ``task.generate_dataset(1)`` (the benchmark simulators).

    Returns per-level coverage, a calibration error (mean |empirical - nominal|),
    pooled ranks, plus recovery RMSE, TARP joint calibration, and the log-prob
    proper score.
    """
    torch.manual_seed(seed)
    dim = task.dim_params if task is not None else testset[0].shape[1]
    covered = {lvl: [0, 0] for lvl in nominal}   # level -> [in_interval, total]
    norm_ranks = []
    # Joint (TARP) + recovery accumulators. Both reference-free and computed from
    # the same posterior samples, so no extra sampling.
    recovery_sq_sum = 0.0            # sum of squared (post-mean - truth) over unknown dims
    recovery_dim_count = 0
    tarp_f = []                      # per-draw TARP credibility of the true point
    tarp_levels = np.linspace(0.05, 0.95, 19)
    lp_th, lp_x, lp_cm = [], [], []  # conditionings for the log-prob proper score

    for _ in range(n_test):
        if testset is not None:
            j = int(torch.randint(0, testset[0].shape[0], (1,)).item())
            theta_true = testset[0][j].float()
            x = testset[1][j].float()
        else:
            theta_true, x = task.generate_dataset(1)
            theta_true = theta_true.squeeze(0)
            x = x.squeeze(0)

        n_known = int(torch.randint(1, dim, (1,)).item())
        perm = torch.randperm(dim)
        known = perm[:n_known]
        unknown = perm[n_known:]

        cmask = torch.zeros(dim)
        cmask[known] = 1.0
        lp_th.append(theta_true); lp_x.append(x); lp_cm.append(cmask)
        samples = model.sample(x.to(device), cmask, theta_true,
                               n_samples=n_posterior, n_steps=n_steps,
                               solver=solver).cpu()

        # --- marginal SBC (per unknown dimension) ---
        for j in unknown.tolist():
            sj = samples[:, j]
            tj = theta_true[j]
            norm_ranks.append(float((sj < tj).float().mean()))
            for lvl in nominal:
                lo = torch.quantile(sj, (1 - lvl) / 2)
                hi = torch.quantile(sj, 1 - (1 - lvl) / 2)
                covered[lvl][1] += 1
                if lo <= tj <= hi:
                    covered[lvl][0] += 1

        # --- joint diagnostics over the unknown block ---
        su = samples[:, unknown]              # (n_post, n_unknown) joint samples
        tu = theta_true[unknown]              # (n_unknown,) true unknown values
        # recovery: how close the posterior mean lands to the truth
        post_mean = su.mean(0)
        recovery_sq_sum += float(((post_mean - tu) ** 2).sum())
        recovery_dim_count += unknown.numel()
        # TARP: credibility of the true point vs random reference points. Distance-
        # based, so it sees the joint (correlations), unlike the per-dimension SBC.
        # References are ISOTROPIC Gaussians at a fixed scale around the posterior
        # mean: centered near the posterior (robust to where it sits in [0,1]),
        # fixed model-independent scale (so over/under-dispersion is still caught),
        # and isotropic so they probe the perpendicular/correlated directions that
        # prior- or uniform-shaped references miss. Several per draw.
        n_ref = 15
        refs = post_mean.unsqueeze(0) + 0.10 * torch.randn(n_ref, unknown.numel())
        for r in range(n_ref):
            ref = refs[r]
            d_star = torch.norm(tu - ref)
            d_samp = torch.norm(su - ref.unsqueeze(0), dim=1)
            tarp_f.append(float((d_samp < d_star).float().mean()))

    coverage = {lvl: covered[lvl][0] / max(covered[lvl][1], 1) for lvl in nominal}
    calib_error = float(np.mean([abs(coverage[lvl] - lvl) for lvl in nominal]))

    recovery_rmse = float(np.sqrt(recovery_sq_sum / max(recovery_dim_count, 1)))
    tarp_f = np.array(tarp_f)
    tarp_ecp = np.array([(tarp_f <= a).mean() for a in tarp_levels])
    tarp_calib_error = float(np.mean(np.abs(tarp_ecp - tarp_levels)))
    tarp_cov90 = float((tarp_f <= 0.9).mean())

    # Conditional log-prob proper score (higher = closer to the true conditional).
    # One batched change-of-variables pass over all collected conditionings.
    logprob_score = float("nan")
    if lp_th:
        TH = torch.stack(lp_th).to(device)
        X = torch.stack(lp_x).to(device)
        CM = torch.stack(lp_cm).to(device)
        logprob_score = float(conditional_log_prob(model, X, CM, TH, n_steps=30).mean())

    return {
        "nominal": list(nominal),
        "coverage": [coverage[lvl] for lvl in nominal],
        "calibration_error": calib_error,
        "norm_ranks": np.array(norm_ranks, dtype=np.float32),
        "n_evaluated": len(norm_ranks),
        "recovery_rmse": recovery_rmse,
        "tarp_calibration_error": tarp_calib_error,
        "tarp_cov90": tarp_cov90,
        "logprob_score": logprob_score,
    }
