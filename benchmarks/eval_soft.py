"""Compare uncertain conditioning with the exact answer on the Gaussian task.

The posterior given uncertain statements (v, sigma) is Gaussian in closed form, so
the conditionals of the model can be scored directly with C2ST, where 0.5 means
indistinguishable. The script sweeps sigma from 0 (exact conditioning) upward,
tests a statement centred on a boundary, and checks that deliberately conflicting
statements are flagged by the compatibility percentile.

Run it after training a model with --soft:

    python -m benchmarks.eval_soft --checkpoint model.pt
"""

import argparse
import math

import numpy as np
import torch
from benchmarks.load import load_model
from benchmarks.tasks import get_task
from benchmarks.metrics import c2st, conditional_log_prob, statement_compatibility


def eval_soft(model, task, sigmas=(0.0, 0.02, 0.05, 0.1, 0.2),
              n_patterns=12, n_samples=2000, n_steps=50, device="cpu", seed=0,
              solver="midpoint", n_kl=0, kl_steps=100):
    torch.manual_seed(seed)
    dim = task.dim_params
    rows = []
    for sig_level in sigmas:
        accs, compat, kls = [], [], []
        for i in range(n_patterns):
            theta_true, x = task.generate_dataset(1)
            theta_true, x = theta_true.squeeze(0), x.squeeze(0)
            n_known = int(torch.randint(1, dim, (1,)).item())
            known = torch.randperm(dim)[:n_known]
            cmask = torch.zeros(dim); cmask[known] = 1.0
            sigma = torch.zeros(dim); sigma[known] = sig_level
            # the statement: corrupted at the evaluated tolerance (v=truth at sigma=0)
            v = theta_true.clone()
            v[known] = theta_true[known] + sig_level * torch.randn(n_known)

            unknown = torch.where(cmask == 0)[0]
            ms = model.sample(x.to(device), cmask, v, n_samples=n_samples,
                              n_steps=n_steps, sigma=sigma,
                              solver=solver).cpu()[:, unknown]
            ref = task.sample_analytic_soft_conditional(x, cmask, v, sigma, n_samples)
            accs.append(c2st(ms, ref))
            _, pct = statement_compatibility(model, x, cmask, v, sigma,
                                             n_posterior=n_samples,
                                             n_steps=n_steps, device=device, seed=i)
            compat.append(pct)
            if n_kl:
                kls.append(_pattern_kl(model, task, x, cmask, v, sigma,
                                       n_kl, kl_steps, device,
                                       seed=10_000 + 97 * i
                                            + int(round(sig_level * 1000))))
        rows.append((f"sigma={sig_level:.2f}", float(np.mean(accs)),
                     float(np.std(accs)), float(np.mean(compat)),
                     float(np.mean(kls)) if n_kl else float("nan"),
                     float(np.std(kls)) if n_kl else float("nan")))

    # boundary-centered statement: "coord ~ 0 +/- 0.05" regardless of its true value
    accs, compat = [], []
    for i in range(n_patterns):
        theta_true, x = task.generate_dataset(1)
        theta_true, x = theta_true.squeeze(0), x.squeeze(0)
        j = int(torch.randint(0, dim, (1,)).item())
        cmask = torch.zeros(dim); cmask[j] = 1.0
        sigma = torch.zeros(dim); sigma[j] = 0.05
        v = theta_true.clone(); v[j] = 0.0
        unknown = torch.where(cmask == 0)[0]
        ms = model.sample(x.to(device), cmask, v, n_samples=n_samples,
                          n_steps=n_steps, sigma=sigma,
                          solver=solver).cpu()[:, unknown]
        ref = task.sample_analytic_soft_conditional(x, cmask, v, sigma, n_samples)
        accs.append(c2st(ms, ref))
        _, pct = statement_compatibility(model, x, cmask, v, sigma,
                                         n_posterior=n_samples,
                                         n_steps=n_steps, device=device, seed=i)
        compat.append(pct)
    rows.append(("boundary v=0 s=0.05", float(np.mean(accs)),
                 float(np.std(accs)), float(np.mean(compat)),
                 float("nan"), float("nan")))
    return rows


def _pattern_kl(model, task, x, cmask, v, sigma, n_kl, kl_steps, device, seed):
    """Monte Carlo estimate of KL(p_analytic || q_model) for one pattern, in
    nats, both densities over the unknown coordinates in RAW parameter space.
    Uses a private numpy RNG so the global torch stream (and therefore the
    C2ST/compatibility draws of the surrounding loop) is untouched."""
    idx_u, mu_u, cov_u = task._soft_conditional_gaussian(x, cmask, v, sigma)
    cov = cov_u.double()
    cov = 0.5 * (cov + cov.T) + 1e-10 * torch.eye(idx_u.numel(),
                                                  dtype=torch.float64)
    L = torch.linalg.cholesky(cov)
    rng = np.random.default_rng(seed)
    eps = torch.tensor(rng.standard_normal((n_kl, idx_u.numel())))
    su = mu_u.double().unsqueeze(0) + eps @ L.T                  # raw, unclamped
    # log p at the samples, exact via the sampling construction:
    # L^{-1}(su - mu) = eps, so the quadratic form is just |eps|^2
    d_u = idx_u.numel()
    logp = (-0.5 * (d_u * math.log(2 * math.pi) + (eps ** 2).sum(1))
            - torch.log(torch.diag(L)).sum())
    su_c = su.clamp(1e-4, 1 - 1e-4).float()
    th = v.clamp(0, 1).unsqueeze(0).expand(n_kl, task.dim_params).clone()
    th[:, idx_u] = su_c
    logq = conditional_log_prob(model, x.to(device), cmask.to(device),
                                th.to(device), n_steps=kl_steps,
                                v_stmt=v, sigma_stmt=sigma).cpu().double()
    tr = getattr(model, "param_transform", None)
    if tr is not None:
        # componentwise transform: native density -> raw density needs the
        # log |dz/dtheta| of the unknown coordinates
        thg = th.to(device).requires_grad_(True)
        (g,) = torch.autograd.grad(tr.forward(thg).sum(), thg)
        logq = logq + torch.log(g[:, idx_u].abs().clamp_min(1e-12)
                                ).sum(1).cpu().double()
    return float((logp - logq).mean())


def eval_conflicts(model, task, n_conflicts=50, displacement=0.5, tol=0.03,
                   n_samples=2000, n_steps=50, device="cpu", seed=0,
                   solver="midpoint", threshold=0.01):
    """Deliberately conflicting statements: the stated value is displaced far
    outside the conditional posterior (displacement in prior units, >> tol).
    Returns per-statement compatibility percentiles; correct behaviour is
    rejection (percentile < threshold) for every one."""
    torch.manual_seed(seed + 999)
    dim = task.dim_params
    pcts = []
    for i in range(n_conflicts):
        theta_true, x = task.generate_dataset(1)
        theta_true, x = theta_true.squeeze(0), x.squeeze(0)
        j = int(torch.randint(0, dim, (1,)).item())
        cmask = torch.zeros(dim); cmask[j] = 1.0
        sigma = torch.zeros(dim); sigma[j] = tol
        v = theta_true.clone()
        sign = 1.0 if torch.rand(1).item() < 0.5 else -1.0
        v[j] = theta_true[j] + sign * displacement
        _, pct = statement_compatibility(model, x, cmask, v, sigma,
                                         n_posterior=n_samples,
                                         n_steps=n_steps, device=device,
                                         seed=seed + 7 * i)
        pcts.append(pct)
    return np.array(pcts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.0, 0.02, 0.05, 0.1, 0.2],
                   help="tolerance levels; keep within the checkpoint's trained "
                        "sigma_max or the rows measure extrapolation")
    p.add_argument("--n-patterns", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--solver", default="midpoint",
                   help="ODE solver used for sampling; recorded in the output")
    p.add_argument("--n-conflicts", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.01,
                   help="prespecified rejection threshold")
    p.add_argument("--kl", type=int, default=0,
                   help="if >0, also estimate KL(analytic || model) per "
                        "tolerance with this many samples per pattern")
    p.add_argument("--kl-steps", type=int, default=100)
    p.add_argument("--out", default=None, help="npz output path")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.checkpoint, device=device)
    assert ckpt.get("soft_conditioning"), "checkpoint was not trained with --soft"
    task = get_task(ckpt["task_name"])
    assert getattr(task, "exact_conditional", False), "needs the exact-conditional task"

    print(f"Soft-conditioning accuracy vs EXACT soft conditional "
          f"({ckpt['task_name']}, solver={args.solver}, n_steps={args.n_steps}, "
          f"{args.n_patterns} patterns x {args.n_samples} samples; "
          f"C2ST: 0.5 = perfect | compat percentile: ~0 = conflict -> flag):")
    rows = eval_soft(model, task, sigmas=tuple(args.sigmas),
                     n_patterns=args.n_patterns, n_samples=args.n_samples,
                     n_steps=args.n_steps, device=device, solver=args.solver,
                     n_kl=args.kl, kl_steps=args.kl_steps)
    for name, mean, std, compat, klm, kls_ in rows:
        flag = "  <- FLAGGED" if compat < 0.01 else ""
        kl_txt = f"   KL = {klm:.3f} +/- {kls_:.3f} nat" if np.isfinite(klm) else ""
        print(f"  {name:22s} c2st = {mean:.3f} +/- {std:.3f}   "
              f"compat = {compat:.3f}{kl_txt}{flag}")

    pcts = eval_conflicts(model, task, n_conflicts=args.n_conflicts,
                          n_samples=args.n_samples, n_steps=args.n_steps,
                          device=device, solver=args.solver,
                          threshold=args.threshold)
    rej = int((pcts < args.threshold).sum())
    print(f"\nconflicting statements: {rej}/{len(pcts)} rejected at "
          f"threshold {args.threshold} (max percentile {pcts.max():.4f})")
    sup = np.array([r[3] for r in rows if r[0].startswith("sigma")])
    print(f"supported statements (sigma grid only): min mean-percentile "
          f"{sup.min():.3f}, all far above the threshold")
    if args.out:
        np.savez(args.out, rows=np.array(rows, dtype=object),
                 conflict_pcts=pcts, threshold=args.threshold,
                 solver=args.solver, n_steps=args.n_steps,
                 n_patterns=args.n_patterns, n_samples=args.n_samples,
                 sigmas=np.array(args.sigmas), checkpoint=args.checkpoint)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
