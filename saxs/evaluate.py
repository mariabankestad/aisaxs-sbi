"""Evaluate trained models on held-out simulated structures.

  gap   The training loss on N training and N validation structures, with the
        same SLD and noise draws. A clear difference points to overfitting at
        the level of structures.

  test  Posterior samples for the test bank, at a fixed noise level, in the two
        main windows, with k random parameters known exactly. Reported for each
        seed and for the pooled ensemble: coverage of the central 50 % and 90 %
        intervals of the unknown parameters, and the error of the posterior
        median and the 90 % width, in [0, 1] units.

All seeds see identical curves, noise, masks and statements.

    python -m saxs.evaluate --ckpt c0.pt c1.pt c2.pt --test-bank TEST_DIR \
        --train-bank TRAIN_DIR --val-bank VAL_DIR --out results.npz
"""
import argparse
import time

import numpy as np
import torch
from saxs.bank import StructureBank
from saxs.infer import Posterior
from cfm.model import sample_condition_masks, sample_soft_statements

KNOWN_BINS = ((0, 0), (1, 3), (4, 6), (7, 8))


def noisy_x(post, log_I, widx, sigma, seed, bs=512):
    """Features with noise fixed by `seed`, so every checkpoint sees the same
    noise (the ingests differ only in the amplitude centring mu_ref)."""
    out = []
    for s in range(0, len(log_I), bs):
        torch.manual_seed(seed + s)
        sig = torch.full((min(bs, len(log_I) - s),), sigma, device=log_I.device) \
            if np.isscalar(sigma) else sigma[s:s + bs]
        out.append(post.ingest.augment(log_I[s:s + bs], widx, sig))
    return torch.cat(out)


# ---- gap ---------------------------------------------------------------------

@torch.no_grad()
def val_loss(post, theta, log_I, widx, seed, bs=256):
    """Training loss for one window; theta (N, 9) in [0, 1]."""
    m, dev = post.model, post.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(theta)
    sig = torch.exp(np.log(0.0007) + torch.rand(n, generator=g) * (np.log(0.010) - np.log(0.0007))).to(dev)
    x = noisy_x(post, log_I, widx, sig, seed + 7)
    torch.manual_seed(seed)
    mask = sample_condition_masks(n, n_params=theta.shape[1], device="cpu").to(dev)
    stm = sample_soft_statements(theta.cpu(), mask.cpu(), p_hard=post.ckpt["p_hard"],
                                 sigma_max=post.sigma_max, sigma_dist=post.ckpt["sigma_dist"]).to(dev)
    t = torch.rand(n).to(dev)
    noise = torch.randn(n, theta.shape[1]).to(dev)
    th = m.param_transform.forward(theta)
    per = []
    for s in range(0, n, bs):
        sl = slice(s, s + bs)
        th_in = torch.where(mask[sl].bool(), th[sl], (1 - t[sl, None]) * noise[sl] + t[sl, None] * th[sl])
        v = m(th_in, t[sl], x[sl], mask[sl], statements=stm[sl])
        lm = 1 - mask[sl]
        per.append((((v - (th[sl] - noise[sl])) ** 2) * lm).sum(1) / lm.sum(1).clamp_min(1))
    per = torch.cat(per).cpu().numpy()
    k = mask.sum(1).cpu().numpy()
    return per.mean(), [per[(k >= a) & (k <= b)].mean() for a, b in KNOWN_BINS]


# ---- posterior sampling, batched over cases -----------------------------------

@torch.no_grad()
def sample_cases(post, x, mask, theta_true, n_samples, n_steps, case_bs):
    """Midpoint flow for many cases at once; exact statements at theta_true on
    masked columns. Returns (C, n_samples, 9) in [0, 1] units."""
    m, dev = post.model, post.device
    tr = m.param_transform
    C, d = theta_true.shape
    out = torch.empty(C, n_samples, d)
    for s in range(0, C, case_bs):
        sl = slice(s, min(s + case_bs, C)); c = sl.stop - sl.start
        R = c * n_samples
        xb = x[sl].repeat_interleave(n_samples, 0)
        mb = mask[sl].repeat_interleave(n_samples, 0)
        tb = theta_true[sl].repeat_interleave(n_samples, 0)
        obs = m.encode_observation(x[sl]).repeat_interleave(n_samples, 0)
        stm = torch.stack([tb, torch.zeros_like(tb)], -1) * mb.unsqueeze(-1)
        um = 1 - mb
        z = torch.randn(R, d, device=dev) * um + tr.forward(tb.clamp(0, 1)) * mb
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t0 = i * dt
            k1 = m(z, torch.full((R,), t0, device=dev), xb, mb, obs_tokens=obs, statements=stm)
            k2 = m(z + 0.5 * dt * k1 * um, torch.full((R,), t0 + 0.5 * dt, device=dev), xb, mb,
                   obs_tokens=obs, statements=stm)
            z = z + dt * k2 * um
        out[sl] = tr.inverse(z).reshape(c, n_samples, d).cpu()
    return out


def metrics(samples, theta, mask):
    """samples (C, S, 9), theta (C, 9), mask (C, 9) -> per-case arrays over
    unknown columns: rank in [0,1], |median - truth|, 90 % width."""
    rank = (samples < theta[:, None, :]).float().mean(1)
    med = samples.median(1).values
    lo, hi = torch.quantile(samples, torch.tensor([0.05, 0.95]), dim=1)
    unk = 1 - mask
    return rank, (med - theta).abs(), hi - lo, unk


def summarize(name, rank, err, wid, unk, k_known):
    rows = []
    for a, b in ((0, 8),) + KNOWN_BINS[:3]:
        sel = ((k_known >= a) & (k_known <= b))[:, None] & unk.bool()
        if not sel.any():
            continue
        r = rank[sel]
        c50 = ((r > 0.25) & (r < 0.75)).float().mean().item()
        c90 = ((r > 0.05) & (r < 0.95)).float().mean().item()
        rmse = err[sel].pow(2).mean().sqrt().item()
        w90 = wid[sel].mean().item()
        rows.append(f"    {name:9s} known {a}-{b}: cov50 {c50:.3f}  cov90 {c90:.3f}  "
                    f"rmse {rmse:.3f}  width90 {w90:.3f}  (n={int(sel.sum())})")
    return rows


def per_param(names, rank, err, wid, unk):
    rows = []
    for j, n in enumerate(names):
        sel = unk[:, j].bool()
        r = rank[sel, j]
        rows.append(f"      {n:17s} cov90 {((r > 0.05) & (r < 0.95)).float().mean():.3f}  "
                    f"cov50 {((r > 0.25) & (r < 0.75)).float().mean():.3f}  "
                    f"rmse {err[sel, j].pow(2).mean().sqrt():.3f}  width90 {wid[sel, j].mean():.3f}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--test-bank", required=True)
    p.add_argument("--train-bank", default=None)
    p.add_argument("--val-bank", default=None)
    p.add_argument("--n-gap", type=int, default=2048, help="structures per side in the gap test")
    p.add_argument("--n-sld-gap", type=int, default=4)
    p.add_argument("--n-test", type=int, default=1024)
    p.add_argument("--sigma", type=float, default=0.0029, help="noise token for the test (common sigma)")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=25)
    p.add_argument("--case-bs", type=int, default=32)
    p.add_argument("--skip-gap", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    dev = args.device
    posts = [Posterior(c, device=dev) for c in args.ckpt]
    names = posts[0].phys_names
    space = posts[0].space
    print(f"{len(posts)} checkpoints, device {dev}")
    res = {}

    if not args.skip_gap:
        print("\n== gap: same val loss on training vs val structures (windows 0 and 1)")
        sides = {}
        for side, path in (("train", args.train_bank), ("val", args.val_bank)):
            b = StructureBank(path, device="cpu", use_halves=False, space=space)
            g = torch.Generator().manual_seed(11)
            idx = torch.randperm(len(b), generator=g)[:args.n_gap]
            b.S, b.structural = b.S[idx], b.structural[idx]
            th, lI = b.materialize_raw(args.n_sld_gap, seed=4243)
            sides[side] = (th.to(dev), lI.to(dev))
            print(f"  {side}: {len(idx)} structures x {args.n_sld_gap} SLD draws from {path}")
        for i, pp in enumerate(posts):
            for w in (0, 1):
                line = []
                for side in ("train", "val"):
                    tot, bins = val_loss(pp, *sides[side], w, seed=777)
                    res[f"gap_s{i}_w{w}_{side}"] = np.array([tot] + bins)
                    line.append(f"{side} {tot:.4f} ({' / '.join(f'{v:.3f}' for v in bins)})")
                print(f"  seed {i} window {pp.windows[w]}: " + " | ".join(line))
        print("  (in brackets: known 0 / 1-3 / 4-6 / 7-8)")

    if not args.skip_test:
        print(f"\n== test bank: sigma {args.sigma}, {args.n_samples} samples x {len(posts)} seeds, "
              f"midpoint {args.n_steps} steps, exact statements at the truth")
        tb = StructureBank(args.test_bank, device="cpu", use_halves=False, space=space)
        th, lI = tb.materialize_raw(1, seed=5001)
        th, lI = th[:args.n_test].to(dev), lI[:args.n_test].to(dev)
        C, d = th.shape
        g = torch.Generator().manual_seed(5002)
        # two cases per structure: nothing known, and k ~ U{1..6} random known params
        k = torch.cat([torch.zeros(C, dtype=torch.long), torch.randint(1, 7, (C,), generator=g)])
        mask = torch.zeros(2 * C, d)
        for r in range(C, 2 * C):
            mask[r, torch.randperm(d, generator=g)[:k[r]]] = 1
        mask = mask.to(dev)
        th2, lI2 = th.repeat(2, 1), lI.repeat(2, 1)
        for w in (0, 1):
            print(f"\n  window {posts[0].windows[w]}")
            pooled = []
            for i, pp in enumerate(posts):
                t0 = time.time()
                x = noisy_x(pp, lI2, w, args.sigma, seed=6000 + w)
                torch.manual_seed(7000 + 10 * w + i)
                s = sample_cases(pp, x, mask, th2, args.n_samples, args.n_steps, args.case_bs)
                pooled.append(s)
                rank, err, wid, unk = metrics(s, th2.cpu(), mask.cpu())
                for row in summarize(f"seed {i}", rank, err, wid, unk, k):
                    print(row)
                print(f"    ({time.time() - t0:.0f}s)")
                res[f"rank_s{i}_w{w}"] = rank.numpy()
            s = torch.cat(pooled, 1)
            rank, err, wid, unk = metrics(s, th2.cpu(), mask.cpu())
            for row in summarize("ensemble", rank, err, wid, unk, k):
                print(row)
            print("    ensemble per parameter (all cases):")
            for row in per_param(names, rank, err, wid, unk):
                print(row)
            res[f"rank_ens_w{w}"] = rank.numpy(); res[f"err_ens_w{w}"] = err.numpy()
            res[f"wid_ens_w{w}"] = wid.numpy()
        res["mask"] = mask.cpu().numpy(); res["theta"] = th2.cpu().numpy(); res["k"] = k.numpy()
    np.savez(args.out, names=np.array(names), ckpts=np.array(args.ckpt), **res)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
