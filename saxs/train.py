"""Train the flow-matching transformer on a bank of simulated structures.

Each batch draws structures and fresh SLDs, adds noise, and trains on random
conditional queries: every parameter is unknown, known exactly, or known with an
uncertainty. Checkpoints are selected on the validation loss, weighted over the
q windows as in training.

Training needs a train bank and a validation bank (see saxs/bank.py) and a noise
template, a CSV file with the columns q and sd. The models in the paper were
trained with:

    python -m saxs.train --bank BANK_DIR --val-bank VAL_DIR --template NOISE.csv \
        --cosine --max-epochs 1400 --patience 5000 --no-ot-pairing \
        --seed 0 --save model.pt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from saxs.bank import StructureBank
from saxs.windows import TrainingWindows, P_MAIN, QLO_SET, Q_HI
from cfm.transform import ParamTransform
from cfm.model import (CFMTransformer, sample_condition_masks,
                                           sample_soft_statements)
from cfm.ema import EMA

VAL_SEED = 4242                                 # same validation set in every run
KNOWN_BINS = ((0, 0), (1, 3), (4, 6), (7, 8))   # validation loss by number of known parameters


def draw_sigma(n, lo, hi, device, generator=None):
    u = torch.rand(n, device=device, generator=generator)
    return torch.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))


def main():
    p = argparse.ArgumentParser(description="Train the flow-matching transformer on a structure bank")
    p.add_argument("--bank", required=True, help="train bank dir(s), comma separated")
    p.add_argument("--val-bank", required=True, help="val bank dir")
    p.add_argument("--n-val-sld", type=int, default=8, help="fixed SLD draws per val structure")
    p.add_argument("--no-halves", action="store_true",
                   help="train on the 6-realization mean even if the bank stores halves")
    p.add_argument("--save", required=True)
    # model
    p.add_argument("--n-dct", type=int, default=64)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-obs-tokens", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-enc-layers", type=int, default=2)
    p.add_argument("--conditioning", default="cross_two")
    p.add_argument("--dropout", type=float, default=0.1)
    # optimization
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=800)
    p.add_argument("--patience", type=int, default=120)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--plateau-patience", type=int, default=25)
    p.add_argument("--cosine", action="store_true", help="cosine schedule instead of plateau decay")
    p.add_argument("--no-ot-pairing", action="store_true")
    p.add_argument("--no-stratified-t", action="store_true")
    p.add_argument("--no-paired", action="store_true")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--epoch-samples", type=int, default=262144)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    # uncertain statements
    p.add_argument("--p-hard", type=float, default=0.7)
    p.add_argument("--sigma-max", type=float, default=0.15)
    p.add_argument("--sigma-dist", choices=["uniform", "loguniform"], default="loguniform")
    # observation pipeline
    p.add_argument("--template", required=True,
                   help="noise template, a CSV file with the columns q and sd")
    p.add_argument("--stretch", type=float, default=0.6945,
                   help="q-axis convention of the simulations")
    p.add_argument("--noise-sigma-min", type=float, default=0.0007)
    p.add_argument("--noise-sigma-max", type=float, default=0.010)
    p.add_argument("--p-main", type=float, nargs=2, default=list(P_MAIN))
    p.add_argument("--qlo-set", type=float, nargs="+", default=list(QLO_SET))
    p.add_argument("--no-log-grid", action="store_true")
    p.add_argument("--no-log-mean", action="store_true")
    args = p.parse_args()
    if Path(args.save).exists():
        raise SystemExit(f"{args.save} exists; choose a new --save path")

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"Using device: {device}")

    bank = StructureBank(args.bank, device=device, use_halves=not args.no_halves)
    space = bank.space
    val_bank = StructureBank(args.val_bank, device=device, use_halves=False, space=space)
    print(f"bank: {len(bank):,} structures, val: {len(val_bank):,} structures, "
          f"realization halves: {'ON' if bank.halves is not None else 'off'}")
    for k, n in enumerate(space.names):
        print(f"  {n:17s} {space.lo[k]:g} - {space.hi[k]:g}{'  (log)' if space.log[k] else ''}")
    n_par = len(space)

    ingest = TrainingWindows(bank.q.cpu().numpy(), p_main=tuple(args.p_main),
                       qlo_set=tuple(args.qlo_set), q_hi=Q_HI, n_dct=args.n_dct,
                       stretch=args.stretch, template_csv=args.template,
                       log_grid=not args.no_log_grid, use_log_mean=not args.no_log_mean,
                       device=device)
    n_win, n_main = len(ingest.windows), ingest.n_main
    if ingest.use_log_mean:
        with torch.no_grad():
            _, lI = bank.sample_batch_raw(8192)
            w0 = ingest.windows[0]
            y = lI[:, w0.sl]
            if w0.resample is not None:
                y = y @ w0.resample
            mu = y.mean(1)
            ingest.mu_ref, ingest.mu_scale = float(mu.mean()), float(mu.std().clamp_min(1e-6))
        print(f"amplitude feature live, centred {ingest.mu_ref:.4f} / scaled {ingest.mu_scale:.4f}")
    print("windows: " + ", ".join(f"[{w.q_lo:g}, {w.q_hi:g}]" for w in ingest.windows)
          + f"  (main p = {args.p_main})")

    def augment(log_I, widx, gen=None):
        sigma = draw_sigma(log_I.shape[0], args.noise_sigma_min, args.noise_sigma_max, device, gen)
        return ingest.augment(log_I, widx, sigma)

    cycle = [0]

    def window_plan(n):
        """Main windows get their share of rows; the rest is split over up to 4
        jitter windows, cycling a counter so all are visited uniformly."""
        segs, used = [], 0
        for i, pm in enumerate(args.p_main):
            m = int(round(pm * n))
            if m:
                segs.append((i, m)); used += m
        left, k = n - used, min(n_win - n_main, 4)
        base, extra = divmod(left, k)
        for s in range(k):
            m = base + (1 if s < extra else 0)
            if m:
                segs.append((n_main + cycle[0] % (n_win - n_main), m)); cycle[0] += 1
        return segs

    def mixed_augment(log_I):
        xs, off = [], 0
        for w, m in window_plan(log_I.shape[0]):
            xs.append(augment(log_I[off:off + m], w)); off += m
        return torch.cat(xs)

    model = CFMTransformer(n_params=n_par, n_dct=args.n_dct, d_model=args.d_model,
                           n_obs_tokens=args.n_obs_tokens, n_heads=args.n_heads,
                           n_layers=args.n_layers, n_enc_layers=args.n_enc_layers,
                           conditioning=args.conditioning, dropout=args.dropout,
                           soft_conditioning=True, stmt_embed="shared",
                           n_ctx_features=ingest.n_ctx_features).to(device)
    model.sigma_max = args.sigma_max
    n_train = args.epoch_samples
    theta_stats, _ = bank.sample_batch_raw(min(n_train, 131072))
    transform = ParamTransform.fit(theta_stats.cpu()).to(device)
    model.param_transform = transform
    print(f"model parameters: {sum(q.numel() for q in model.parameters()):,}; "
          f"epoch = {n_train:,} fresh samples")

    # reference statistics of the observation features, stored in the checkpoint
    rng_s = np.random.default_rng(args.seed + 3)
    xs = []
    for _ in range(0, min(n_train, 131072), 4096):
        xs.append(augment(bank.sample_batch_raw(4096)[1], ingest.draw_window_idx(rng_s)))
    xd = torch.cat(xs); x_ref_mean, x_ref_std = xd.mean(0), xd.std(0).clamp_min(1e-6)
    del xs, xd

    # fixed validation: one target per window, identical masks/statements/t/noise
    theta_val, logI_val = val_bank.materialize_raw(args.n_val_sld, seed=VAL_SEED)
    theta_val_model = transform.forward(theta_val)
    x_val = []
    for wi in range(n_win):
        gen = torch.Generator(device=device).manual_seed(VAL_SEED + 7)
        chunks = []
        for s in range(0, len(theta_val), args.batch_size):
            with torch.random.fork_rng(devices=[device] if "cuda" in device else []):
                torch.manual_seed(VAL_SEED + 7 + s)
                chunks.append(augment(logI_val[s:s + args.batch_size], wi, gen))
        x_val.append(torch.cat(chunks))
    stmt_kw = dict(p_hard=args.p_hard, sigma_max=args.sigma_max, sigma_dist=args.sigma_dist)
    with torch.random.fork_rng(devices=[device] if "cuda" in device else []):
        torch.manual_seed(VAL_SEED)
        val_mask = sample_condition_masks(len(theta_val), n_params=n_par, device=device)
        val_stmts = sample_soft_statements(theta_val, val_mask, **stmt_kw)
        val_t = torch.rand(len(theta_val), device=device)
        val_noise = torch.randn_like(theta_val_model)
    w_sel = list(args.p_main) + [(1 - sum(args.p_main)) / (n_win - n_main)] * (n_win - n_main)
    print(f"val: {len(theta_val):,} curves x {n_win} windows; selection = window-weighted val loss")

    decay = [q for q in model.parameters() if q.ndim >= 2]
    no_decay = [q for q in model.parameters() if q.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-2, end_factor=1.0,
                                             total_iters=args.warmup_epochs)
    if args.cosine:
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs - args.warmup_epochs,
                                                         eta_min=args.lr / 10)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[args.warmup_epochs])
        plateau = None
    else:
        sched = None
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=args.plateau_factor,
                                                             patience=args.plateau_patience,
                                                             min_lr=args.lr / 100)
    ema = None if args.no_ema else EMA(model, decay=0.999)
    if not args.no_ot_pairing:
        from scipy.optimize import linear_sum_assignment

    n_known_val = val_mask.sum(1)

    def val_pass(xv):
        """Validation loss of one window, in total and per sample."""
        tot, nb, per = 0.0, 0, []
        for s in range(0, len(theta_val), args.batch_size):
            th = theta_val_model[s:s + args.batch_size]; B = th.shape[0]
            m = val_mask[s:s + B]; t = val_t[s:s + B]; nz = val_noise[s:s + B]
            th_in = torch.where(m.bool(), th, (1 - t[:, None]) * nz + t[:, None] * th)
            v = model(th_in, t, xv[s:s + B], m, statements=val_stmts[s:s + B])
            lm = 1 - m
            se = (v - (th - nz)) ** 2 * lm
            tot += (se.sum() / lm.sum().clamp_min(1)).item(); nb += 1
            per.append(se.sum(1) / lm.sum(1).clamp_min(1))
        return tot / max(nb, 1), torch.cat(per)

    def by_known(per):
        return [float(per[(n_known_val >= a) & (n_known_val <= b)].mean()) for a, b in KNOWN_BINS]

    best, stall, t0 = float("inf"), 0, time.time()
    for epoch in range(args.max_epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for _ in range(0, n_train, args.batch_size):
            theta_raw, log_I = bank.sample_batch_raw(args.batch_size)
            x_obs = mixed_augment(log_I)
            theta = transform.forward(theta_raw)
            B = theta.shape[0]
            mask = sample_condition_masks(B, n_params=n_par, device=device)
            if not args.no_paired:
                for _r in range(8):
                    bad = torch.where((1 - mask).sum(1) < 2)[0]
                    if not bad.numel():
                        break
                    mask[bad] = sample_condition_masks(bad.numel(), n_params=n_par, device=device)
            stmts = sample_soft_statements(theta_raw, mask, **stmt_kw)
            t = (((torch.randperm(B, device=device).float() + torch.rand(B, device=device)) / B)
                 if not args.no_stratified_t else torch.rand(B, device=device))
            noise = torch.randn_like(theta)
            if not args.no_ot_pairing:
                with torch.no_grad():
                    r, c = linear_sum_assignment(torch.cdist(noise, theta).cpu().numpy())
                    noise = noise[torch.from_numpy(r[np.argsort(c)]).to(device)]
            th_noised = (1 - t[:, None]) * noise + t[:, None] * theta
            v_true = theta - noise

            def masked_loss(mk, st):
                v = model(torch.where(mk.bool(), theta, th_noised), t, x_obs, mk, statements=st)
                lm = 1 - mk
                return ((v - v_true) ** 2 * lm).sum() / lm.sum() if lm.sum() > 0 else None

            loss = masked_loss(mask, stmts)
            if loss is None:
                continue
            if not args.no_paired:
                rows = torch.arange(B, device=device)
                j = torch.multinomial(1 - mask, 1).squeeze(1)
                mask_f = mask.clone(); mask_f[rows, j] = 1.0
                st_f = stmts.clone(); st_f[rows, j, 0] = theta_raw[rows, j]; st_f[rows, j, 1] = 0.0
                lf = masked_loss(mask_f, st_f)
                if lf is not None:
                    loss = loss + lf
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema:
                ema.update(model)
            ep_loss += loss.item(); nb += 1
        if sched is not None:
            sched.step()
        elif epoch < args.warmup_epochs:
            warm.step()

        backup = ema.swap_in(model) if ema else None
        model.eval()
        with torch.no_grad():
            res = [val_pass(xv) for xv in x_val]
        vw = [r[0] for r in res]
        vk = {w: by_known(res[w][1]) for w in range(n_main)}
        score = float(np.dot(w_sel, vw))
        if plateau is not None and epoch >= args.warmup_epochs:
            plateau.step(score)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:4d} | train {ep_loss/max(nb,1):.4f} | val {score:.4f} "
                  f"| meas-window {vw[0]:.4f} | sim-window {vw[1]:.4f} "
                  f"| lr {opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.0f}s", flush=True)
            for w in range(n_main):
                print(f"      by known params [{ingest.windows[w].q_lo:g}, {ingest.windows[w].q_hi:g}]: "
                      + ", ".join(f"{a}-{b}: {l:.4f}" for (a, b), l in zip(KNOWN_BINS, vk[w])), flush=True)
        if score < best:
            best, stall = score, 0
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_params": n_par, "n_dct": args.n_dct, "d_model": args.d_model,
                "n_obs_tokens": args.n_obs_tokens, "n_heads": args.n_heads,
                "n_layers": args.n_layers, "n_enc_layers": args.n_enc_layers,
                "time_scale": model.time_scale, "conditioning": model.conditioning,
                "n_ctx_features": ingest.n_ctx_features,
                "param_transform": transform.state_dict(), "param_space": space.state_dict(),
                "x_stats": None, "x_ref_stats": [x_ref_mean.cpu(), x_ref_std.cpu()],
                "soft_conditioning": True, "stmt_embed": "shared", "sigma_max": args.sigma_max,
                "p_hard": args.p_hard, "sigma_dist": args.sigma_dist, "use_ema": bool(ema),
                "best_val_loss": best, "val_by_window": vw, "epoch": epoch + 1,
                "val_by_known": {"bins": KNOWN_BINS, "main_windows": vk},
                "ingest": ingest.state_dict(), "p_main": list(args.p_main),
                "noise_sigma_range": [args.noise_sigma_min, args.noise_sigma_max],
                "paired": not args.no_paired, "halves": bank.halves is not None,
                "structure_bank": args.bank, "n_structures": len(bank), "args": vars(args),
            }, args.save)
        else:
            stall += 1
        if ema:
            ema.swap_out(model, backup)
        if stall >= args.patience:
            print(f"  early stop at epoch {epoch+1}")
            break
    print(f"training done in {time.time()-t0:.0f}s, best window-weighted val {best:.4f}")


if __name__ == "__main__":
    main()
