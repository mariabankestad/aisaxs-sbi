"""Train the flow-matching transformer on the Gaussian task.

    python -m benchmarks.train --task correlated_gaussian --soft --save model.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from cfm.model import (CFMTransformer, sample_condition_masks,
                                           sample_soft_statements)
from cfm.transform import ParamTransform
from benchmarks.tasks import get_task


def train_benchmark(
    task_name,
    n_train=50000,
    n_val=5000,
    d_model=128,
    n_obs_tokens=4,
    n_heads=4,
    n_layers=6,
    n_enc_layers=2,
    batch_size=256,
    learning_rate=3e-4,
    max_epochs=500,
    patience=30,
    device="cuda",
    save_path=None,
    seed=42,
    conditioning="cross",
    size_label="",
    soft_conditioning=False,
    stmt_embed="joint",
    p_hard=0.5,
    sigma_max=0.2,
    sigma_dist="uniform",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    task = get_task(task_name)
    dim_x = task.dim_obs
    n_params = task.dim_params

    print(f"Task: {task_name} (dim_params={n_params}, dim_obs={dim_x})")
    print(f"Generating {n_train} training and {n_val} validation samples...")

    theta_train, x_train = task.generate_dataset(n_train)
    theta_val, x_val = task.generate_dataset(n_val)

    theta_train = theta_train.to(device)
    x_train = x_train.to(device)
    theta_val = theta_val.to(device)
    x_val = x_val.to(device)

    print(f"  theta: {theta_train.shape}, x: {x_train.shape}")

    # n_dct is used as dim_x - 2 so the obs_encoder input is dim_x
    # (the model adds 2 internally for the SAXS mu/sigma convention)
    n_dct_equiv = max(dim_x - 2, 0)

    model = CFMTransformer(
        n_params=n_params,
        n_dct=n_dct_equiv,
        d_model=d_model,
        n_obs_tokens=n_obs_tokens,
        n_heads=n_heads,
        n_enc_layers=n_enc_layers,
        n_layers=n_layers,
        conditioning=conditioning,
        soft_conditioning=soft_conditioning,
        stmt_embed=stmt_embed,
    ).to(device)
    if soft_conditioning:
        model.sigma_max = sigma_max

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Unconstrained standardized parameter space (eps-padded probit + per-dim
    # z-score), fit on the training params. Avoids the boundary pile-up the hard
    # clamp causes when a posterior sits near 0 or 1.
    # The flow-matching loss runs in this space; sample() maps back to [0,1]
    # internally, so evaluation is unchanged.
    transform = ParamTransform.fit(theta_train.cpu()).to(device)
    model.param_transform = transform
    theta_train_raw = theta_train.clone()   # raw [0,1], for soft statements
    theta_val_raw = theta_val.clone()
    theta_train = transform.forward(theta_train)
    theta_val = transform.forward(theta_val)
    print(f"ParamTransform: eps={transform.eps:.2e} (probit + z-score)")

    # Weight decay applies to the weight matrices only, not to biases and norm scales.
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 1e-4},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
    )
    warmup_epochs = 10
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs
    )
    # eta_min keeps the learning rate above zero at the end of the cosine schedule.
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs - warmup_epochs, eta_min=3e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Fixed conditional masks for validation (same distribution as training) so
    # checkpoints are selected for conditional, not just full-posterior, quality.
    val_condition_mask = sample_condition_masks(theta_val.shape[0], n_params=n_params, device=device)
    stmt_kwargs = dict(p_hard=p_hard, sigma_max=sigma_max, sigma_dist=sigma_dist)
    val_statements = (sample_soft_statements(theta_val_raw, val_condition_mask,
                                             **stmt_kwargs)
                      if soft_conditioning else None)
    if soft_conditioning:
        print(f"Soft conditioning: stmt_embed={stmt_embed}, p_hard={p_hard}, "
              f"sigma~{sigma_dist}(max={sigma_max})")

    print(f"\nTraining (max {max_epochs} epochs, patience {patience})...")
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            theta = theta_train[idx]
            x_obs = x_train[idx]
            B = theta.shape[0]

            condition_mask = sample_condition_masks(B, n_params=n_params, device=device)
            statements = (sample_soft_statements(theta_train_raw[idx], condition_mask,
                                                 **stmt_kwargs)
                          if soft_conditioning else None)

            t = torch.rand(B, device=device)
            noise = torch.randn_like(theta)

            theta_noised = (1 - t.unsqueeze(1)) * noise + t.unsqueeze(1) * theta
            theta_input = torch.where(condition_mask.bool(), theta, theta_noised)

            velocity_pred = model(theta_input, t, x_obs, condition_mask,
                                  statements=statements)

            velocity_true = theta - noise
            loss_mask = 1 - condition_mask
            n_unknown = loss_mask.sum()
            if n_unknown > 0:
                loss = ((velocity_pred - velocity_true) ** 2 * loss_mask).sum() / n_unknown
            else:
                continue

            if not torch.isfinite(loss):       # fail loudly, never save a NaN model
                raise RuntimeError(f"non-finite loss at epoch {epoch}, task {task_name}")

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_val_batches = 0
            for start in range(0, theta_val.shape[0], batch_size):
                theta = theta_val[start : start + batch_size]
                x_obs = x_val[start : start + batch_size]
                B = theta.shape[0]

                condition_mask = val_condition_mask[start : start + B]
                stmts = (val_statements[start : start + B]
                         if soft_conditioning else None)
                t = torch.rand(B, device=device)
                noise = torch.randn_like(theta)
                theta_noised = (1 - t.unsqueeze(1)) * noise + t.unsqueeze(1) * theta
                theta_input = torch.where(condition_mask.bool(), theta, theta_noised)

                velocity_pred = model(theta_input, t, x_obs, condition_mask,
                                      statements=stmts)
                velocity_true = theta - noise
                loss_mask = 1 - condition_mask
                loss = ((velocity_pred - velocity_true) ** 2 * loss_mask).sum() / loss_mask.sum().clamp_min(1)

                val_loss += loss.item()
                n_val_batches += 1

            avg_val_loss = val_loss / max(n_val_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:4d} | train {avg_train_loss:.4f} | val {avg_val_loss:.4f} | lr {lr:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            if save_path:
                save_obj = Path(save_path)
                save_obj.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "task_name": task_name,
                    "dim_x": dim_x,
                    "n_params": n_params,
                    "n_dct": n_dct_equiv,
                    "d_model": d_model,
                    "n_obs_tokens": n_obs_tokens,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "time_scale": model.time_scale,
                    "conditioning": model.conditioning,
                    "n_enc_layers": getattr(model, "n_enc_layers", 2),
                    "soft_conditioning": soft_conditioning,
                    "stmt_embed": stmt_embed,
                    "sigma_max": sigma_max if soft_conditioning else None,
                    "size_label": size_label,
                    "param_transform": transform.state_dict(),
                    "seed": seed,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch + 1,
                }, save_obj)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.0f}s (best val loss: {best_val_loss:.4f})")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train CFM Transformer on benchmark tasks")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (correlated_gaussian)")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-val", type=int, default=5000)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-obs-tokens", type=int, default=4,
                        help="K summary tokens for cross_two/cross_single (compression)")
    parser.add_argument("--n-enc-layers", type=int, default=2,
                        help="observation-encoder depth for cross_tab (amortized)")
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditioning", type=str, default="cross_two",
                        choices=["cross_two", "cross_single", "joint", "mlp",
                                 "cross_tab"])
    parser.add_argument("--size-label", type=str, default="",
                        help="label stored in the checkpoint")
    parser.add_argument("--soft", action="store_true",
                        help="soft conditioning: train on statements (v, sigma)")
    parser.add_argument("--stmt-embed", choices=["joint", "shared"], default="joint")
    parser.add_argument("--p-hard", type=float, default=0.5)
    parser.add_argument("--sigma-max", type=float, default=0.2)
    parser.add_argument("--sigma-dist", choices=["uniform", "loguniform"],
                        default="uniform")
    args = parser.parse_args()

    if args.save is None:
        args.save = f"checkpoints/bench_{args.task}.pt"

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_benchmark(
        task_name=args.task,
        n_train=args.n_train,
        n_val=args.n_val,
        d_model=args.d_model,
        n_obs_tokens=args.n_obs_tokens,
        n_heads=args.n_heads,
        n_enc_layers=args.n_enc_layers,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device,
        save_path=args.save,
        seed=args.seed,
        conditioning=args.conditioning,
        size_label=args.size_label,
        soft_conditioning=args.soft,
        stmt_embed=args.stmt_embed,
        p_hard=args.p_hard,
        sigma_max=args.sigma_max,
        sigma_dist=args.sigma_dist,
    )


if __name__ == "__main__":
    main()
