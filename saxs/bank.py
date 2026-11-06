"""Bank of simulated structures.

The scattering of a structure is a quadratic form in its density contrasts, so
three stored cross spectra give the curve for any SLD triple:

    I(q) = d1^2 S11 + d2^2 S22 + 2 d1 d2 S12,   d_i = SLD_grf_i - SLD_tubes

Training can therefore draw fresh SLDs for every batch at almost no cost.

A bank is a folder of chunk_*.npz files with the keys S (n, 3, n_q), structural
(n, 6), info (n, n_info), q, param_names, bounds_lo, bounds_hi, log_scale and
info_keys. All chunks must share the same parameter box. A chunk may also store
S_halves, the spectra of two halves of the realizations, which training can then
draw from.
"""
from pathlib import Path

import numpy as np
import torch

from .params import ParamSpace


def sld_weights_torch(sld):
    """SLD triples (B, 3) [tubes, grf1, grf2] -> contrast weights (B, 3) for
    [S11, S22, S12]."""
    d1 = sld[:, 1] - sld[:, 0]
    d2 = sld[:, 2] - sld[:, 0]
    return torch.stack([d1 * d1, d2 * d2, 2.0 * d1 * d2], dim=1)


def _chunks(bank_dirs):
    files = []
    for d in str(bank_dirs).split(","):
        f = sorted(Path(d.strip()).glob("chunk_*.npz"))
        if not f:
            raise FileNotFoundError(f"no chunk_*.npz in {d}")
        files += f
    return files


class StructureBank:
    def __init__(self, bank_dirs, device="cuda", use_halves=True, space=None):
        S_parts, H_parts, st_parts, info_parts = [], [], [], []
        meta, q, info_keys = None, None, None
        for f in _chunks(bank_dirs):
            z = np.load(f)
            m = (tuple(z["param_names"]), tuple(np.round(z["bounds_lo"], 10)),
                 tuple(np.round(z["bounds_hi"], 10)), tuple(z["log_scale"]))
            if meta is None:
                meta, q, info_keys = m, z["q"], list(z["info_keys"])
            elif m != meta:
                raise ValueError(f"{f}: parameter box differs from the first chunk: "
                                 f"banks with different boxes must not be mixed")
            S_parts.append(z["S"]); st_parts.append(z["structural"]); info_parts.append(z["info"])
            H_parts.append(z["S_halves"] if "S_halves" in z.files else None)
        self.space = space or ParamSpace.from_bank_meta(*meta)
        self.device = device
        self.q = torch.tensor(q, dtype=torch.float32, device=device)
        self.S = torch.tensor(np.concatenate(S_parts), dtype=torch.float32, device=device)
        self.halves = None
        if use_halves and all(h is not None for h in H_parts):
            self.halves = torch.tensor(np.concatenate(H_parts), dtype=torch.float32, device=device)
        structural = np.concatenate(st_parts)
        self.structural = torch.tensor(self.space.to_unit(structural, slice(0, self.space.n_struct)),
                                       dtype=torch.float32, device=device)
        self.info = np.concatenate(info_parts)
        self.info_keys = info_keys
        n = self.space.n_struct
        self.sld_lo = torch.tensor(self.space.lo[n:], dtype=torch.float32, device=device)
        self.sld_hi = torch.tensor(self.space.hi[n:], dtype=torch.float32, device=device)

    def __len__(self):
        return self.S.shape[0]

    def _curves(self, idx, sld, generator=None):
        if self.halves is not None:
            h = torch.randint(0, 2, (idx.numel(),), device=self.device, generator=generator)
            S = self.halves[idx, h]
        else:
            S = self.S[idx]
        I = torch.einsum("bk,bkq->bq", sld_weights_torch(sld), S)
        return torch.log10(I.clamp_min(1e-12))

    def _theta(self, idx, sld):
        return torch.cat([self.structural[idx], (sld - self.sld_lo) / (self.sld_hi - self.sld_lo)], 1)

    def sample_batch_raw(self, batch_size, generator=None):
        """Random structures x fresh uniform SLDs -> (theta in [0,1] (B,9), log10 I)."""
        idx = torch.randint(0, len(self), (batch_size,), device=self.device, generator=generator)
        u = torch.rand(batch_size, 3, device=self.device, generator=generator)
        sld = self.sld_lo + u * (self.sld_hi - self.sld_lo)
        return self._theta(idx, sld), self._curves(idx, sld, generator)

    def materialize_raw(self, n_sld_per_structure, seed=0):
        """Fixed validation set: every structure with n fixed SLD draws, using the mean
        spectra over all realizations.
        """
        g = torch.Generator(device=self.device).manual_seed(seed)
        idx = torch.arange(len(self), device=self.device).repeat_interleave(n_sld_per_structure)
        u = torch.rand(idx.numel(), 3, device=self.device, generator=g)
        sld = self.sld_lo + u * (self.sld_hi - self.sld_lo)
        I = torch.einsum("bk,bkq->bq", sld_weights_torch(sld), self.S[idx])
        return self._theta(idx, sld), torch.log10(I.clamp_min(1e-12))
