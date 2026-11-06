"""Posterior inference with a trained SAXS model.

Three coordinate systems, kept explicit:
  bank units  the values the bank stores (tube_scaling, kappa_sc, volfrac_grf,
              no_rods_per_unit, min_space in reference pixels, size_cv, SLDs);
  physical    what a person states or reads: radius_nm = R_PER_AF / tube_scaling,
              min_space_nm = min_space * REF_PX_NM, all others = bank units;
  unit        [0, 1] in the coordinates the bank was sampled in (log for the
              log-sampled parameters, see params.py); the model's statements
              live here, and its flow runs in probit(unit) (ParamTransform).

Statements:
  exact={name: value}                      hard conditioning
  soft={name: (value, sd, kind)}           Gaussian statement, kind is
      "phys"  sd in physical units (nm for radius/min_space). Exact for linear
              parameters; for log-sampled parameters it is APPROXIMATED as a
              lognormal with relative sd = sd/value (delta method) and the
              approximation is printed; warning above 10 % relative sd.
      "rel"   (log-sampled parameters only) sd of the natural log, i.e. an
              exact lognormal statement, which is what the model was trained on.
      "unit"  sd in the model's [0, 1] coordinates (as trained).
  A statement wider than the trained ceiling (sigma_max in unit coordinates)
  is refused, not extrapolated.

Intensity: one common offset delta for all curves of a series (observe_series).

    from saxs.infer import Posterior
    post = Posterior("checkpoints/saxs_seed0.pt")
    xs, sig = post.observe_series(q_raw, curves, delta=-0.5, window=(0.05, 0.30))
    s = post.sample(xs[0], soft={"radius_nm": (1.5, 0.2, "phys")})   # physical units
"""

import numpy as np
import torch
from saxs.params import ParamSpace
from cfm.model import CFMTransformer
from cfm.transform import ParamTransform
from cfm.ingest import WindowedIngest

# geometry of the training simulations
BOX_NM = 743.5
R_PER_AF = 0.005 * BOX_NM          # radius_nm = R_PER_AF / tube_scaling
REF_PX_NM = BOX_NM / 1024          # min_space_nm = min_space * REF_PX_NM
PHYS = {"tube_scaling": ("radius_nm", "inv", R_PER_AF),
        "min_space": ("min_space_nm", "lin", REF_PX_NM)}
REL_WARN = 0.10


def load_model(path, device="cpu"):
    """Load a model and its checkpoint dictionary."""
    ck = torch.load(path, map_location=device, weights_only=True)
    model = CFMTransformer(n_params=ck["n_params"], n_dct=ck["n_dct"], d_model=ck["d_model"],
                           n_obs_tokens=ck["n_obs_tokens"], n_heads=ck["n_heads"],
                           n_layers=ck["n_layers"], time_scale=ck["time_scale"],
                           conditioning=ck["conditioning"], n_enc_layers=ck["n_enc_layers"],
                           soft_conditioning=ck["soft_conditioning"], stmt_embed=ck["stmt_embed"],
                           n_ctx_features=ck["n_ctx_features"]).to(device)
    model.sigma_max = ck["sigma_max"]
    model.load_state_dict(ck["model_state_dict"])
    model.param_transform = ParamTransform.from_state_dict(ck["param_transform"]).to(device)
    model.eval()
    return model, ck


class Posterior:
    def __init__(self, ckpt_path, device="cpu"):
        self.device = device
        self.model, self.ckpt = load_model(ckpt_path, device)
        self.space = ParamSpace.from_state(self.ckpt["param_space"])
        self.ingest = WindowedIngest.from_state(self.ckpt["ingest"], device=device)
        self.windows = [(round(w.q_lo, 6), round(w.q_hi, 6)) for w in self.ingest.windows]
        self.names = self.space.names
        self.phys_names = tuple(PHYS[n][0] if n in PHYS else n for n in self.names)
        self.sigma_max = float(self.ckpt["sigma_max"])

    # ---- coordinates ---------------------------------------------------------

    def index(self, name):
        if name in self.names:
            return self.names.index(name)
        if name in self.phys_names:
            return self.phys_names.index(name)
        raise KeyError(f"unknown parameter {name!r}; known: {self.phys_names}")

    def bank_to_phys(self, th):
        """(..., 9) bank units -> physical units."""
        out = np.array(th, dtype=np.float64, copy=True)
        for n in PHYS:
            k = self.names.index(n)
            out[..., k] = self._col_to_phys(k, out[..., k])
        return out

    def phys_to_bank(self, ph):
        out = np.array(ph, dtype=np.float64, copy=True)
        for n in PHYS:
            k = self.names.index(n)
            out[..., k] = self._col_to_bank(k, out[..., k])
        return out

    def phys_to_unit(self, ph):
        return self.space.to_unit(self.phys_to_bank(ph))

    def unit_to_phys(self, u):
        return self.bank_to_phys(self.space.from_unit(np.asarray(u, dtype=np.float64)))

    def _log_dphys_du(self, u):
        """log |d phys_k / d u_k| per column, (..., 9). phys is monotone in u
        per parameter, so the Jacobian is diagonal."""
        u = np.asarray(u, dtype=np.float64)
        span = self.space._b - self.space._a
        th = self.space.from_unit(u)
        out = np.log(np.abs(span)) + np.where(self.space.log, np.log(th), 0.0)   # d bank / d u
        for n, (_, kind, c) in PHYS.items():
            k = self.names.index(n)
            # inv: phys = c/th, |dphys/dth| = c/th^2;  lin: c
            out[..., k] += np.log(c) - 2 * np.log(th[..., k]) if kind == "inv" else np.log(c)
        return out

    # ---- statements ----------------------------------------------------------

    def statements(self, exact=None, soft=None, verbose=True):
        """-> (mask (9,), value_unit (9,), sigma_unit (9,)) torch tensors."""
        n = len(self.names)
        mask, val, sig = np.zeros(n), np.zeros(n), np.zeros(n)
        for name, v in (exact or {}).items():
            k = self.index(name)
            u = self._unit_col(k, self._as_phys(name, k, v))
            if not 0.0 <= u <= 1.0:
                raise ValueError(f"exact {name} = {v} lies outside the trained box")
            mask[k], val[k], sig[k] = 1, u, 0.0
        for name, spec in (soft or {}).items():
            if len(spec) != 3:
                raise ValueError(f"soft {name}: give (value, sd, kind) with kind in phys/rel/unit")
            v, sd, kind = spec
            k = self.index(name)
            if mask[k]:
                raise ValueError(f"{name} given twice")
            if sd <= 0:
                raise ValueError(f"soft {name}: sd must be > 0 (use exact= for a fixed value)")
            span = self.space._b[k] - self.space._a[k]
            if kind == "unit":
                u, s = float(v), float(sd)
            else:
                ph = self._as_phys(name, k, v)
                u = self._unit_col(k, ph)
                if kind == "rel":
                    if not self.space.log[k]:
                        raise ValueError(f"{name} is sampled linearly: give kind='phys'")
                    s = sd / span
                elif kind == "phys":
                    if self.space.log[k]:
                        rel = sd / ph
                        s = rel / span
                        if verbose:
                            print(f"note: {name} = {ph:g} +- {sd:g} (Gaussian, physical units) is "
                                  f"passed as a lognormal with relative sd {rel:.3f}, because the "
                                  f"model's statements are Gaussian in log({self.names[k]})")
                        if rel > REL_WARN:
                            print(f"WARNING: relative sd {rel:.2f} > {REL_WARN:.2f}: the lognormal "
                                  f"approximation of a Gaussian in nm is visibly skewed")
                    else:
                        s = sd / (self._phys_scale(k) * span)
                else:
                    raise ValueError(f"soft {name}: kind must be phys, rel or unit, not {kind!r}")
            if s > self.sigma_max + 1e-9:
                raise ValueError(f"soft {name}: sd = {s:.3f} in unit coordinates exceeds the trained "
                                 f"ceiling {self.sigma_max:.3f}; refused (would be extrapolation)")
            mask[k], val[k], sig[k] = 1, u, s
        f = lambda a: torch.tensor(a, dtype=torch.float32)
        return f(mask), f(val), f(sig)

    def _col_to_phys(self, k, v):
        n = self.names[k]
        if n not in PHYS:
            return v
        kind, c = PHYS[n][1:]
        return c / v if kind == "inv" else c * v

    def _col_to_bank(self, k, v):
        n = self.names[k]
        if n not in PHYS:
            return v
        kind, c = PHYS[n][1:]
        return c / v if kind == "inv" else v / c

    def _as_phys(self, name, k, v):
        """A value given under the bank name is converted to physical units."""
        v = float(v)
        if name in self.names and name != self.phys_names[k]:
            v = self._col_to_phys(k, v)
        if self.space.log[k] and v <= 0:
            raise ValueError(f"{name} must be > 0")
        return v

    def _unit_col(self, k, v_phys):
        return float(self.space.to_unit(np.array([self._col_to_bank(k, v_phys)]), [k])[0])

    def _phys_scale(self, k):
        """d phys / d bank for linear parameters."""
        n = self.names[k]
        return PHYS[n][2] if n in PHYS and PHYS[n][1] == "lin" else 1.0

    # ---- observations --------------------------------------------------------

    def window_index(self, window):
        w = (round(window[0], 6), round(window[1], 6))
        if w not in self.windows:
            raise ValueError(f"window {window} not trained; trained: {self.windows}")
        return self.windows.index(w)

    def observe(self, q_raw, I_raw, window=(0.05, 0.30), delta=0.0, sigma=None):
        """Measured curve (q in 1/A, linear I) -> (x (dim_x,), sigma).

        With sigma=None the noise level is estimated from the curve, on window 0
        whatever the analysis window. This estimator is calibrated for measured
        curves, so pass the true sigma for simulated curves.
        """
        x, s = self.ingest.ingest_real(np.asarray(q_raw, float), np.asarray(I_raw, float),
                                       widx=self.window_index(window), sigma=sigma, delta=delta)
        return x[0], s

    def observe_series(self, q_raw, curves, window=(0.05, 0.30), delta=0.0):
        """All curves of a series with ONE shared offset delta."""
        out = [self.observe(q_raw, c, window, delta) for c in curves]
        return [o[0] for o in out], [o[1] for o in out]

    # ---- posterior -----------------------------------------------------------

    def sample(self, x, exact=None, soft=None, n_samples=2000, n_steps=50, solver="midpoint",
               units="phys"):
        """Posterior samples (n, 9) in physical units (units='phys'), bank units
        ('bank') or [0, 1] ('unit'). Known columns echo the statement value."""
        mask, val, sig = self.statements(exact, soft)
        u = self.model.sample(x, mask, val, n_samples=n_samples, n_steps=n_steps,
                              sigma=sig, solver=solver).cpu().numpy().astype(np.float64)
        if units == "unit":
            return u
        bank = self.space.from_unit(u)
        return bank if units == "bank" else self.bank_to_phys(bank)

    def log_prob(self, theta_phys, x, exact=None, soft=None, n_steps=64):
        """log density of the UNKNOWN parameters at theta_phys (B, 9), in physical
        units, by the instantaneous change of variables with the exact
        divergence (n_unknown backward passes per field evaluation, RK4):
            log p_z(z1) = log N(z0) - int_0^1 div v dt,
            log p_phys  = log p_z - sum_unknown log|d phys / d z|.
        Known columns of theta_phys are ignored (set from the statements)."""
        mask, val, sig = self.statements(exact, soft, verbose=False)
        dev, m = self.device, self.model
        tr = m.param_transform
        th = np.asarray(theta_phys, dtype=np.float64)
        B = th.shape[0]
        u = self.phys_to_unit(th)
        u_t = torch.tensor(u, dtype=torch.float32, device=dev)
        maskb = mask.to(dev).unsqueeze(0).expand(B, -1)
        fixed = tr.forward(val.to(dev).clamp(0, 1).unsqueeze(0).expand(B, -1))
        z = torch.where(maskb.bool(), fixed, tr.forward(u_t))
        unk = torch.nonzero(mask == 0).flatten().tolist()
        stm = (torch.stack([val, sig], -1).to(dev) * mask.to(dev).unsqueeze(-1)).unsqueeze(0).expand(B, -1, -1)
        xb = x.to(dev).unsqueeze(0).expand(B, -1)
        with torch.no_grad():
            obs = m.encode_observation(x.to(dev).unsqueeze(0))
        um = 1 - maskb

        def field(zz, t):
            with torch.enable_grad():
                zz = zz.detach().requires_grad_(True)
                v = m.forward(zz, torch.full((B,), t, device=dev), xb, maskb,
                              obs_tokens=obs, statements=stm) * um
                div = sum(torch.autograd.grad(v[:, k].sum(), zz, retain_graph=True)[0][:, k]
                          for k in unk)
            return v.detach(), div.detach()

        logdet = torch.zeros(B, device=dev, dtype=torch.float64)
        dt = 1.0 / n_steps
        for i in range(n_steps):                     # backwards, t = 1 -> 0
            t0 = 1.0 - i * dt
            k1, d1 = field(z, t0)
            k2, d2 = field(z - 0.5 * dt * k1, t0 - 0.5 * dt)
            k3, d3 = field(z - 0.5 * dt * k2, t0 - 0.5 * dt)
            k4, d4 = field(z - dt * k3, t0 - dt)
            z = z - dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            logdet += (dt / 6 * (d1 + 2 * d2 + 2 * d3 + d4)).double()
        z0 = z.double()[:, unk]
        log_pz = -0.5 * (z0 ** 2).sum(1) - 0.5 * len(unk) * np.log(2 * np.pi) - logdet
        # z (standardized probit) -> unit: u = (Phi(z*std + mean) - eps) / (1 - 2 eps)
        zt = tr.forward(u_t).double()
        y = zt * tr.std.double() + tr.mean.double()
        log_du_dz = (-0.5 * y ** 2 - 0.5 * np.log(2 * np.pi) + torch.log(tr.std.double())
                     - np.log(1 - 2 * tr.eps))
        jac = log_du_dz.cpu().numpy() + self._log_dphys_du(u)
        return log_pz.cpu().numpy() - jac[:, unk].sum(1)
