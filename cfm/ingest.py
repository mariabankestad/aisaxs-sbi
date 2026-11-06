"""Turn a SAXS curve into the observation vector of the model.

The same code path serves training and inference, so both see the same layout.
Training calls `augment`, which cuts a q window from a simulated curve and adds
fresh noise. Inference calls `ingest_real`, which averages a measured curve onto
the simulation grid inside the window and estimates its noise level.

    x = [ mu, sigma_std, dct_1 .. dct_n, sigma_noise (, q_lo, q_hi) ]

mu             mean of log10 I in the window, centred and scaled, or 0 if unused
sigma_std      standard deviation of log10 I in the window
dct_k          DCT coefficients of the standardized curve
sigma_noise    noise level of the curve
q_lo, q_hi     window edges, present when the model handles several windows

Conventions:
  - grid: the simulator's linear pixel grid, with q = pixel * K / stretch;
  - noise: Gaussian in log10 I, with a q-dependence read from a noise template
    (standard deviation per measured point) and an amplitude sigma / sigma_ref.
    It is divided by sqrt(pts_per_bin), because a measured curve is averaged
    over about two points per simulation bin before reaching the model.
"""
from pathlib import Path

import numpy as np
import torch

K_PIXEL = 5.8703078e-4      # pixel index -> 1/A
STRETCH = 0.6945            # simulator length-unit convention
RAW_DQ = 2.206e-4           # spacing of the measured q grid, 1/A
SIGMA_REF = 0.002893        # noise amplitude of the template

QLO_SET = (0.008, 0.009, 0.010, 0.011, 0.0125)
QHI_SET = (0.31, 0.33, 0.35, 0.37, 0.39)
REAL_WINDOW = (0.010, 0.35)


def make_dct2_ortho(N, M):
    """Orthonormal DCT-II basis matrix (N, M)."""
    j = np.arange(N, dtype=np.float64)[:, None]
    k = np.arange(M, dtype=np.float64)[None, :]
    Phi = np.cos(np.pi / N * (j + 0.5) * k)
    alpha = np.sqrt(2.0 / N) * np.ones((1, M), dtype=np.float64)
    alpha[0, 0] = 1.0 / np.sqrt(N)
    return (Phi * alpha).astype(np.float32)


def _interp_loglog(xq, xp, fp):
    """Interpolate fp(xp) at xq, linear in log-log, slope-extrapolated at ends."""
    lq, lp, lf = np.log(xq), np.log(xp), np.log(fp)
    out = np.interp(lq, lp, lf)
    lo, hi = lq < lp[0], lq > lp[-1]
    if lo.any():
        s = (lf[1] - lf[0]) / (lp[1] - lp[0])
        out[lo] = lf[0] + s * (lq[lo] - lp[0])
    if hi.any():
        s = (lf[-1] - lf[-2]) / (lp[-1] - lp[-2])
        out[hi] = lf[-1] + s * (lq[hi] - lp[-1])
    return np.exp(out)


class _Window:
    __slots__ = ("q_lo", "q_hi", "sl", "basis", "noise_sd", "q_d", "resample")

    def __init__(self, q_lo, q_hi, sl, basis, noise_sd, q_d, resample=None):
        self.q_lo, self.q_hi, self.sl = q_lo, q_hi, sl
        self.basis, self.noise_sd, self.q_d = basis, noise_sd, q_d
        # (n_window, n_out) linear-interpolation matrix onto the log-q grid,
        # or None when the analysis grid is the native (linear) one
        self.resample = resample


class WindowedIngest:
    """Observation pipeline shared by training and inference."""

    def __init__(self, q_pixel, n_dct=64, stretch=STRETCH, K=K_PIXEL,
                 template_csv=None, template_arrays=None,
                 sigma_ref=SIGMA_REF,
                 qlo_set=QLO_SET, qhi_set=QHI_SET, real_window=REAL_WINDOW,
                 p_real=0.5, jitter=False, log_grid=False, n_log=None,
                 use_log_mean=False, mu_ref=0.0, mu_scale=1.0, device="cpu"):
        self.n_dct = n_dct
        self.stretch = float(stretch)
        self.K = float(K)
        self.sigma_ref = float(sigma_ref)
        self.p_real = float(p_real)
        self.device = device

        q_pixel = np.asarray(q_pixel, dtype=np.float64)
        self.q_pixel = q_pixel
        self.q_d = q_pixel * K / stretch            # sim grid in data units
        sim_dq = float(np.diff(self.q_d).mean())
        self.pts_per_bin = sim_dq / RAW_DQ          # ~2.0 raw points per sim bin

        if template_arrays is not None:
            self._tq, self._tsd = (np.asarray(template_arrays[0], dtype=np.float64),
                                   np.asarray(template_arrays[1], dtype=np.float64))
        else:
            t = np.genfromtxt(Path(template_csv), delimiter=",", names=True)
            self._tq, self._tsd = t["q"], t["sd"]

        self.jitter = bool(jitter)
        # Amplitude feature. Measured intensities are on a relative scale, so the
        # log-mean of a measured curve cannot be compared directly with a simulated
        # one. With use_log_mean=True the feature is (mu - mu_ref) / mu_scale, with
        # constants fitted on training curves, and an offset delta aligns measured
        # curves with the simulations at inference (ingest_real(delta=...)).
        self.use_log_mean = bool(use_log_mean)
        self.mu_ref, self.mu_scale = float(mu_ref), float(mu_scale)
        self.log_grid = bool(log_grid)
        self.n_log = n_log
        pairs = [tuple(real_window)]
        if self.jitter:
            for lo in qlo_set:
                for hi in qhi_set:
                    if (lo, hi) != tuple(real_window):
                        pairs.append((lo, hi))
        self.windows = [self._build(lo, hi) for lo, hi in pairs]
        # With a single fixed window the edges carry no information and are omitted.
        self.n_ctx_features = 3 if self.jitter else 1
        self.dim_x = n_dct + 2 + self.n_ctx_features

    def _build(self, q_lo, q_hi):
        i0 = int(np.searchsorted(self.q_d, q_lo, side="left"))
        i1 = int(np.searchsorted(self.q_d, q_hi, side="right"))
        assert i1 - i0 > self.n_dct + 1, f"window ({q_lo},{q_hi}) too short"
        q_slice = self.q_d[i0:i1]
        # Noise per simulation bin: template (per measured point) / sqrt(points per
        # bin). It is added on the linear grid, before any resampling in log q,
        # because that is the grid of the measurement.
        sd = _interp_loglog(q_slice, self._tq, self._tsd) / np.sqrt(self.pts_per_bin)
        noise_sd = torch.from_numpy(sd.astype(np.float32)).to(self.device)

        resample = None
        n_out = i1 - i0
        if self.log_grid:
            # Equal resolution per decade of q. On the linear grid most points lie
            # at high q, so the DCT would spend most of its coefficients there.
            n_out = self.n_log or (i1 - i0)
            q_out = np.logspace(np.log10(q_slice[0]), np.log10(q_slice[-1]), n_out)
            j = np.clip(np.searchsorted(q_slice, q_out) - 1, 0, len(q_slice) - 2)
            t = ((np.log(q_out) - np.log(q_slice[j]))
                 / (np.log(q_slice[j + 1]) - np.log(q_slice[j])))
            R = np.zeros((len(q_slice), n_out), dtype=np.float32)
            R[j, np.arange(n_out)] = 1.0 - t
            R[j + 1, np.arange(n_out)] = t
            resample = torch.from_numpy(R).to(self.device)

        basis = torch.from_numpy(
            make_dct2_ortho(n_out, self.n_dct + 1)).to(self.device)
        return _Window(q_lo, q_hi, slice(i0, i1), basis, noise_sd,
                       q_slice.astype(np.float32), resample)

    # ---- training side -------------------------------------------------------

    def draw_window_idx(self, rng):
        if len(self.windows) == 1 or rng.random() < self.p_real:
            return 0
        return int(rng.integers(1, len(self.windows)))

    def _ctx(self, sigma, w):
        if not self.jitter:
            return sigma.unsqueeze(1)
        return torch.stack([sigma,
                            torch.full_like(sigma, w.q_lo),
                            torch.full_like(sigma, w.q_hi)], dim=1)

    def augment(self, log_I, widx, sigma):
        """log_I (B, Nq) log10 curves on self.device; sigma (B,) noise tokens.

        Returns x (B, dim_x). Noise is injected fresh on every call.
        """
        w = self.windows[widx]
        y = log_I[:, w.sl]
        scale = (sigma / self.sigma_ref).unsqueeze(1)
        y = y + torch.randn_like(y) * w.noise_sd.unsqueeze(0) * scale
        if w.resample is not None:
            y = y @ w.resample          # linear grid -> log-q grid

        mu = y.mean(dim=1, keepdim=True)
        sig = y.std(dim=1, keepdim=True, correction=0).clamp_min(1e-8)
        coeffs = ((y - mu) / sig) @ w.basis          # (B, n_dct + 1); column 0 is the mean
        B = y.shape[0]
        amp = ((mu - self.mu_ref) / self.mu_scale if self.use_log_mean
               else torch.zeros(B, 1, device=y.device, dtype=y.dtype))
        return torch.cat([amp, sig, coeffs[:, 1:], self._ctx(sigma, w)], dim=1)

    # ---- inference side ------------------------------------------------------

    def measure_sigma(self, q_raw, I_raw):
        """Noise amplitude of a curve, from second differences of log10 I inside the
        window. Second differences remove the smooth shape of the curve. The noise
        feature is defined by this estimator, so the noise template must use it too.
        """
        q_lo, q_hi = self.windows[0].q_lo, self.windows[0].q_hi
        m = np.isfinite(I_raw) & (I_raw > 0) & (q_raw >= q_lo) & (q_raw <= q_hi)
        ly = np.log10(I_raw[m])
        d2 = ly[:-2] - 2 * ly[1:-1] + ly[2:]
        return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0))

    def ingest_real(self, q_raw, I_raw, widx=0, sigma=None, delta=0.0):
        """Measured curve -> (x (1, dim_x), sigma). The curve is averaged onto the
        simulation grid inside the window. Its noise level is estimated from the
        curve unless `sigma` is given.
        """
        w = self.windows[widx]
        if sigma is None:
            sigma = self.measure_sigma(q_raw, I_raw)

        q_c = self.q_d[w.sl]
        edges = np.empty(len(q_c) + 1)
        edges[1:-1] = 0.5 * (q_c[1:] + q_c[:-1])
        edges[0] = q_c[0] - (edges[1] - q_c[0])
        edges[-1] = q_c[-1] + (q_c[-1] - edges[-2])

        m = np.isfinite(I_raw) & (I_raw > 0)
        idx = np.digitize(q_raw[m], edges) - 1
        ly = np.log10(I_raw[m])
        y = np.full(len(q_c), np.nan)
        for i in range(len(q_c)):
            mi = idx == i
            if mi.any():
                y[i] = ly[mi].mean()
        if np.isnan(y).any():   # fill empty bins from their neighbours
            ok = ~np.isnan(y)
            y = np.interp(np.arange(len(y)), np.flatnonzero(ok), y[ok])

        yt = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(self.device)
        if w.resample is not None:
            yt = yt @ w.resample
        mu = yt.mean(dim=1, keepdim=True)
        sig = (yt - mu).pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        coeffs = ((yt - mu) / sig) @ w.basis
        s_t = torch.tensor([sigma], dtype=torch.float32, device=self.device)
        # delta is the assumed offset in log10 I between measurement and simulation.
        amp = (((mu - delta) - self.mu_ref) / self.mu_scale if self.use_log_mean
               else torch.zeros(1, 1, device=self.device))
        x = torch.cat([amp, sig, coeffs[:, 1:], self._ctx(s_t, w)], dim=1)
        return x, sigma

    # ---- checkpoint metadata --------------------------------------------------

    @classmethod
    def from_state(cls, state, device="cpu"):
        """Rebuild the training-time pipeline from the `ingest` entry of a checkpoint.
        Window indices keep the order they had during training.
        """
        windows = state["windows"]
        real, rest = tuple(windows[0]), windows[1:]
        obj = cls(np.asarray(state["q_pixel"]), n_dct=state["n_dct"],
                  stretch=state["stretch"], K=state["K"],
                  template_arrays=(state["template_q"], state["template_sd"]),
                  sigma_ref=state["sigma_ref"], qlo_set=(), qhi_set=(),
                  real_window=real, p_real=state["p_real"],
                  jitter=state.get("jitter", True),
                  log_grid=state.get("log_grid", False),
                  n_log=state.get("n_log"),
                  use_log_mean=state.get("use_log_mean", False),
                  mu_ref=state.get("mu_ref", 0.0),
                  mu_scale=state.get("mu_scale", 1.0), device=device)
        for lo, hi in rest:
            obj.windows.append(obj._build(lo, hi))
        return obj

    def state_dict(self):
        return {"n_dct": self.n_dct, "stretch": self.stretch, "K": self.K,
                "sigma_ref": self.sigma_ref, "p_real": self.p_real,
                "jitter": self.jitter, "n_ctx_features": self.n_ctx_features,
                "log_grid": self.log_grid, "n_log": self.n_log,
                "use_log_mean": self.use_log_mean,
                "mu_ref": self.mu_ref, "mu_scale": self.mu_scale,
                "pts_per_bin": self.pts_per_bin,
                "windows": [(w.q_lo, w.q_hi) for w in self.windows],
                "template_q": self._tq.tolist(),
                "template_sd": self._tsd.tolist(),
                "q_pixel": self.q_pixel.tolist()}
