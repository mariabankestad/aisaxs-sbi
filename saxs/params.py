"""Parameter space: 6 structural parameters from the simulation bank and 3 scattering
length densities (SLDs) drawn during training.

Every parameter is mapped to [0, 1] in the coordinates it was sampled in: linearly
for uniform parameters and on a log scale for log-sampled ones. The prior is then
uniform on [0, 1], as ParamTransform assumes. An uncertainty in [0, 1] units is
therefore a relative uncertainty for the log-sampled parameters.
"""
import numpy as np
import torch

SLD_NAMES = ("SLD_tubes", "SLD_grf_1", "SLD_grf_2")
SLD_BOUNDS = ((0.7, 1.0), (0.5, 0.99), (0.5, 0.99))


class ParamSpace:
    def __init__(self, names, lo, hi, log):
        self.names = tuple(str(n) for n in names)
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        self.log = np.asarray(log, dtype=bool)
        self._a = np.where(self.log, np.log(np.where(self.log, self.lo, 1.0)), self.lo)
        self._b = np.where(self.log, np.log(np.where(self.log, self.hi, 1.0)), self.hi)

    @classmethod
    def from_bank_meta(cls, names, lo, hi, log):
        """Structural part from a bank file + the SLD block."""
        return cls(list(names) + list(SLD_NAMES),
                   list(lo) + [b[0] for b in SLD_BOUNDS],
                   list(hi) + [b[1] for b in SLD_BOUNDS],
                   list(log) + [False] * len(SLD_NAMES))

    def __len__(self):
        return len(self.names)

    @property
    def n_struct(self):
        return len(self.names) - len(SLD_NAMES)

    def to_unit(self, raw, cols=None):
        """Raw values (..., k) -> [0, 1]. cols selects a subset of columns."""
        sl = slice(None) if cols is None else cols
        a, b, lg = self._a[sl], self._b[sl], self.log[sl]
        if torch.is_tensor(raw):
            a, b = torch.as_tensor(a, dtype=raw.dtype, device=raw.device), \
                   torch.as_tensor(b, dtype=raw.dtype, device=raw.device)
            lg = torch.as_tensor(lg, device=raw.device)
            v = torch.where(lg, torch.log(raw.clamp_min(1e-30)), raw)
        else:
            v = np.where(lg, np.log(np.maximum(raw, 1e-30)), raw)
        return (v - a) / (b - a)

    def from_unit(self, u, cols=None):
        sl = slice(None) if cols is None else cols
        a, b, lg = self._a[sl], self._b[sl], self.log[sl]
        if torch.is_tensor(u):
            a, b = torch.as_tensor(a, dtype=u.dtype, device=u.device), \
                   torch.as_tensor(b, dtype=u.dtype, device=u.device)
            v = a + u * (b - a)
            return torch.where(torch.as_tensor(lg, device=u.device), torch.exp(v), v)
        v = a + u * (b - a)
        return np.where(lg, np.exp(v), v)

    def state_dict(self):
        return {"names": list(self.names), "lo": self.lo.tolist(),
                "hi": self.hi.tolist(), "log": self.log.tolist()}

    @classmethod
    def from_state(cls, s):
        return cls(s["names"], s["lo"], s["hi"], s["log"])
