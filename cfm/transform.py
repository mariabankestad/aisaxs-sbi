"""Map parameters between [0, 1] and the unconstrained space of the flow."""
import numpy as np
import torch

_SQRT2 = float(np.sqrt(2.0))


class ParamTransform:
    """Map bounded parameters in [0, 1] to an unconstrained, standardized space.

    Flow matching transports a Gaussian base distribution to the parameters. If the
    flow ran directly in [0, 1], samples would have to be clamped, which piles
    posterior mass onto the bounds. Instead, parameters are padded by eps, mapped
    through the inverse Gaussian CDF (probit) and standardized per dimension:

        theta in [0, 1]  ->  pad by eps  ->  probit  ->  z-score

    A uniform prior on [0, 1] then maps almost exactly to the standard normal base.
    A point 1e-4 from a bound lands near z = -3.7, compared with about -9.2 for a
    logit, so posteriors close to a bound stay well represented.

    `eps` is chosen below the closest approach of the training data to a bound. The
    transform is invertible, so samples land inside (0, 1) without clamping.
    """

    def __init__(self, eps, mean, std, method="normal"):
        self.eps = float(eps)
        self.mean = mean
        self.std = std
        self.method = method

    @staticmethod
    def _to_unbounded(pad, method):
        if method == "normal":
            return _SQRT2 * torch.erfinv(2.0 * pad - 1.0)   # probit
        elif method == "logit":
            return torch.logit(pad)
        raise ValueError(f"unknown method: {method}")

    @staticmethod
    def _to_unit(y, method):
        if method == "normal":
            return 0.5 * (1.0 + torch.erf(y / _SQRT2))      # Gaussian CDF
        elif method == "logit":
            return torch.sigmoid(y)
        raise ValueError(f"unknown method: {method}")

    @classmethod
    def fit(cls, theta01, method="normal", eps=None, eps_floor=1e-7, eps_cap=1e-4):
        # Tasks with an unbounded prior can produce values just outside [0, 1],
        # which the probit would send to infinity.
        theta01 = torch.as_tensor(theta01, dtype=torch.float32).clamp(0.0, 1.0)
        if eps is None:
            closest = torch.minimum(theta01, 1.0 - theta01).min().item()
            eps = min(max(0.1 * closest, eps_floor), eps_cap)
        pad = theta01 * (1.0 - 2.0 * eps) + eps
        y = cls._to_unbounded(pad, method)
        mean = y.mean(0)
        std = y.std(0).clamp_min(1e-6)
        return cls(eps, mean, std, method=method)

    def to(self, device):
        return ParamTransform(self.eps, self.mean.to(device), self.std.to(device),
                              method=self.method)

    def forward(self, theta01):
        """[0,1] -> unconstrained standardized space."""
        theta01 = theta01.clamp(0.0, 1.0)
        pad = theta01 * (1.0 - 2.0 * self.eps) + self.eps
        return (self._to_unbounded(pad, self.method) - self.mean) / self.std

    def inverse(self, z):
        """unconstrained standardized space -> [0,1]."""
        pad = self._to_unit(z * self.std + self.mean, self.method)
        return ((pad - self.eps) / (1.0 - 2.0 * self.eps)).clamp(0.0, 1.0)

    def state_dict(self):
        return {"eps": self.eps, "mean": self.mean.detach().cpu(),
                "std": self.std.detach().cpu(), "method": self.method}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["eps"], torch.as_tensor(state["mean"]),
                   torch.as_tensor(state["std"]), method=state.get("method", "normal"))
