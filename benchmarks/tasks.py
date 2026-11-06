"""Benchmark task with exact conditional posteriors.

Parameters are returned in [0, 1], the range the model works in.
"""

import torch
import numpy as np


class BenchmarkTask:
    """Base class for benchmark tasks."""

    name: str
    dim_params: int
    dim_obs: int
    # True when the analytic conditionals are exact, not an approximation.
    exact_conditional = False

    def simulate(self, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_prior(self, n: int) -> torch.Tensor:
        raise NotImplementedError

    def analytic_conditional_mean_std(self, obs, condition_mask, theta_fixed):
        """Return analytic conditional posterior if available. None otherwise."""
        return None

    def generate_dataset(self, n: int):
        theta = self.sample_prior(n)
        x = self.simulate(theta)
        return theta, x


class CorrelatedGaussian(BenchmarkTask):
    """Correlated linear-Gaussian task with exact analytic conditionals.

    A latent z ~ N(0, Sigma_z) has correlated components. The parameters live in
    [0, 1] through theta = 0.5 + s * z, and the observation is x = A z + noise. The
    joint distribution of (theta, x) is Gaussian, so the posterior and every
    conditional p(theta_unknown | x, theta_known) are known in closed form. `s` is
    small, so theta stays inside [0, 1] with high probability.
    """

    name = "correlated_gaussian"
    exact_conditional = True

    def __init__(self, dim_params=5, dim_obs=15, noise_std=0.05, s=0.1,
                 rank=None, corr_rank=None, seed=0):
        self.dim_params = dim_params
        self.dim_obs = dim_obs
        self.noise_std = noise_std
        self.s = s
        # With rank < dim_params, some parameter combinations are not determined by
        # x. The posterior stays wide in those directions, and conditioning on a
        # subset of the parameters resolves the rest.
        self.rank = rank or max(dim_params // 2, 1)
        rng = np.random.RandomState(seed)
        # Correlated latent prior covariance: low-rank + diagonal, unit-scaled.
        k = corr_rank or max(dim_params // 2, 1)
        B = rng.randn(dim_params, k)
        Sigma_z = B @ B.T + np.eye(dim_params)
        d = np.sqrt(np.diag(Sigma_z))
        Sigma_z = Sigma_z / np.outer(d, d)          # correlation matrix (unit diag)
        self.Sigma_z = torch.tensor(Sigma_z, dtype=torch.float32)
        self.L_z = torch.linalg.cholesky(self.Sigma_z)
        A = (rng.randn(dim_obs, self.rank) @ rng.randn(self.rank, dim_params)) \
            / np.sqrt(self.rank * dim_params)
        self.A = torch.tensor(A.astype(np.float32))
        # Posterior precision pieces (prior mean of z is 0).
        self._Sig_z_inv = torch.linalg.inv(self.Sigma_z)
        self._AtA = self.A.T @ self.A / (noise_std ** 2)

    def sample_prior(self, n):
        z = torch.randn(n, self.dim_params) @ self.L_z.T
        return 0.5 + self.s * z

    def simulate(self, theta):
        z = (theta - 0.5) / self.s
        return z @ self.A.T + self.noise_std * torch.randn(theta.shape[0], self.dim_obs)

    def _theta_posterior(self, x):
        """Gaussian posterior over theta given a single observation x."""
        post_prec = self._Sig_z_inv + self._AtA
        post_cov_z = torch.linalg.inv(post_prec)
        mu_z = post_cov_z @ (self.A.T @ x.squeeze() / (self.noise_std ** 2))
        mu_theta = 0.5 + self.s * mu_z
        cov_theta = (self.s ** 2) * post_cov_z
        return mu_theta, cov_theta

    def _conditional_gaussian(self, x, condition_mask, theta_fixed):
        """Mean/cov of p(theta_unknown | x, theta_known) via Gaussian conditioning."""
        mu, cov = self._theta_posterior(x)
        u = torch.where(condition_mask == 0)[0]
        k = torch.where(condition_mask == 1)[0]
        if k.numel() == 0:
            return u, mu[u], cov[u][:, u]
        cov_uk = cov[u][:, k]
        cov_kk = cov[k][:, k]
        cov_kk_inv = torch.linalg.inv(cov_kk)
        mu_cond = mu[u] + cov_uk @ cov_kk_inv @ (theta_fixed[k] - mu[k])
        cov_cond = cov[u][:, u] - cov_uk @ cov_kk_inv @ cov_uk.T
        return u, mu_cond, cov_cond

    def analytic_conditional_mean_std(self, obs, condition_mask, theta_fixed):
        u, mu_cond, cov_cond = self._conditional_gaussian(obs, condition_mask, theta_fixed)
        mean_full = theta_fixed.clone()
        std_full = torch.zeros(self.dim_params)
        mean_full[u] = mu_cond
        std_full[u] = torch.sqrt(torch.diag(cov_cond).clamp_min(1e-12))
        return mean_full, std_full

    def sample_analytic_conditional(self, obs, condition_mask, theta_fixed, n_samples):
        """Samples of theta_unknown from the FULL (correlated) analytic conditional."""
        u, mu_cond, cov_cond = self._conditional_gaussian(obs, condition_mask, theta_fixed)
        cov_cond = cov_cond + 1e-8 * torch.eye(u.numel())
        L = torch.linalg.cholesky(cov_cond)
        return (mu_cond.unsqueeze(0) + torch.randn(n_samples, u.numel()) @ L.T).clamp(0, 1)

    def _soft_conditional_gaussian(self, obs, condition_mask, v_stmt, sigma_stmt):
        """Exact p(theta_unknown | x, statements) for SOFT conditioning.

        A statement (v_i, sigma_i) on a known coordinate is a Gaussian likelihood
        N(v_i; theta_i, sigma_i^2). sigma_i = 0 statements are exact conditioning;
        sigma_i > 0 are fused as likelihood tilts of the x-posterior. Everything is
        Gaussian, so the result is exact. Returns (idx_u, mean_u, cov_u) with idx_u
        the sorted unknown indices.
        """
        mu, C = self._theta_posterior(obs)
        known = condition_mask == 1
        hard = known & (sigma_stmt <= 0)
        soft = known & (sigma_stmt > 0)
        idx_rest = torch.where(~hard)[0]           # soft-known + unknown coords

        # 1) exact Gaussian conditioning on the hard statements.
        if hard.any():
            ih = torch.where(hard)[0]
            Chh = C[ih][:, ih]
            Crh = C[idx_rest][:, ih]
            mu_r = mu[idx_rest] + Crh @ torch.linalg.solve(Chh, v_stmt[ih] - mu[ih])
            C_r = C[idx_rest][:, idx_rest] - Crh @ torch.linalg.solve(Chh, Crh.T)
        else:
            mu_r, C_r = mu[idx_rest], C[idx_rest][:, idx_rest]

        # 2) fuse the Gaussian statement likelihoods on the soft coords.
        if soft.any():
            rest_list = idx_rest.tolist()
            pos = torch.tensor([rest_list.index(int(j)) for j in torch.where(soft)[0]])
            S = torch.zeros(pos.numel(), idx_rest.numel())
            S[torch.arange(pos.numel()), pos] = 1.0
            Dinv = torch.diag(1.0 / sigma_stmt[soft] ** 2)
            natural = torch.linalg.solve(C_r, mu_r) + S.T @ Dinv @ v_stmt[soft]
            prec = torch.linalg.inv(C_r) + S.T @ Dinv @ S
            C_r = torch.linalg.inv(prec)
            mu_r = C_r @ natural

        # 3) marginal over the unknown coordinates (sorted order).
        idx_u = torch.where(condition_mask == 0)[0]
        rest_list = idx_rest.tolist()
        pu = torch.tensor([rest_list.index(int(j)) for j in idx_u])
        return idx_u, mu_r[pu], C_r[pu][:, pu]

    def sample_analytic_soft_conditional(self, obs, condition_mask, v_stmt,
                                         sigma_stmt, n_samples):
        """Samples of theta_unknown from the exact SOFT conditional, sorted-unknown
        order, given per-coordinate statements (v, sigma) on the known coords."""
        idx_u, mu_u, cov_u = self._soft_conditional_gaussian(
            obs, condition_mask, v_stmt, sigma_stmt)
        cov_u = cov_u + 1e-8 * torch.eye(idx_u.numel())
        L = torch.linalg.cholesky(cov_u)
        return (mu_u.unsqueeze(0) + torch.randn(n_samples, idx_u.numel()) @ L.T).clamp(0, 1)


def get_task(name, **kwargs):
    if name != CorrelatedGaussian.name:
        raise ValueError(f"unknown task: {name}")
    return CorrelatedGaussian(**kwargs)
