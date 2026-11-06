"""Flow-matching transformer for amortized conditional inference.

Each parameter is one token. A token says whether its parameter is unknown,
known exactly, or known with an uncertainty, so one trained network can answer
any conditional query. The backbone (attention and feed-forward blocks modulated
by the flow time) is shared; the `conditioning` argument selects how the
observation x reaches the parameter tokens:

  "cross_two"     x is compressed once into K summary tokens. In every layer the
                  parameter tokens first cross-attend to the summary, then
                  self-attend. This is the compressed model used in the paper.
  "cross_single"  as above, with one attention over parameter and summary tokens.
  "cross_tab"     one token per observation coefficient, encoded once.
  "joint"         one token per observation coefficient, attended to jointly with
                  the parameter tokens at every step. This is the joint-attention
                  baseline.
  "mlp"           no attention between tokens; x is added to each parameter token.

Because the summary tokens depend only on x, they are computed once per
observation and reused across solver steps, samples and conditional queries.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, time_scale=1000.0):
        super().__init__()
        self.dim = dim
        # Flow-matching time t lives in [0, 1]. The sinusoidal frequencies here
        # are all <= 1, so without scaling almost all embedding dimensions are
        # nearly constant across t. Scaling t, as for a continuous timestep,
        # spreads it across the frequency bank.
        self.time_scale = time_scale

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = (t * self.time_scale)[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Norm: time embedding modulates scale and shift."""

    def __init__(self, d_model, d_time):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_time, 2 * d_model)

    def forward(self, x, t_emb):
        scale, shift = self.proj(t_emb).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class TransformerBlock(nn.Module):
    """Shared block: (optional) self-attention over the token set, then FFN,
    each with time-modulated AdaLayerNorm and a residual. Used by cross_single
    (tokens = [theta, obs]), joint (tokens = [theta, x]), and mlp (no attention).
    """

    def __init__(self, d_model, n_heads, d_time, dropout=0.0, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                              batch_first=True)
            self.norm_attn = AdaLayerNorm(d_model, d_time)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm_ffn = AdaLayerNorm(d_model, d_time)

    def forward(self, tokens, t_emb):
        if self.use_attention:
            h = self.norm_attn(tokens, t_emb)
            tokens = tokens + self.attn(h, h, h, need_weights=False)[0]
        h = self.norm_ffn(tokens, t_emb)
        tokens = tokens + self.ffn(h)
        return tokens


class CrossThenSelfBlock(nn.Module):
    """Two-step block for cross_two: parameter tokens first cross-attend to the
    (fixed) observation tokens, then self-attend among themselves, then FFN. The
    cross and self attentions have SEPARATE projections, so 'reading x' and
    'relating parameters' use distinct subspaces.
    """

    def __init__(self, d_model, n_heads, d_time, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm_cross = AdaLayerNorm(d_model, d_time)
        self.norm_self = AdaLayerNorm(d_model, d_time)
        self.norm_ffn = AdaLayerNorm(d_model, d_time)

    def forward(self, param_tokens, obs_tokens, t_emb):
        h = self.norm_cross(param_tokens, t_emb)
        param_tokens = param_tokens + self.cross_attn(h, obs_tokens, obs_tokens,
                                                      need_weights=False)[0]
        h = self.norm_self(param_tokens, t_emb)
        param_tokens = param_tokens + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm_ffn(param_tokens, t_emb)
        param_tokens = param_tokens + self.ffn(h)
        return param_tokens


class PlainTransformerBlock(nn.Module):
    """Time-independent pre-norm self-attention + FFN. Used to contextualize the
    observation tokens ONCE (no AdaLN time modulation), so the result depends only
    on x and can be cached and reused across ODE steps, samples, and conditionings.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, tokens):
        h = self.norm1(tokens)
        tokens = tokens + self.attn(h, h, h, need_weights=False)[0]
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class CFMTransformer(nn.Module):
    """Flow-matching transformer with a selectable conditioning mechanism.

    forward returns the per-parameter velocity; sample() integrates the ODE and
    (optionally) maps back through a bounded ParamTransform.
    """

    def __init__(
        self,
        n_params=8,
        n_dct=32,
        d_model=128,
        n_obs_tokens=4,
        n_heads=4,
        n_layers=6,
        n_enc_layers=2,
        dropout=0.0,
        time_scale=1000.0,
        conditioning="cross_two",
        soft_conditioning=False,
        stmt_embed="joint",
        n_ctx_features=0,
    ):
        super().__init__()
        assert conditioning in ("cross_two", "cross_single", "joint", "mlp",
                                "cross_tab"), conditioning
        self.n_params = n_params
        self.n_obs_tokens = n_obs_tokens
        self.n_enc_layers = n_enc_layers
        self.d_model = d_model
        self.time_scale = time_scale
        self.conditioning = conditioning
        self.n_ctx_features = n_ctx_features
        # Observation layout: mean, standard deviation and DCT coefficients, followed
        # by optional context features (the noise level and the window edges).
        self.dim_x = n_dct + 2 + n_ctx_features
        self.param_transform = None
        # Optional per-dimension observation standardization (mean, std) fit on the
        # TRAINING observations. DCT coefficients span very different scales
        # (low-frequency >> high-frequency); standardizing gives the obs encoder a
        # well-scaled input. It is applied only in _prep_x, so x is passed unscaled
        # everywhere else and is never standardized twice.
        self.x_stats = None

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(d_model, time_scale=time_scale),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Parameter token pieces, shared by all modes.
        self.param_id_embed = nn.Embedding(n_params, d_model)
        self.value_proj = nn.Linear(1, d_model)          # embeds any scalar value
        self.cond_embed = nn.Embedding(2, d_model)       # 0=unknown, 1=known/observed

        # Uncertain conditioning: a known parameter carries a statement (v, sigma),
        # "the value is about v with uncertainty sigma", instead of an exact value.
        # v is in [0, 1] units and may lie outside the box near a boundary, and
        # sigma = 0 recovers exact conditioning. When trained on corrupted statements,
        # the optimal model is the posterior given the uncertain statement.
        #
        # stmt_embed selects how the statement enters the token:
        #   "joint"   one Linear(2, d) on (v, sigma).
        #   "shared"  the value goes through value_proj in the transformed space, as
        #             for an exactly known parameter, plus a Linear(2, d) on
        #             (sigma, overflow), where overflow is the excursion beyond [0, 1].
        self.soft_conditioning = soft_conditioning
        self.stmt_embed = stmt_embed
        self.sigma_max = None            # largest trained uncertainty, set by the trainer
        if soft_conditioning:
            if stmt_embed == "shared":
                self.stmt_aux_proj = nn.Linear(2, d_model)   # (sigma, overflow)
            else:
                self.stmt_proj = nn.Linear(2, d_model)       # (v_raw, sigma)

        # The observation is not layer-normalized across its features: for a small
        # dim_x this would remove almost all of its information. Any per-feature
        # standardization uses dataset statistics instead (see x_stats).

        # Mode-specific x representation.
        if conditioning in ("cross_two", "cross_single"):
            self.obs_encoder = nn.Sequential(
                nn.Linear(self.dim_x, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, n_obs_tokens * d_model),
            )
            self.obs_pos_embed = nn.Embedding(n_obs_tokens, d_model)
        elif conditioning == "joint":
            self.x_id_embed = nn.Embedding(self.dim_x, d_model)
        elif conditioning == "cross_tab":
            # One token per observation coefficient, without compression, with a
            # position embedding to distinguish coefficients. A small time-independent
            # encoder contextualizes them. The result depends only on x, so it is
            # computed once and reused across solver steps, samples and queries.
            self.x_id_embed = nn.Embedding(self.dim_x, d_model)
            self.obs_encoder_blocks = nn.ModuleList([
                PlainTransformerBlock(d_model, n_heads, dropout)
                for _ in range(n_enc_layers)
            ])
        else:  # mlp
            self.x_vec_encoder = nn.Sequential(
                nn.Linear(self.dim_x, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        if conditioning in ("cross_two", "cross_tab"):
            self.blocks = nn.ModuleList([
                CrossThenSelfBlock(d_model, n_heads, d_model, dropout)
                for _ in range(n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_model, dropout,
                                 use_attention=(conditioning != "mlp"))
                for _ in range(n_layers)
            ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        # Zero-init the velocity head and the AdaLN time-modulation, so the model
        # starts as a near-identity flow (zero velocity, neutral time conditioning)
        # and learns incrementally. Standard for diffusion/flow transformers (DiT)
        # and improves early-training stability.
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        for m in self.modules():
            if isinstance(m, AdaLayerNorm):
                nn.init.zeros_(m.proj.weight)
                nn.init.zeros_(m.proj.bias)

    def _param_tokens(self, theta_input, condition_mask, statements=None):
        B = theta_input.shape[0]
        pid = torch.arange(self.n_params, device=theta_input.device).expand(B, -1)
        val = self.value_proj(theta_input.unsqueeze(-1))
        if self.soft_conditioning and statements is not None:
            # Known tokens carry the statement instead of the exact value, so the
            # network never sees the true value behind an uncertain statement.
            if self.stmt_embed == "shared":
                # sigma=0 consistency by construction: the value part goes through
                # the SAME value_proj as the flowing coordinates, in the transformed
                # space; tolerance and the raw out-of-box overflow enter separately.
                v_raw, sig = statements[..., 0], statements[..., 1]
                tr = self.param_transform
                v_t = tr.forward(v_raw) if tr is not None else v_raw.clamp(0, 1)
                overflow = v_raw - v_raw.clamp(0, 1)
                stmt = (self.value_proj(v_t.unsqueeze(-1))
                        + self.stmt_aux_proj(torch.stack([sig, overflow], dim=-1)))
            else:
                stmt = self.stmt_proj(statements)                 # (B, d, D)
            val = torch.where(condition_mask.unsqueeze(-1).bool(), stmt, val)
        return self.param_id_embed(pid) + val + self.cond_embed(condition_mask.long())

    def _prep_x(self, x):
        """Standardize the raw observation if x_stats is set (single entry point)."""
        if self.x_stats is None:
            return x
        mean, std = self.x_stats
        return (x - mean.to(x.device)) / std.to(x.device)

    def encode_observation(self, x):
        """Observation tokens, for the conditionings that read a fixed context.

        The result depends ONLY on x, so it is computed once and reused across
        every ODE step, sample, and conditioning subset. Takes the RAW x (it
        standardizes internally), and returns (B, n_tokens, d_model):
          cross_tab            one token per coefficient, contextualized
          cross_two/_single    n_obs_tokens compressed tokens
        """
        x = self._prep_x(x)
        B = x.shape[0]
        if self.conditioning == "cross_tab":
            pos = torch.arange(self.dim_x, device=x.device).expand(B, -1)
            obs = self.value_proj(x.unsqueeze(-1)) + self.x_id_embed(pos)
            for blk in self.obs_encoder_blocks:
                obs = blk(obs)
            return obs
        if self.conditioning in ("cross_two", "cross_single"):
            pos = torch.arange(self.n_obs_tokens, device=x.device).expand(B, -1)
            return (self.obs_encoder(x).reshape(B, self.n_obs_tokens, self.d_model)
                    + self.obs_pos_embed(pos))
        raise ValueError(f"{self.conditioning} has no cacheable observation encoding")

    def forward(self, theta_input, t, x_obs, condition_mask, obs_tokens=None,
                statements=None):
        """
        theta_input:   (B, n_params) noised for unknown dims, true for known dims
        t:             (B,) flow-matching time in [0, 1]
        x_obs:         (B, dim_x) observation
        condition_mask:(B, n_params) 1=known/fixed, 0=unknown/infer
        obs_tokens:    (cross_tab) optional precomputed observation encoding to reuse
                       (from encode_observation); avoids re-encoding x every step.
        statements:    (soft conditioning) (B, n_params, 2) [v_raw, sigma] per known
                       parameter; replaces the exact value at known tokens.
        returns velocity (B, n_params) over the parameter dimensions.
        """
        B = theta_input.shape[0]
        device = theta_input.device
        t_emb = self.time_mlp(t)

        theta_tokens = self._param_tokens(theta_input, condition_mask, statements)
        x = self._prep_x(x_obs)

        if self.conditioning == "cross_tab":
            # Parameter tokens cross- then self-attend to the (cached) per-coefficient
            # observation tokens. The observation is encoded once and reused.
            if obs_tokens is None:
                # encode_observation standardizes internally -> pass the RAW x_obs,
                # not the already-prepped x (double-standardization otherwise).
                obs_tokens = self.encode_observation(x_obs)
            if obs_tokens.shape[0] == 1 and B > 1:      # broadcast a cached encoding
                obs_tokens = obs_tokens.expand(B, -1, -1)
            tokens = theta_tokens
            for block in self.blocks:
                tokens = block(tokens, obs_tokens, t_emb)
            return self.output_head(self.output_norm(tokens)).squeeze(-1)

        if self.conditioning == "cross_two":
            # Fixed obs context; parameter tokens cross- then self-attend each layer.
            # The encoding depends only on x, so sample() computes it once and
            # passes it in.
            if obs_tokens is None:
                pos = torch.arange(self.n_obs_tokens, device=device).expand(B, -1)
                obs = (self.obs_encoder(x).reshape(B, self.n_obs_tokens, self.d_model)
                       + self.obs_pos_embed(pos))
            else:
                obs = (obs_tokens.expand(B, -1, -1)
                       if obs_tokens.shape[0] == 1 and B > 1 else obs_tokens)
            tokens = theta_tokens
            for block in self.blocks:
                tokens = block(tokens, obs, t_emb)
            theta_out = self.output_norm(tokens)
            return self.output_head(theta_out).squeeze(-1)

        if self.conditioning == "cross_single":
            pos = torch.arange(self.n_obs_tokens, device=device).expand(B, -1)
            obs = (self.obs_encoder(x).reshape(B, self.n_obs_tokens, self.d_model)
                   + self.obs_pos_embed(pos))
            tokens = torch.cat([theta_tokens, obs], dim=1)

        elif self.conditioning == "joint":
            xid = torch.arange(self.dim_x, device=device).expand(B, -1)
            x_tokens = (
                self.value_proj(x.unsqueeze(-1))
                + self.x_id_embed(xid)
                + self.cond_embed(torch.ones(B, self.dim_x, dtype=torch.long, device=device))
            )
            tokens = torch.cat([theta_tokens, x_tokens], dim=1)

        else:  # mlp: inject x as a vector into each parameter token, no attention
            x_vec = self.x_vec_encoder(x).unsqueeze(1)          # (B, 1, D)
            tokens = theta_tokens + x_vec

        for block in self.blocks:
            tokens = block(tokens, t_emb)

        theta_out = self.output_norm(tokens[:, : self.n_params])
        return self.output_head(theta_out).squeeze(-1)

    @staticmethod
    def _sample_rk45(theta, field, unknown_mask, dt0, rtol, atol):
        """Adaptive Dormand-Prince 5(4) over t in [0, 1], batched.

        Step-size control uses the max error norm over the entire batch, so
        the step is as small as the hardest trajectory demands. FSAL: the
        7th stage of an accepted step is the 1st stage of the next.
        """
        t, dt = 0.0, float(dt0)
        k1 = field(theta, t)
        while t < 1.0 - 1e-9:
            dt = min(dt, 1.0 - t)
            m = unknown_mask
            k2 = field(theta + dt * (1 / 5 * k1) * m, t + dt / 5)
            k3 = field(theta + dt * (3 / 40 * k1 + 9 / 40 * k2) * m,
                       t + 3 * dt / 10)
            k4 = field(theta + dt * (44 / 45 * k1 - 56 / 15 * k2
                                     + 32 / 9 * k3) * m, t + 4 * dt / 5)
            k5 = field(theta + dt * (19372 / 6561 * k1 - 25360 / 2187 * k2
                                     + 64448 / 6561 * k3 - 212 / 729 * k4) * m,
                       t + 8 * dt / 9)
            k6 = field(theta + dt * (9017 / 3168 * k1 - 355 / 33 * k2
                                     + 46732 / 5247 * k3 + 49 / 176 * k4
                                     - 5103 / 18656 * k5) * m, t + dt)
            y5 = theta + dt * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4
                               - 2187 / 6784 * k5 + 11 / 84 * k6) * m
            k7 = field(y5, t + dt)
            y4 = theta + dt * (5179 / 57600 * k1 + 7571 / 16695 * k3
                               + 393 / 640 * k4 - 92097 / 339200 * k5
                               + 187 / 2100 * k6 + 1 / 40 * k7) * m
            scale = atol + rtol * torch.maximum(theta.abs(), y5.abs())
            err = float(((y5 - y4).abs() * m / scale).max())
            if err <= 1.0:
                t += dt
                theta, k1 = y5, k7
            dt *= min(5.0, max(0.2, 0.9 * (max(err, 1e-10)) ** -0.2))
        return theta

    @torch.no_grad()
    def sample(self, x_obs, condition_mask, theta_fixed, n_samples=1000, n_steps=20,
               obs_tokens=None, sigma=None, solver="midpoint",
               rtol=1e-3, atol=1e-4):
        """Sample from the conditional posterior p(theta_unknown | x, theta_known).

        x_obs: (dim_x,), condition_mask: (n_params,) 1=known, theta_fixed:
        (n_params,) known values in [0,1]. Returns (n_samples, n_params) in [0,1].
        obs_tokens: (cross_tab) a precomputed observation encoding to reuse across
        conditionals; if None it is encoded here.
        sigma: (soft conditioning) (n_params,) per-parameter tolerance; the known
        values become statements "about theta_fixed[i], tolerance sigma[i]".
        sigma=None or 0 is hard conditioning. theta_fixed may lie outside [0,1]
        when sigma>0 (a one-sided boundary statement); with soft statements the
        returned known columns echo the clamped statement value, not a sample.

        solver: ODE integrator for the probability flow.
          "euler"    1 network eval/step, local error O(dt^2), global O(dt)
          "midpoint" 2 evals/step, global O(dt^2)   <- default
          "heun"     2 evals/step, global O(dt^2), trapezoidal variant
          "rk4"      4 evals/step, global O(dt^4)
          "rk45"     adaptive Dormand-Prince 5(4); n_steps sets only the
                     INITIAL step (1/n_steps), then the step size is controlled
                     by rtol/atol. The error norm is the max over the whole
                     sample batch, so one hard trajectory slows all of them --
                     the price of batched sampling.
        Explicit Euler was the original default; at a fixed number of network
        evaluations the second-order methods integrate the same field far more
        accurately, so the posterior carries less discretization bias. Pass
        solver="euler" to reproduce numbers produced before this default changed.
        Note the cost accounting: n_steps counts STEPS, not evaluations. The
        network evaluations actually spent are recorded in self.last_nfe.
        """
        assert solver in ("euler", "midpoint", "heun", "rk4", "rk45"), solver
        device = next(self.parameters()).device
        x_batch = x_obs.unsqueeze(0).expand(n_samples, -1).to(device)
        cond_batch = condition_mask.unsqueeze(0).expand(n_samples, -1).to(device)
        fixed_raw = theta_fixed.to(device)
        fixed_batch = fixed_raw.clamp(0, 1).unsqueeze(0).expand(n_samples, -1)
        unknown_mask = 1 - cond_batch

        statements = None
        if self.soft_conditioning:
            sig = (torch.zeros(self.n_params, device=device) if sigma is None
                   else sigma.to(device))
            if self.sigma_max is not None and float(sig.max()) > self.sigma_max + 1e-6:
                print(f"WARNING: requested tolerance {float(sig.max()):.3f} exceeds "
                      f"the largest trained value {self.sigma_max:.3f}; the result "
                      f"is an extrapolation.")
            statements = torch.stack([fixed_raw, sig], dim=-1)      # (d, 2)
            statements = (statements * condition_mask.to(device).unsqueeze(-1)
                          ).unsqueeze(0).expand(n_samples, -1, -1)

        # Conditioned values are given in [0,1]; map into the model's working space
        # so known dims are held at the right place during the flow.
        if self.param_transform is not None:
            fixed_batch = self.param_transform.forward(fixed_batch)

        theta = torch.randn(n_samples, self.n_params, device=device)
        theta = theta * unknown_mask + fixed_batch * cond_batch

        # Encode the observation ONCE and reuse it across all ODE steps (and, via
        # sample_conditionals, across conditioning subsets too). Applies to every
        # conditioning whose observation context is time-independent.
        if self.conditioning in ("cross_tab", "cross_two", "cross_single") \
                and obs_tokens is None:
            obs_tokens = self.encode_observation(x_obs.unsqueeze(0).to(device))

        dt = 1.0 / n_steps
        nfe = 0

        def field(th, t_scalar):
            nonlocal nfe
            nfe += 1
            t = torch.full((n_samples,), t_scalar, device=device)
            return self.forward(th, t, x_batch, cond_batch,
                                obs_tokens=obs_tokens, statements=statements)

        if solver == "rk45":
            theta = self._sample_rk45(theta, field, unknown_mask,
                                      dt0=dt, rtol=rtol, atol=atol)
        else:
            for step in range(n_steps):
                t0 = step * dt
                k1 = field(theta, t0)
                if solver == "euler":
                    theta = theta + dt * k1 * unknown_mask
                elif solver == "midpoint":
                    # evaluate the field at the half step; the known coordinates
                    # stay pinned in the intermediate state too, or the
                    # conditioning would leak into the stage evaluation
                    k2 = field(theta + 0.5 * dt * k1 * unknown_mask,
                               t0 + 0.5 * dt)
                    theta = theta + dt * k2 * unknown_mask
                elif solver == "heun":
                    k2 = field(theta + dt * k1 * unknown_mask, t0 + dt)
                    theta = theta + 0.5 * dt * (k1 + k2) * unknown_mask
                else:   # rk4
                    k2 = field(theta + 0.5 * dt * k1 * unknown_mask,
                               t0 + 0.5 * dt)
                    k3 = field(theta + 0.5 * dt * k2 * unknown_mask,
                               t0 + 0.5 * dt)
                    k4 = field(theta + dt * k3 * unknown_mask, t0 + dt)
                    theta = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) \
                        * unknown_mask
        self.last_nfe = nfe

        if self.param_transform is not None:
            return self.param_transform.inverse(theta)
        return theta.clamp(0, 1)

    @torch.no_grad()
    def sample_conditionals(self, x_obs, condition_masks, theta_fixed,
                            n_samples=1000, n_steps=20, solver="midpoint"):
        """Sample MANY conditional posteriors for ONE observation, encoding the
        observation once and reusing it across all of them (cross_tab). This is
        where cross's amortization pays off: the observation cost is paid once and
        each additional conditioning adds only the cheap parameter-side flow, while
        joint/mlp re-process the observation for every conditioning.

        condition_masks: list of (n_params,) masks. Returns a list of sample sets.
        """
        device = next(self.parameters()).device
        obs_tokens = None
        if self.conditioning in ("cross_tab", "cross_two", "cross_single"):
            obs_tokens = self.encode_observation(x_obs.unsqueeze(0).to(device))
        return [
            self.sample(x_obs, cm, theta_fixed, n_samples=n_samples,
                        n_steps=n_steps, obs_tokens=obs_tokens, solver=solver)
            for cm in condition_masks
        ]


def sample_soft_statements(theta_raw, condition_mask, p_hard=0.5, sigma_max=0.2,
                           sigma_dist="uniform", sigma_min=0.005):
    """Training statements for soft conditioning: (B, n_params, 2) of [v_raw, sigma].

    Per KNOWN parameter independently: with prob p_hard the statement is exact
    (sigma=0, v=theta); otherwise sigma is drawn from sigma_dist and
    v = theta + sigma*eps in RAW [0,1] units, unclamped (near a boundary the
    statement may leave the box; clamping would destroy the likelihood shape
    exactly where soft conditioning is most used). Unknown positions are zeroed.

    sigma_dist:
      "uniform"    : sigma ~ U(0, sigma_max)
      "loguniform" : log sigma ~ U(log sigma_min, log sigma_max), with most
                     mass at small uncertainties.
    """
    if sigma_dist == "loguniform":
        u = torch.rand_like(theta_raw)
        sig = sigma_min * (sigma_max / sigma_min) ** u
    else:
        sig = torch.rand_like(theta_raw) * sigma_max
    hard = torch.rand_like(theta_raw) < p_hard
    sig = torch.where(hard, torch.zeros_like(sig), sig)
    v = theta_raw + sig * torch.randn_like(theta_raw)
    return torch.stack([v, sig], dim=-1) * condition_mask.unsqueeze(-1)


def sample_condition_masks(batch_size, n_params=8, device="cpu",
                           structured_groups=None):
    """Sample random conditioning patterns over the parameters for training.

    Uniform coverage of ALL masked/inferred combinations. Each sample draws a
    rate u ~ U(0,1) and marks each parameter known with probability u (random-
    rate Bernoulli). This gives equal weight to every NUMBER of fixed parameters
    (0..n) and, within each, an even spread over the specific subsets, so every
    combination of fixed/inferred parameters is trained and none is favored. The
    full posterior (nothing fixed) naturally gets ~1/(n+1) of the mass. A safety
    step guarantees at least one unknown so the masked velocity loss is always
    defined (1 = known/fixed, 0 = unknown/infer). Works for any n_params >= 2.

    ``structured_groups`` (optional list of index lists) overlays a fraction of
    task-meaningful group conditionals on top. Leave it None for pure uniform
    coverage; passing groups deliberately skews toward those patterns.
    """
    # Random-rate Bernoulli -> uniform over the number fixed, even over subsets.
    u = torch.rand(batch_size, 1, device=device)
    masks = (torch.rand(batch_size, n_params, device=device) < u).float()
    # Redraw any all-known rows (invalid) rather than flipping one bit, so their
    # mass redistributes uniformly instead of piling onto the "infer one" size.
    for _ in range(8):
        bad = torch.where(masks.sum(1) == n_params)[0]
        if bad.numel() == 0:
            break
        u2 = torch.rand(bad.numel(), 1, device=device)
        masks[bad] = (torch.rand(bad.numel(), n_params, device=device) < u2).float()

    # Optional: overwrite a fraction with group-structured conditionals.
    if structured_groups:
        group_tensors = [torch.as_tensor(g, device=device, dtype=torch.long)
                         for g in structured_groups]
        g_sel = torch.rand(batch_size, device=device) < 0.30
        for idx in torch.where(g_sel)[0]:
            g = group_tensors[torch.randint(0, len(group_tensors), (1,)).item()]
            if torch.rand(1, device=device).item() < 0.5:
                masks[idx, :] = 0.0
                masks[idx, g] = 1.0    # fix this group, infer the rest
            else:
                masks[idx, :] = 1.0
                masks[idx, g] = 0.0    # infer this group, fix the rest

    # Safety: never leave a sample fully known (loss would be undefined).
    all_known = masks.sum(1) == n_params
    for idx in torch.where(all_known)[0]:
        masks[idx, torch.randint(0, n_params, (1,), device=device).item()] = 0.0

    return masks
