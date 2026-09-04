import numpy as np
import torch

from nnsvs.base import BaseModel, PredictionType


def _bcmt_to_btc(x):
    """Convert (B, 1, C, T) to (B, T, C)"""
    return x.squeeze(1).transpose(1, 2)


def _btc_to_bcmt(x):
    """Convert (B, T, C) to (B, 1, C, T)"""
    return x.transpose(1, 2)[:, None, :, :]


class FlowMatching(BaseModel):
    """Rectified flow (flow matching) generator for a single feature stream.

    This is a drop-in replacement of :py:class:`GaussianDiffusion`; the same
    denoiser (:py:class:`nnsvs.diffsinger.DiffNet`) and encoder can be used.
    Instead of predicting the noise of a diffusion process, the model predicts
    the velocity field of the straight path between the Gaussian prior and the
    data:

    .. code-block::

        # training
        x1 = y / norm_scale        # data
        x0 ~ N(0, I)               # prior
        t ~ U{0, ..., num_timesteps-1}
        x_t = (t / num_timesteps) * x1 + (1 - t / num_timesteps) * x0
        u = x1 - x0                # target velocity (constant along the path)
        v = denoise_fn(x_t, t, cond)

        # inference (Euler ODE solver with K_step steps)
        x = x0
        for i in range(K_step):
            x = x + (1 / K_step) * denoise_fn(x, i / K_step, cond)
        y = x * norm_scale

    :py:meth:`forward` returns ``(u, v)`` so that the existing loss computation
    for diffusion models in :py:mod:`nnsvs.bin.train_acoustic` (which computes
    ``criterion(stream[0], stream[1])`` for two-element tuples) works as the
    flow matching loss without any change.

    Args:
        in_dim (int): Input dimension. Must be equal to ``encoder.in_dim``
            if the encoder is used.
        out_dim (int): Output dimension. Must be equal to ``denoise_fn.in_dim``.
        denoise_fn (nn.Module): Velocity estimator (e.g. ``DiffNet``).
        encoder (nn.Module): Optional encoder that maps conditional features
            to hidden representations of ``denoise_fn.encoder_hidden_dim``.
        num_timesteps (int): Number of discrete time steps used for the
            time-step embedding of ``denoise_fn``.
        K_step (int): Number of ODE steps at inference time.
        norm_scale (float): Scale to convert mean-var normalized data
            (~ N(0, 1)) to the data space of the flow. Unlike diffusion models,
            flow matching does not assume the data to be in [-1, 1], so the
            default is 1.0.
        solver (str): ODE solver. ``euler`` or ``midpoint``.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        denoise_fn,
        encoder=None,
        num_timesteps=1000,
        K_step=100,
        norm_scale=1.0,
        solver="euler",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.denoise_fn = denoise_fn
        self.encoder = encoder
        self.num_timesteps = num_timesteps
        self.K_step = K_step
        self.norm_scale = norm_scale
        self.solver = solver

        if encoder is not None:
            assert encoder.in_dim == in_dim, "encoder input dim must match in_dim"
        assert out_dim == denoise_fn.in_dim, "denoise_fn input dim must match out_dim"
        if solver not in ["euler", "midpoint"]:
            raise ValueError(f"Unknown ODE solver: {solver}")

    def prediction_type(self):
        return PredictionType.DIFFUSION

    def _norm(self, x):
        return x / self.norm_scale

    def _denorm(self, x):
        return x * self.norm_scale

    def _encode(self, cond, lengths=None):
        if self.encoder is not None:
            cond = self.encoder(cond, lengths)
        # (B, T, H) -> (B, H, T)
        return cond.transpose(1, 2)

    def _forward_flow(self, cond, lengths=None, y=None):
        """Sample a point on the flow path and predict its velocity

        Args:
            cond (torch.Tensor): conditioning features of shape (B, T, in_dim)
                (or (B, T, encoder_hidden_dim) if the encoder is not used)
            lengths (torch.Tensor): lengths of each sequence in the batch
            y (torch.Tensor): ground truth of shape (B, T, out_dim)

        Returns:
            tuple: (u, v_pred, x_t, t) where u, v_pred and x_t are of shape
            (B, 1, out_dim, T) and t is of shape (B, 1, 1, 1) normalized to
            [0, 1).
        """
        assert y is not None, "target features must be given at training time"
        B = cond.shape[0]
        cond = self._encode(cond, lengths)

        # NOTE: the time step is passed to denoise_fn as an integer so that
        # the sinusoidal time-step embedding works in the same way as DDPM
        t = torch.randint(0, self.num_timesteps, (B,), device=cond.device).long()
        # (B, 1, out_dim, T)
        x1 = _btc_to_bcmt(self._norm(y))
        x0 = torch.randn_like(x1)
        tt = (t.float() / self.num_timesteps).view(B, 1, 1, 1)
        xt = tt * x1 + (1.0 - tt) * x0
        v_pred = self.denoise_fn(xt, t, cond)

        return x1 - x0, v_pred, xt, tt

    def forward(self, cond, lengths=None, y=None):
        """Forward step

        Args:
            cond (torch.Tensor): conditioning features of shape (B, T, in_dim)
            lengths (torch.Tensor): lengths of each sequence in the batch
            y (torch.Tensor): ground truth of shape (B, T, out_dim)

        Returns:
            tuple: (target velocity, predicted velocity), both of shape
            (B, T, out_dim)
        """
        u, v_pred, _, _ = self._forward_flow(cond, lengths, y)
        return _bcmt_to_btc(u), _bcmt_to_btc(v_pred)

    def _clip_velocity(self, x, v, t_float, clip_bounds=None):
        """Optionally constrain the velocity so that the endpoint is bounded

        The one-step estimate of the endpoint ``x1_hat = x + (1 - t) * v`` is
        clipped to ``clip_bounds`` and then the velocity is re-computed.
        No-op if ``clip_bounds`` is None.

        Args:
            x (torch.Tensor): current state of shape (B, 1, out_dim, T)
            v (torch.Tensor): predicted velocity of shape (B, 1, out_dim, T)
            t_float (float): current time in [0, 1)
            clip_bounds (tuple): (lower, upper) tensors broadcastable to x

        Returns:
            torch.Tensor: velocity of shape (B, 1, out_dim, T)
        """
        if clip_bounds is None:
            return v
        lower, upper = clip_bounds
        x1_hat = x + (1.0 - t_float) * v
        x1_hat = torch.minimum(torch.maximum(x1_hat, lower), upper)
        return (x1_hat - x) / (1.0 - t_float)

    def _velocity(self, x, cond, t_float, clip_bounds=None):
        t = torch.full(
            (x.shape[0],),
            int(t_float * self.num_timesteps),
            device=x.device,
            dtype=torch.long,
        )
        v = self.denoise_fn(x, t, cond)
        return self._clip_velocity(x, v, t_float, clip_bounds)

    def _sample(self, cond, lengths=None, clip_bounds=None):
        """Solve the ODE from the Gaussian prior to the data

        Args:
            cond (torch.Tensor): conditioning features of shape (B, T, in_dim)
            lengths (torch.Tensor): lengths of each sequence in the batch
            clip_bounds (tuple): optional (lower, upper) bounds of the endpoint
                in the normalized data space, broadcastable to
                (B, 1, out_dim, T)

        Returns:
            torch.Tensor: generated features of shape (B, T, out_dim)
        """
        cond = self._encode(cond, lengths)
        shape = (cond.shape[0], 1, self.out_dim, cond.shape[2])
        x = torch.randn(shape, device=cond.device)

        dt = 1.0 / self.K_step
        for i in range(self.K_step):
            t_float = i * dt
            if self.solver == "euler":
                v = self._velocity(x, cond, t_float, clip_bounds)
            else:
                # midpoint (2nd order Runge-Kutta)
                v_mid = self._velocity(x, cond, t_float, clip_bounds)
                v = self._velocity(
                    x + 0.5 * dt * v_mid, cond, t_float + 0.5 * dt, clip_bounds
                )
            x = x + dt * v

        return self._denorm(_bcmt_to_btc(x))

    @torch.no_grad()
    def inference(self, cond, lengths=None):
        """Inference step

        Args:
            cond (torch.Tensor): conditioning features of shape (B, T, in_dim)
            lengths (torch.Tensor): lengths of each sequence in the batch

        Returns:
            torch.Tensor: generated features of shape (B, T, out_dim)
        """
        return self._sample(cond, lengths)


class FlowMatchingF0(FlowMatching):
    """Flow matching-based log-F0 generator with residual F0 prediction

    With ``predict_residual=True`` (default), the flow models the residual
    between the log-F0 of the musical score and the target log-F0, following
    the convention of NNSVS's residual F0 prediction models such as
    :py:class:`nnsvs.acoustic_models.BiLSTMResF0NonAttentiveDecoder`:

    .. code-block::

        x1 = (lf0_target_denorm - lf0_score_denorm) / residual_scale
        lf0_pred = (lf0_score_denorm + x1 * residual_scale - out_lf0_mean)
                    / out_lf0_scale

    where ``residual_scale = residual_scale_cent * log(2) / 1200``; i.e.
    ``x1 = 1`` corresponds to a residual of ``residual_scale_cent`` cent.
    Unlike the deterministic models, the residual is not bounded by a scaled
    tanh but by clipping the one-step estimate of the endpoint during the ODE
    integration (``clip_cent``).

    With ``predict_residual=False``, the flow models the normalized log-F0
    directly (as in TechSinger) and ``clip_cent`` constrains the generated
    log-F0 to be within ``clip_cent`` cent of the score.

    NOTE: ``in_lf0_min``, ``in_lf0_max``, ``out_lf0_mean`` and
    ``out_lf0_scale`` are automatically set by the parent multi-stream model
    (see ``NPSSMDNMultistreamParametricModel._set_lf0_params``), so they can be
    left unspecified. ``in_lf0_idx`` is NOT propagated and must be set
    explicitly to the same value as the parent's ``in_lf0_idx``.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension. Must be 1.
        denoise_fn (nn.Module): Velocity estimator (e.g. ``DiffNet``).
        encoder (nn.Module): Optional encoder.
        num_timesteps (int): Number of discrete time steps.
        K_step (int): Number of ODE steps at inference time.
        norm_scale (float): Scale to convert the data to the space of the flow.
        solver (str): ODE solver. ``euler`` or ``midpoint``.
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum of lf0 in the training data of input feats
        in_lf0_max (float): maximum of lf0 in the training data of input feats
        out_lf0_idx (int): index of lf0 in output features. Must be 0.
        out_lf0_mean (float): mean of lf0 in the training data of output feats
        out_lf0_scale (float): scale of lf0 in the training data of output feats
        predict_residual (bool): If True, model the residual F0 with respect to
            the log-F0 of the musical score.
        residual_scale_cent (float): Scale of the residual F0 in cent.
        clip_cent (float): Maximum deviation from the score's log-F0 in cent.
            Set to null to disable the endpoint clipping.
    """

    def __init__(
        self,
        in_dim,
        out_dim=1,
        denoise_fn=None,
        encoder=None,
        num_timesteps=1000,
        K_step=100,
        norm_scale=1.0,
        solver="euler",
        in_lf0_idx=51,
        in_lf0_min=None,
        in_lf0_max=None,
        out_lf0_idx=0,
        out_lf0_mean=None,
        out_lf0_scale=None,
        predict_residual=True,
        residual_scale_cent=600,
        clip_cent=600,
    ):
        assert out_dim == 1, "FlowMatchingF0 only supports out_dim=1"
        assert out_lf0_idx == 0, "out_lf0_idx must be 0"
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            denoise_fn=denoise_fn,
            encoder=encoder,
            num_timesteps=num_timesteps,
            K_step=K_step,
            norm_scale=norm_scale,
            solver=solver,
        )
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale
        self.predict_residual = predict_residual
        self.residual_scale_cent = residual_scale_cent
        self.clip_cent = clip_cent

    def has_residual_lf0_prediction(self):
        return True

    @property
    def residual_scale(self):
        """Scale of the residual F0 in the log-F0 (log-Hz) domain"""
        return self.residual_scale_cent * np.log(2) / 1200

    def _lf0_score_denorm(self, x):
        """Denormalize the log-F0 of the musical score

        Args:
            x (torch.Tensor): input features of shape (B, T, in_dim)

        Returns:
            torch.Tensor: log-F0 of the musical score of shape (B, T, 1)
        """
        lf0_score = x[:, :, self.in_lf0_idx].unsqueeze(-1)
        return lf0_score * (self.in_lf0_max - self.in_lf0_min) + self.in_lf0_min

    def _to_normalized_lf0(self, lf0_denorm):
        return (lf0_denorm - self.out_lf0_mean) / self.out_lf0_scale

    def _compute_clip_bounds(self, lf0_score_denorm):
        """Compute the bounds of the endpoint in the flow's data space

        Args:
            lf0_score_denorm (torch.Tensor): log-F0 of the musical score of
                shape (B, T, 1)

        Returns:
            tuple: (lower, upper) tensors of shape (B, 1, 1, T), or None if
            the endpoint clipping is disabled
        """
        if self.clip_cent is None:
            return None
        # maximum deviation in the log-F0 (log-Hz) domain
        max_lf0_ratio = self.clip_cent * np.log(2) / 1200
        if self.predict_residual:
            upper = torch.full_like(
                lf0_score_denorm, max_lf0_ratio / self.residual_scale
            )
            lower = -upper
        else:
            upper = self._to_normalized_lf0(lf0_score_denorm + max_lf0_ratio)
            lower = self._to_normalized_lf0(lf0_score_denorm - max_lf0_ratio)
        return (
            _btc_to_bcmt(self._norm(lower)),
            _btc_to_bcmt(self._norm(upper)),
        )

    def forward(self, x, lengths=None, y=None):
        """Forward step

        Args:
            x (torch.Tensor): input features of shape (B, T, in_dim)
            lengths (torch.Tensor): lengths of each sequence in the batch
            y (torch.Tensor): normalized target log-F0 of shape (B, T, 1)

        Returns:
            tuple: ((target velocity, predicted velocity), residual log-F0).
            The velocities are of shape (B, T, 1) and the residual log-F0
            (a one-step estimate, in the denormalized log-F0 domain, used for
            the pitch regularization loss) is of shape (B, T, 1).
        """
        assert y is not None, "target log-F0 must be given at training time"
        lf0_score_denorm = self._lf0_score_denorm(x)

        if self.predict_residual:
            lf0_target_denorm = y * self.out_lf0_scale + self.out_lf0_mean
            y_flow = (lf0_target_denorm - lf0_score_denorm) / self.residual_scale
        else:
            y_flow = y

        u, v_pred, xt, tt = self._forward_flow(x, lengths, y_flow)

        # One-step estimate of the endpoint: x1_hat = x_t + (1 - t) * v
        # NOTE: this is noisy for small t, so it is recommended to start with
        # pitch_reg_weight = 0
        x1_hat = self._denorm(_bcmt_to_btc(xt + (1.0 - tt) * v_pred))
        if self.predict_residual:
            lf0_residual = x1_hat * self.residual_scale
        else:
            lf0_residual = (
                x1_hat * self.out_lf0_scale + self.out_lf0_mean - lf0_score_denorm
            )

        return (_bcmt_to_btc(u), _bcmt_to_btc(v_pred)), lf0_residual

    @torch.no_grad()
    def inference(self, x, lengths=None):
        """Inference step

        Args:
            x (torch.Tensor): input features of shape (B, T, in_dim)
            lengths (torch.Tensor): lengths of each sequence in the batch

        Returns:
            torch.Tensor: normalized log-F0 of shape (B, T, 1)
        """
        lf0_score_denorm = self._lf0_score_denorm(x)
        clip_bounds = self._compute_clip_bounds(lf0_score_denorm)
        x1 = self._sample(x, lengths, clip_bounds=clip_bounds)

        if self.predict_residual:
            lf0_denorm = lf0_score_denorm + x1 * self.residual_scale
            return self._to_normalized_lf0(lf0_denorm)
        return x1
