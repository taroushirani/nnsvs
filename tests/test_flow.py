import numpy as np
import pytest
import torch

from nnsvs.acoustic_models import NPSSMDNMultistreamParametricModel
from nnsvs.base import PredictionType
from nnsvs.diffsinger import DiffNet
from nnsvs.model import FFN, LSTMEncoder
from nnsvs.techsinger import FlowMatching, FlowMatchingF0
from nnsvs.util import init_seed

# dummy log-F0 statistics
IN_LF0_MIN = 5.3936276
IN_LF0_MAX = 6.491111
OUT_LF0_MEAN = 5.953093881972361
OUT_LF0_SCALE = 0.23435173188961034


def _diffnet(in_dim, encoder_hidden_dim):
    return DiffNet(
        in_dim=in_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        residual_layers=2,
        residual_channels=4,
        dilation_cycle_length=2,
    )


def _encoder(in_dim, out_dim):
    return LSTMEncoder(
        in_dim=in_dim,
        hidden_dim=2,
        out_dim=out_dim,
        num_layers=1,
        dropout=0.0,
        init_type="none",
    )


@pytest.mark.parametrize("solver", ["euler", "midpoint"])
@pytest.mark.parametrize("norm_scale", [1.0, 3.0])
def test_flow_matching(solver, norm_scale):
    params = {
        "in_dim": 20,
        "out_dim": 5,
        "denoise_fn": _diffnet(5, 8),
        "encoder": _encoder(20, 8),
        "num_timesteps": 1000,
        "K_step": 4,
        "norm_scale": norm_scale,
        "solver": solver,
    }
    model = FlowMatching(**params)
    assert model.prediction_type() == PredictionType.DIFFUSION
    assert not model.is_autoregressive()
    assert not model.has_residual_lf0_prediction()

    B, T = 4, 100
    init_seed(B * T)
    x = torch.rand(B, T, model.in_dim)
    y = torch.rand(B, T, model.out_dim)
    lengths = torch.Tensor([T] * B).long()

    u, v_pred = model(x, lengths, y)
    assert u.shape == y.shape
    assert v_pred.shape == y.shape

    y_hat = model.inference(x, lengths)
    assert y_hat.shape == y.shape
    assert torch.isfinite(y_hat).all()


def test_flow_matching_without_encoder():
    encoder = _encoder(20, 8)
    model = FlowMatching(
        in_dim=20, out_dim=5, denoise_fn=_diffnet(5, 8), encoder=None, K_step=4
    )
    B, T = 4, 100
    init_seed(B * T)
    x = torch.rand(B, T, 20)
    y = torch.rand(B, T, 5)
    lengths = torch.Tensor([T] * B).long()

    encoder_outs = encoder(x, lengths)
    u, v_pred = model(encoder_outs, lengths, y)
    assert u.shape == y.shape
    assert v_pred.shape == y.shape
    assert model.inference(encoder_outs, lengths).shape == y.shape


def test_flow_matching_target_velocity():
    """The target velocity must be x1 - x0 and hence have unit-ish statistics"""
    model = FlowMatching(
        in_dim=8, out_dim=2, denoise_fn=_diffnet(2, 8), encoder=None, K_step=2
    )
    B, T = 4, 100
    init_seed(B * T)
    cond = torch.rand(B, T, 8)
    y = torch.zeros(B, T, 2)
    # with y = 0 (i.e. x1 = 0), the target velocity is -x0 ~ N(0, 1)
    u, _ = model(cond, torch.Tensor([T] * B).long(), y)
    assert abs(u.mean().item()) < 0.1
    assert abs(u.std().item() - 1.0) < 0.1


@pytest.mark.parametrize("solver", ["euler", "midpoint"])
@pytest.mark.parametrize("predict_residual", [True, False])
def test_flow_matching_f0(solver, predict_residual):
    in_dim, in_lf0_idx = 10, 3
    params = {
        "in_dim": in_dim,
        "out_dim": 1,
        "denoise_fn": _diffnet(1, 8),
        "encoder": _encoder(in_dim, 8),
        "K_step": 4,
        "solver": solver,
        "in_lf0_idx": in_lf0_idx,
        "in_lf0_min": IN_LF0_MIN,
        "in_lf0_max": IN_LF0_MAX,
        "out_lf0_idx": 0,
        "out_lf0_mean": OUT_LF0_MEAN,
        "out_lf0_scale": OUT_LF0_SCALE,
        "predict_residual": predict_residual,
        "residual_scale_cent": 600,
        "clip_cent": 600,
    }
    model = FlowMatchingF0(**params)
    assert model.prediction_type() == PredictionType.DIFFUSION
    assert model.has_residual_lf0_prediction()

    B, T = 4, 100
    init_seed(B * T)
    x = torch.rand(B, T, in_dim)
    y = torch.randn(B, T, 1)
    lengths = torch.Tensor([T] * B).long()

    (u, v_pred), lf0_residual = model(x, lengths, y)
    assert u.shape == y.shape
    assert v_pred.shape == y.shape
    assert lf0_residual.shape == y.shape

    lf0 = model.inference(x, lengths)
    assert lf0.shape == y.shape
    assert torch.isfinite(lf0).all()


def test_flow_matching_f0_residual_roundtrip():
    """Feeding the target as the ODE endpoint must reproduce the target"""
    in_dim, in_lf0_idx = 10, 3
    model = FlowMatchingF0(
        in_dim=in_dim,
        denoise_fn=_diffnet(1, 8),
        encoder=_encoder(in_dim, 8),
        K_step=2,
        in_lf0_idx=in_lf0_idx,
        in_lf0_min=IN_LF0_MIN,
        in_lf0_max=IN_LF0_MAX,
        out_lf0_mean=OUT_LF0_MEAN,
        out_lf0_scale=OUT_LF0_SCALE,
    )
    B, T = 2, 30
    init_seed(B * T)
    x = torch.rand(B, T, in_dim)
    y = torch.randn(B, T, 1) * 0.5

    lf0_score_denorm = model._lf0_score_denorm(x)
    lf0_target_denorm = y * OUT_LF0_SCALE + OUT_LF0_MEAN
    # the normalized target of the flow
    y_flow = (lf0_target_denorm - lf0_score_denorm) / model.residual_scale
    # the inverse mapping applied in inference()
    lf0 = model._to_normalized_lf0(lf0_score_denorm + y_flow * model.residual_scale)
    assert torch.allclose(lf0, y, atol=1e-5)


@pytest.mark.parametrize("predict_residual", [True, False])
def test_flow_matching_f0_endpoint_clip(predict_residual):
    """The generated log-F0 must be within clip_cent cent of the score

    NOTE: the Euler solver's last step exactly lands on the clipped endpoint
    estimate, so the bound holds exactly.
    """
    in_dim, in_lf0_idx = 10, 3
    clip_cent = 300
    model = FlowMatchingF0(
        in_dim=in_dim,
        denoise_fn=_diffnet(1, 8),
        encoder=_encoder(in_dim, 8),
        K_step=8,
        solver="euler",
        in_lf0_idx=in_lf0_idx,
        in_lf0_min=IN_LF0_MIN,
        in_lf0_max=IN_LF0_MAX,
        out_lf0_mean=OUT_LF0_MEAN,
        out_lf0_scale=OUT_LF0_SCALE,
        predict_residual=predict_residual,
        residual_scale_cent=600,
        clip_cent=clip_cent,
    )
    B, T = 4, 100
    init_seed(B * T)
    x = torch.rand(B, T, in_dim)
    lengths = torch.Tensor([T] * B).long()

    lf0 = model.inference(x, lengths)
    lf0_score_denorm = model._lf0_score_denorm(x)
    lf0_denorm = lf0 * OUT_LF0_SCALE + OUT_LF0_MEAN
    dev_cent = (lf0_denorm - lf0_score_denorm).abs() * 1200 / np.log(2)
    assert dev_cent.max().item() <= clip_cent + 1e-2


def test_flow_matching_f0_no_clip():
    model = FlowMatchingF0(
        in_dim=10,
        denoise_fn=_diffnet(1, 8),
        encoder=_encoder(10, 8),
        K_step=4,
        in_lf0_idx=3,
        in_lf0_min=IN_LF0_MIN,
        in_lf0_max=IN_LF0_MAX,
        out_lf0_mean=OUT_LF0_MEAN,
        out_lf0_scale=OUT_LF0_SCALE,
        clip_cent=None,
    )
    B, T = 2, 30
    init_seed(B * T)
    x = torch.rand(B, T, 10)
    lengths = torch.Tensor([T] * B).long()
    assert model._compute_clip_bounds(model._lf0_score_denorm(x)) is None
    assert model.inference(x, lengths).shape == (B, T, 1)


def test_flow_matching_f0_invalid_out_dim():
    with pytest.raises(AssertionError):
        FlowMatchingF0(in_dim=10, out_dim=2, denoise_fn=_diffnet(2, 8))


def test_flow_matching_invalid_solver():
    with pytest.raises(ValueError):
        FlowMatching(in_dim=8, out_dim=2, denoise_fn=_diffnet(2, 8), solver="rk4")


@pytest.mark.parametrize("reduction_factor", [1, 4])
def test_flow_multistream_parametric_model(reduction_factor):
    """All-flow-matching version of the v5 multi-stream acoustic model"""
    in_dim = 10
    in_lf0_idx = 3
    stream_sizes = [4, 1, 1, 2]
    out_dim = sum(stream_sizes)
    # (x, lf0, mgc)
    vuv_in_dim = in_dim + 1 + 4

    params = {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "stream_sizes": stream_sizes,
        "reduction_factor": reduction_factor,
        "lf0_model": FlowMatchingF0(
            in_dim=in_dim,
            out_dim=1,
            denoise_fn=_diffnet(1, 8),
            encoder=_encoder(in_dim, 8),
            K_step=4,
            in_lf0_idx=in_lf0_idx,
        ),
        "mgc_model": FlowMatching(
            in_dim=in_dim + 1,
            out_dim=4,
            denoise_fn=_diffnet(4, 8),
            encoder=_encoder(in_dim + 1, 8),
            K_step=4,
        ),
        "bap_model": FlowMatching(
            in_dim=in_dim + 1,
            out_dim=2,
            denoise_fn=_diffnet(2, 8),
            encoder=_encoder(in_dim + 1, 8),
            K_step=4,
        ),
        "vuv_model": FFN(in_dim=vuv_in_dim, hidden_dim=5, out_dim=1),
        "vuv_model_bap_conditioning": False,
        "vuv_model_bap0_conditioning": False,
        "vuv_model_lf0_conditioning": True,
        "vuv_model_mgc_conditioning": True,
        "in_rest_idx": 0,
        "in_lf0_idx": in_lf0_idx,
        "in_lf0_min": IN_LF0_MIN,
        "in_lf0_max": IN_LF0_MAX,
        "out_lf0_idx": 4,
        "out_lf0_mean": OUT_LF0_MEAN,
        "out_lf0_scale": OUT_LF0_SCALE,
    }
    model = NPSSMDNMultistreamParametricModel(**params)
    assert model.prediction_type() == PredictionType.MULTISTREAM_HYBRID
    assert model.has_residual_lf0_prediction()
    assert not model.is_autoregressive()

    B, T = 4, 100
    init_seed(B * T)
    x = torch.rand(B, T, in_dim)
    y = torch.rand(B, T, out_dim)
    lengths = torch.Tensor([T] * B).long()

    # Training: mgc/lf0/bap are two-element tuples (target, prediction),
    # vuv is a plain tensor
    (mgc, lf0, vuv, bap), lf0_residual = model(x, lengths, y)
    for stream, size in [(mgc, 4), (lf0, 1), (bap, 2)]:
        assert isinstance(stream, tuple) and len(stream) == 2
        assert stream[0].shape == (B, T, size)
        assert stream[1].shape == (B, T, size)
    assert vuv.shape == (B, T, 1)
    assert lf0_residual.shape == (B, T, 1)

    # log-F0 statistics must be propagated from the parent model
    assert model.lf0_model.in_lf0_min == IN_LF0_MIN
    assert model.lf0_model.out_lf0_mean == OUT_LF0_MEAN

    # Inference: (mu, sigma) of the concatenated features
    mu, sigma = model.inference(x, lengths)
    assert mu.shape == (B, T, out_dim)
    assert sigma.shape == (B, T, out_dim)
    assert torch.isfinite(mu).all()
