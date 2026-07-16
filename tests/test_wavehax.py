import numpy as np
import pyworld
import pytest
import torch
from hydra.utils import instantiate
from nnsvs.gen import predict_waveform
from nnsvs.util import load_vocoder
from nnsvs.wavehax import WavehaxWrapper
from omegaconf import OmegaConf

pytest.importorskip("wavehax")

# Small generator size for fast, synthetic-data-only tests.
# There is no official Wavehax checkpoint/recipe for singing voice synthesis,
# so we can only verify the nnsvs-side wiring with a randomly initialized model.
GENERATOR_CFG = {
    "channels": 8,
    "mult_channels": 2,
    "kernel_size": 3,
    "num_blocks": 2,
    "n_fft": 384,
    "hop_length": 240,
    "sample_rate": 48000,
    "prior_type": "pcph_closed_form",
}
SAMPLE_RATE = 48000
HOP_LENGTH = 240
NUM_FRAMES = 10


def _make_acoustic_config(stream_sizes):
    return OmegaConf.create(
        {
            "stream_sizes": stream_sizes,
            "has_dynamic_features": [False] * len(stream_sizes),
            "num_windows": 1,
        }
    )


def _make_vocoder_config(in_channels, feat_names):
    return OmegaConf.create(
        {
            "generator": {
                "_target_": "wavehax.generators.WavehaxGenerator",
                "in_channels": in_channels,
                **GENERATOR_CFG,
            },
            "discriminator": {},
            "data": {"feat_names": feat_names},
        }
    )


def _save_checkpoint(vocoder_config, path):
    generator = instantiate(vocoder_config.generator)
    torch.save(
        {"model": {"generator": generator.state_dict()}, "steps": 0, "epochs": 0},
        path,
    )


def _save_dummy_scalers(model_dir, dim):
    np.save(model_dir / "in_vocoder_scaler_mean.npy", np.zeros(dim, dtype=np.float32))
    np.save(model_dir / "in_vocoder_scaler_var.npy", np.ones(dim, dtype=np.float32))
    np.save(model_dir / "in_vocoder_scaler_scale.npy", np.ones(dim, dtype=np.float32))


def test_load_vocoder_and_predict_waveform_melf0(tmp_path):
    mel_dim = 4
    stream_sizes = [mel_dim, 1, 1]  # mel, lf0, vuv
    acoustic_config = _make_acoustic_config(stream_sizes)

    vocoder_config = _make_vocoder_config(in_channels=mel_dim, feat_names=["mel"])
    model_dir = tmp_path
    OmegaConf.save(vocoder_config, model_dir / "vocoder_model.yaml")
    ckpt_path = model_dir / "checkpoint.pkl"
    _save_checkpoint(vocoder_config, ckpt_path)
    _save_dummy_scalers(model_dir, sum(stream_sizes))

    vocoder, vocoder_in_scaler, loaded_vocoder_config = load_vocoder(
        ckpt_path, "cpu", acoustic_config
    )
    assert isinstance(vocoder, WavehaxWrapper)

    mel = np.random.randn(NUM_FRAMES, mel_dim).astype(np.float32)
    lf0 = np.log(200 * np.ones((NUM_FRAMES, 1), dtype=np.float32))
    vuv = np.ones((NUM_FRAMES, 1), dtype=np.float32)

    wav = predict_waveform(
        "cpu",
        (mel, lf0, vuv),
        vocoder=vocoder,
        vocoder_config=loaded_vocoder_config,
        vocoder_in_scaler=vocoder_in_scaler,
        sample_rate=SAMPLE_RATE,
        feature_type="melf0",
        vocoder_type="wavehax",
    )
    assert wav.shape[0] == NUM_FRAMES * HOP_LENGTH
    assert np.isfinite(wav).all()


def test_load_vocoder_and_predict_waveform_world(tmp_path):
    mgc_dim = 4
    fftlen = pyworld.get_cheaptrick_fft_size(SAMPLE_RATE)
    ap = 0.4 * np.ones((NUM_FRAMES, fftlen // 2 + 1))
    bap = pyworld.code_aperiodicity(ap, SAMPLE_RATE).astype(np.float32)
    bap_dim = bap.shape[-1]

    stream_sizes = [mgc_dim, 1, 1, bap_dim]  # mgc, lf0, vuv, bap
    acoustic_config = _make_acoustic_config(stream_sizes)

    vocoder_config = _make_vocoder_config(
        in_channels=mgc_dim + bap_dim, feat_names=["mcep", "codeap"]
    )
    model_dir = tmp_path
    OmegaConf.save(vocoder_config, model_dir / "vocoder_model.yaml")
    ckpt_path = model_dir / "checkpoint.pkl"
    _save_checkpoint(vocoder_config, ckpt_path)
    _save_dummy_scalers(model_dir, sum(stream_sizes))

    vocoder, vocoder_in_scaler, loaded_vocoder_config = load_vocoder(
        ckpt_path, "cpu", acoustic_config
    )
    assert isinstance(vocoder, WavehaxWrapper)

    mgc = np.random.randn(NUM_FRAMES, mgc_dim).astype(np.float32)
    lf0 = np.log(200 * np.ones((NUM_FRAMES, 1), dtype=np.float32))
    vuv = np.ones((NUM_FRAMES, 1), dtype=np.float32)

    wav = predict_waveform(
        "cpu",
        (mgc, lf0, vuv, bap),
        vocoder=vocoder,
        vocoder_config=loaded_vocoder_config,
        vocoder_in_scaler=vocoder_in_scaler,
        sample_rate=SAMPLE_RATE,
        feature_type="world",
        vocoder_type="wavehax",
    )
    assert wav.shape[0] == NUM_FRAMES * HOP_LENGTH
    assert np.isfinite(wav).all()
