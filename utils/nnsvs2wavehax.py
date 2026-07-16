"""Convert NNSVS's pre-processed features to Wavehax-friendly format
"""
import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pyworld
from nnsvs.util import StandardScaler
from omegaconf import OmegaConf
from scipy.io import wavfile
from tqdm.auto import tqdm
from wavehax.utils import write_hdf5


def split_streams(inputs, stream_sizes=None):
    """Split streams from multi-stream features

    Args:
        inputs (array like): input 3-d array
        stream_sizes (list): sizes for each stream

    Returns:
        list: list of stream features
    """
    if stream_sizes is None:
        stream_sizes = [60, 1, 1, 1]
    ret = []
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    for start_idx, size in zip(start_indices, stream_sizes):
        if len(inputs.shape) == 3:
            s = inputs[:, :, start_idx : start_idx + size]
        else:
            s = inputs[:, start_idx : start_idx + size]
        ret.append(s)

    return ret


def load_utt_list(utt_list):
    """Load a list of utterances.

    Args:
        utt_list (str): path to a file containing a list of utterances

    Returns:
        List[str]: list of utterances
    """
    with open(utt_list) as f:
        utt_ids = f.readlines()
    utt_ids = map(lambda utt_id: utt_id.strip(), utt_ids)
    utt_ids = filter(lambda utt_id: len(utt_id) > 0, utt_ids)
    return list(utt_ids)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert NNSVS's pre-processed features to Wavehax's format",
    )
    parser.add_argument("config_path", type=str, help="Path to recipe config")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument(
        "--feature_type", type=str, default="world", help="Feature type"
    )
    parser.add_argument(
        "--relative_path", action="store_true", help="Use relative path"
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    recipe_config_path = Path(args.config_path)
    assert recipe_config_path.exists()
    os.chdir(recipe_config_path.parent)
    recipe_config = OmegaConf.load(recipe_config_path)

    spk = recipe_config.spk
    sample_rate = recipe_config.sample_rate

    # NOTE: used for de-normalization
    scaler = joblib.load(f"dump/{spk}/norm/out_acoustic_scaler.joblib")

    # Save scaler for Wavehax
    out_stats_dir = Path(f"{args.out_dir}/stats")
    out_stats_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: mgc order is fixed regardless of sampling rate
    if args.feature_type == "world":
        if len(scaler.mean_) > 65:
            # (mgc, lf0, vuv, mcap)
            stream_sizes = [60, 1, 1, len(scaler.mean_) - 62]
        else:
            # (mgc, lf0, vuv, bap)
            stream_sizes = [60, 1, 1, pyworld.get_num_aperiodicities(sample_rate)]
        assert len(scaler.mean_.reshape(-1)) == sum(stream_sizes)

        # NOTE: unlike uSFGAN, wavehax.datasets.AudioFeatDataset only looks up
        # `scaler[feat_name]` for names in `feat_names` -- F0 itself is read
        # directly from the "f0"/"cf0" HDF5 key and is never scaler-transformed,
        # so no F0 entry is needed here.
        wavehax_scaler = {
            "mcep": StandardScaler(
                scaler.mean_[0:60], scaler.var_[0:60], scaler.scale_[0:60]
            ),
            # NOTE: pretend mcap as codeap if mcep-based aperiodicity is used
            "codeap": StandardScaler(
                scaler.mean_[62:], scaler.var_[62:], scaler.scale_[62:]
            ),
        }
        feat_names = ["mcep", "codeap"]
    elif args.feature_type == "melf0":
        stream_sizes = [len(scaler.mean_) - 2, 1, 1]
        mel_dim = stream_sizes[0]
        wavehax_scaler = {
            "mel": StandardScaler(
                scaler.mean_[0:mel_dim],
                scaler.var_[0:mel_dim],
                scaler.scale_[0:mel_dim],
            ),
        }
        feat_names = ["mel"]
    else:
        raise ValueError(f"Unknown feature type: {args.feature_type}")

    joblib.dump(wavehax_scaler, out_stats_dir / "scaler.joblib")

    hop_size = -1
    aux_channels = -1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    for s in ["dev", "eval", "train_no_dev"]:
        list_file = Path(f"data/list/{s}.list")
        assert list_file.exists()
        utt_ids = load_utt_list(list_file)
        dump_norm_dir = Path(f"dump/{spk}/norm/{s}/in_vocoder")

        # Output directories
        out_scp_dir = out_dir / "scp"
        out_hdf5_dir = out_dir / "hdf5"
        out_wav_dir = out_dir / "wav"
        for d in [out_scp_dir, out_hdf5_dir, out_wav_dir]:
            d.mkdir(exist_ok=True, parents=True)

        for utt_id in tqdm(utt_ids, desc=s):
            wave = np.load(dump_norm_dir / f"{utt_id}-wave.npy")
            feats = np.load(dump_norm_dir / f"{utt_id}-feats.npy")

            feats = scaler.inverse_transform(feats)
            if args.feature_type == "world":
                mgc, lf0, vuv, bap = split_streams(feats, stream_sizes)
                aux_channels = mgc.shape[-1] + bap.shape[-1]
            elif args.feature_type == "melf0":
                mel, lf0, vuv = split_streams(feats, stream_sizes)
                aux_channels = mel.shape[1]

            vuv = (vuv > 0.5).astype(np.float32)
            assert len(lf0.shape) == 2
            assert len(vuv.shape) == 2

            # NOTE: wave and feats are already time-aligned by NNSVS's pre-processing
            hop_size = len(wave) // len(feats)

            # Wavehax's AudioFeatDataset reads F0 from either the "f0" (gated
            # to zero for unvoiced frames) or "cf0" (continuous) HDF5 key,
            # depending on the data config's `use_continuous_f0` flag. There is
            # no separate V/UV key.
            contf0 = np.exp(lf0)
            f0 = contf0.copy()
            f0[vuv < 0.5] = 0

            wavehax_feat_path = out_hdf5_dir / f"{utt_id}.h5"
            write_hdf5(wavehax_feat_path, "/f0", f0)
            write_hdf5(wavehax_feat_path, "/cf0", contf0)

            if args.feature_type == "world":
                write_hdf5(wavehax_feat_path, "/mcep", mgc)
                write_hdf5(wavehax_feat_path, "/codeap", bap)
            elif args.feature_type == "melf0":
                write_hdf5(wavehax_feat_path, "/mel", mel)

            # NNSVSの前処理で波形はfloat32に変換されているが、Wavehaxは生の波形を
            # 読み込む想定なので、そのままwavファイルとして書き出す
            wavfile.write(out_wav_dir / f"{utt_id}.wav", sample_rate, wave.reshape(-1))

        # Write scp/list files for Wavehax
        # NOTE: scp: 波形のパス, list: 特徴量のパス
        with open(out_scp_dir / f"{spk}_sr{sample_rate}_{s}.scp", "w") as f:
            for utt_id in utt_ids:
                wav_path = out_wav_dir / f"{utt_id}.wav"
                assert wav_path.exists()
                if args.relative_path:
                    f.write(f"{wav_path}\n")
                else:
                    f.write(f"{wav_path.resolve()}\n")

        with open(out_scp_dir / f"{spk}_sr{sample_rate}_{s}.list", "w") as f:
            for utt_id in utt_ids:
                feat_path = out_hdf5_dir / f"{utt_id}.h5"
                assert feat_path.exists()
                if args.relative_path:
                    f.write(f"{feat_path}\n")
                else:
                    f.write(f"{feat_path.resolve()}\n")

    # Wavehaxの学習にあるとよい情報
    print(
        f"""stream_sizes: {stream_sizes}
hop_size: {hop_size}
sample_rate: {sample_rate}
aux_channels: {aux_channels}
feat_names: {feat_names}
use_continuous_f0: false"""
    )
