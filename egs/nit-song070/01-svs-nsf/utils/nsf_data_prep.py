# coding: utf-8
import argparse
import sys
from glob import glob
import os
from os.path import join, basename, splitext, exists, expanduser
import numpy as np
import librosa
import soundfile as sf
import joblib
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.preprocessing.f0 import interp1d
from nnsvs.multistream import multi_stream_mlpg, get_static_stream_sizes, split_streams

def _midi_to_hz(x, idx, log_f0=False):
    z = np.zeros(len(x))
    indices = x[:, idx] > 0
    z[indices] = librosa.midi_to_hz(x[indices, idx])
    if log_f0:
        z[indices] = np.log(z[indices])
    return z

def get_windows(num_window=1):
    windows = [(0, 0, np.array([1.0]))]
    if num_window >= 2:
        windows.append((1, 1, np.array([-0.5, 0.0, 0.5])))
    if num_window >= 3:
        windows.append((1, 1, np.array([1.0, -2.0, 1.0])))

    if num_window >= 4:
        raise ValueError(f"Not supported num windows: {num_window}")

    return windows

def get_parser():
    parser = argparse.ArgumentParser(
        description="Data preparation script for NSF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("in_dir", type=str, help="Input directory of the normalized song data and acoustic features")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("--test_set",   action='store_true', help="enable test_set data flag")
    parser.add_argument("--question_path", type=str, help="Path of .qst file", default="./conf/jp_qst001_nnsvs.hed")
    parser.add_argument("--relative_f0",   action='store_true')
    parser.add_argument("--stream_sizes",   type=list, default=[180,3,1,3])
    parser.add_argument("--has_dynamic_features", type=list, default=[True,True,False,True])
    parser.add_argument("--num_windows",   type=int, default=3)
    parser.add_argument("--acoustic_out_scaler", type=str, default="./dump/yoko/norm/out_acoustic_scaler.joblib")
    parser.add_argument("--sample_rate", type=int, default=16000)

    return parser


args = get_parser().parse_args(sys.argv[1:])

in_dir=args.in_dir
out_dir=args.out_dir
test_set=args.test_set

question_path=args.question_path
relative_f0=args.relative_f0
stream_sizes=args.stream_sizes
has_dynamic_features=args.has_dynamic_features
num_windows=args.num_windows
acoustic_out_scaler=joblib.load(args.acoustic_out_scaler)
sample_rate=args.sample_rate


binary_dict, continuous_dict = hts.load_question_set(question_path, append_hat_for_LL=False)
pitch_idx=len(binary_dict)+1


feats_files = sorted(glob(join(in_dir, "*-feats.npy")))
for feat_file in feats_files:
    base = splitext(basename(feat_file))[0].replace("-feats", "")
    label_path= join("data/acoustic/label_phone_align", base + ".lab")
    labels = hts.load(label_path)

    acoustic_features = np.load(feat_file)

    windows = get_windows(num_windows)

    if np.any(has_dynamic_features):
        acoustic_features = multi_stream_mlpg(
            acoustic_features, acoustic_out_scaler.var_, windows, stream_sizes,
            has_dynamic_features)
        static_stream_sizes = get_static_stream_sizes(
            stream_sizes, has_dynamic_features, len(windows))
    else:
        static_stream_sizes = stream_sizes
        
    mgc, target_f0, vuv, bap = split_streams(acoustic_features, static_stream_sizes)

    if relative_f0:
        diff_lf0 = target_f0
#        print(diff_lf0.max())
        # need to extract pitch sequence from the musical score
        linguistic_features = fe.linguistic_features(labels, binary_dict, continuous_dict,
                                                     add_frame_features=True,
                                                     subphone_features="coarse_coding")
        f0_score = _midi_to_hz(linguistic_features, pitch_idx, False)[:, None]
        lf0_score = f0_score.copy()
        nonzero_indices = np.nonzero(lf0_score)
        lf0_score[nonzero_indices] = np.log(f0_score[nonzero_indices])
#        lf0_score = interp1d(lf0_score, kind="slinear")
        f0 = diff_lf0 + lf0_score
        f0[vuv < 0.5] = 0
        f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])
    else:
        f0 = target_f0

#    output_path = join(out_dir, base + ".dlf0")
#    with open(output_path, "wb") as f:
#        np.save(f, diff_lf0)

#    output_path = join(out_dir, base + ".sf0")
#    with open(output_path, "wb") as f:
#        np.save(f, f0_score)

#    output_path = join(out_dir, base + ".slf0")
#    with open(output_path, "wb") as f:
#        np.save(f, lf0_score)
        
#    print(f0.shape)
#    print(f0.dtype)
    if test_set:
        feats_out_dir = join(out_dir , "test_input_dirs")
    else:
        feats_out_dir = join(out_dir , "input_dirs")
    if exists(feats_out_dir) != True:
        os.makedirs(feats_out_dir)
        
    f0_output_path = join(feats_out_dir, base + ".f0")
    with open(f0_output_path, "wb") as f:
        np.save(f, f0.astype(np.float32))
#    print(mgc.shape)
#    print(mgc.dtype)
    other_feats_output_path = join(feats_out_dir, base + ".npy")
    with open(other_feats_output_path, "wb") as f:
        np.save(f, np.hstack((mgc, vuv, bap)))
        
if test_set != True:          
    wave_files = sorted(glob(join(in_dir, "*-wave.npy")))
    for wave_file in wave_files:
        base = splitext(basename(wave_file))[0].replace("-wave", "")

        data = np.load(wave_file)
        wave_out_dir = join(out_dir, "output_dirs")
        if exists(wave_out_dir) != True:
            os.makedirs(wave_out_dir)
        
        wav_output_path = join(wave_out_dir, base + ".wav")
        sf.write(wav_output_path, data, sample_rate)
