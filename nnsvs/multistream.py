
# coding: utf-8

# Utils for multi-stream features

import torch
import numpy as np

from nnmnkwii import paramgen

def select_streams(inputs, stream_sizes=[60, 1, 1, 1],
                   streams=[True, True, True, True]):
    ret = []
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    for start_idx, size, enabled in zip(
            start_indices, stream_sizes, streams):
        if not enabled:
            continue
        if len(inputs.shape) == 3:
            s = inputs[:, :, start_idx:start_idx + size]
        elif len(inputs.shape) == 4:
            # MDN(B, T, num_gaussians, C)
            s = inputs[:, :, :, start_idx:start_idx + size]
        else:
            s = inputs[:, start_idx:start_idx + size]
        ret.append(s)

    if isinstance(inputs, torch.Tensor):
        return torch.cat(ret, dim=-1)
    else:
        return np.concatenate(ret, -1)


def split_streams(inputs, stream_sizes=[60, 1, 1, 1]):
    ret = []
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    for start_idx, size in zip(
            start_indices, stream_sizes):
        if len(inputs.shape) == 3:
            s = inputs[:, :, start_idx:start_idx + size]
        elif len(inputs.shape) == 4:
            # MDN(B, T, num_gaussians, C)
            s = inputs[:, :, :, start_idx:start_idx + size]
        else:
            s = inputs[:, start_idx:start_idx + size]
        ret.append(s)

    return ret


def get_static_stream_sizes(stream_sizes, has_dynamic_features, num_windows):
    """Get static dimention for each feature stream.
    """
    static_stream_sizes = np.array(stream_sizes)
    static_stream_sizes[has_dynamic_features] = \
        static_stream_sizes[has_dynamic_features] / num_windows

    return static_stream_sizes


def get_static_features(inputs, num_windows, stream_sizes=[180, 3, 1, 15],
                        has_dynamic_features=[True, True, False, True],
                        streams=[True, True, True, True]):
    """Get static features from static+dynamic features.
    """
    _, _, D = inputs.shape
    if stream_sizes is None or (len(stream_sizes) == 1 and has_dynamic_features[0]):
        return inputs[:, :, :D // num_windows]
    if len(stream_sizes) == 1 and not has_dynamic_features[0]:
        return inputs

    # Multi stream case
    ret = []
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    for start_idx, size, v, enabled in zip(
            start_indices, stream_sizes, has_dynamic_features, streams):
        if not enabled:
            continue
        if v:
            static_features = inputs[:, :, start_idx:start_idx + size // num_windows]
        else:
            static_features = inputs[:, :, start_idx:start_idx + size]
        ret.append(static_features)
    return torch.cat(ret, dim=-1)


def multi_stream_mlpg(inputs, variances, windows,
                      stream_sizes=[180, 3, 1, 3],
                      has_dynamic_features=[True, True, False, True],
                      streams=[True, True, True, True]):
    """Split streams and do apply MLPG if stream has dynamic features.
    """
    T, D = inputs.shape
    if D != sum(stream_sizes):
        raise RuntimeError("You probably have specified wrong dimention params.")
    num_windows = len(windows)

    # Straem indices for static+delta features
    # [0,   180, 183, 184]
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    # [180, 183, 184, 199]
    end_indices = np.cumsum(stream_sizes)

    # Stream sizes for static features
    # [60, 1, 1, 5]
    static_stream_sizes = get_static_stream_sizes(
        stream_sizes, has_dynamic_features, num_windows)

    # [0,  60, 61, 62]
    static_stream_start_indices = np.hstack(
        ([0], np.cumsum(static_stream_sizes)[:-1]))
    # [60, 61, 62, 63]
    static_stream_end_indices = np.cumsum(static_stream_sizes)

    ret = []
    for in_start_idx, in_end_idx, out_start_idx, out_end_idx, v, enabled in zip(
            start_indices, end_indices, static_stream_start_indices,
            static_stream_end_indices, has_dynamic_features, streams):
        if not enabled:
            continue
        x = inputs[:, in_start_idx:in_end_idx]
        if inputs.shape == variances.shape:
            var_ = variances[:, in_start_idx:in_end_idx]
        else:
            var_ = np.tile(variances[in_start_idx:in_end_idx], (T, 1))
        y = paramgen.mlpg(x, var_, windows) if v else x
        ret.append(y)

    return np.concatenate(ret, -1)
