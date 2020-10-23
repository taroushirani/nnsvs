# coding: utf-8

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import numpy as np
import joblib
from tqdm import tqdm
from os.path import basename, splitext, exists, join
import os
import torch
from torch import nn
from torch.nn import functional as F
from nnmnkwii.datasets import FileSourceDataset

from nnsvs.gen import get_windows, predict
from nnsvs.multistream import multi_stream_mlpg
from nnsvs.bin.train import NpyFileSource
from nnsvs.logger import getLogger
from nnsvs.mdn import mdn_get_most_probable_sigma_and_mu, mdn_get_sample

logger = None

use_cuda = torch.cuda.is_available()

def generate(config, models, device, in_feats, scaler, out_dir):
    with torch.no_grad():
        for idx in tqdm(range(len(in_feats))):

            means = []
            vars = []
            if config.stream_wise_training:
                # stream-wise trained model

                # Straem indices for static+delta features
                # [0,   180, 183, 184]
                stream_start_indices = np.hstack(([0], np.cumsum(config.stream_sizes)[:-1]))
                # [180, 183, 184, 199]
                stream_end_indices = np.cumsum(config.stream_sizes)
                
                for stream_id in range(len(config.stream_sizes)):
                    mean, var = predict(config, models[stream_id], device, in_feats[idx], scaler,
                                        stream_start_indices[stream_id], stream_end_indices[stream_id])
                    means.append(mean)
                    vars.append(var)
            else:
                mean, var = predict(config, models[0], device, in_feats[idx], scaler)
                
                means.append(mean)
                vars.append(var)
                
            means = np.concatenate(means, -1)
            vars = np.concatenate(means, -1)
            print(means.shape)
            print(vars.shape)
            
            # Apply MLPG if necessary
            if np.any(config.has_dynamic_features):
                out = multi_stream_mlpg(
                    means, vars, get_windows(config.num_windows),
                    config.stream_sizes, config.has_dynamic_features)
            else:
                out = means

            name = basename(in_feats.collected_files[idx][0])
            out_path = join(out_dir, name)
            np.save(out_path, out, allow_pickle=False)

def resume(config, device, checkpoint, stream_id=None):
    if stream_id is not None:
        assert len(config.stream_sizes) == len(checkpoint)
        model = hydra.utils.instantiate(config.models[stream_id].netG).to(device)
        cp = torch.load(to_absolute_path(checkpoint[stream_id]),
                        map_location=lambda storage, loc: storage)
    else:
        model = hydra.utils.instantiate(config.netG).to(device)
        cp = torch.load(to_absolute_path(checkpoint),
                        map_location=lambda storage, loc: storage)

    model.load_state_dict(cp["state_dict"])

    return model

@hydra.main(config_path="conf/generate/config.yaml")
def my_app(config : DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(config.pretty())

    device = torch.device("cuda" if use_cuda else "cpu")
    in_dir = to_absolute_path(config.in_dir)
    out_dir = to_absolute_path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model_config = OmegaConf.load(to_absolute_path(config.model.model_yaml))

    scaler = joblib.load(to_absolute_path(config.out_scaler_path))
    in_feats = FileSourceDataset(NpyFileSource(in_dir))

    models = []
    if model_config.stream_wise_training:
        assert len(model_config.models) == len(model_config.stream_sizes)
        assert len(config.model.checkpoint) == len(model_config.stream_sizes)
        
        for stream_id in range(len(model_config.stream_sizes)):
            models.append(resume(model_config, device, config.model.checkpoint, stream_id))
    else:
        models.append(resume(model_config, device, config.model.checkpoint, None))
        
    generate(model_config, models, device, in_feats, scaler, out_dir)
            
def entry():
    my_app()


if __name__ == "__main__":
    my_app()
