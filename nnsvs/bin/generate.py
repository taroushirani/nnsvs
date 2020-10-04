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

from nnsvs.gen import get_windows
from nnsvs.multistream import multi_stream_mlpg
from nnsvs.bin.train import NpyFileSource
from nnsvs.logger import getLogger

logger = None

use_cuda = torch.cuda.is_available()

def generate(config, model, device, in_feats, scaler, out_dir, stream_id=None):
    with torch.no_grad():
        for idx in tqdm(range(len(in_feats))):
            feats = torch.from_numpy(in_feats[idx]).unsqueeze(0).to(device)
            out = model(feats, [feats.shape[1]]).squeeze(0).cpu().data.numpy()

            out = scaler.inverse_transform(out)

            # Apply MLPG if necessary
            if np.any(model_config.has_dynamic_features):
                windows = get_windows(3)
                out = multi_stream_mlpg(
                    out, scaler.var_, windows, model_config.stream_sizes,
                    model_config.has_dynamic_features)

            name = basename(in_feats.collected_files[idx][0])
            if stream_id is not None:
                out_path = join(out_dir, name + f"_stream_{stream_id}")
            else:
                out_path = join(out_dir, name)
            np.save(out_path, out, allow_pickle=False)


def resume(config, device, checkpoint, stream_id=None):
    if stream_id is not None and\
       len(config.model.stream_sizes) == len(checkpoint):
        model = hydra.utils.instantiate(config.models[stream_id].netG).to(device)
        checkpoint = torch.load(to_absolute_path(checkpoint[stream_id]),
                                map_location=lambda storage, loc: storage)
    else:
        model = hydra.utils.instantiate(config.netG).to(device)
        checkpoint = torch.load(to_absolute_path(checkpoint),
                                map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint["state_dict"])

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
        
    if model_config.model.stream_wise_training and \
       len(model_config.model.models) == len(config.model.stream_sizes) and \
       len(config.model.checkpoint) == len(config.model.stream_sizes):

        for stream_id in range(len(model_config.model.stream_sizes)):
            model = resume(model_config, device, config.model.checkpoint, stream_id)
            generate(config, model, device, in_feats, scaler, out_dir, stream_id)
    else:
        model = resume(model_config, device, config.model.checkpoint, None)
        generate(config, model, device, in_feats, scaler, out_dir, None)
            
def entry():
    my_app()


if __name__ == "__main__":
    my_app()
