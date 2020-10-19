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
from nnsvs.mdn import mdn_get_most_probable_sigma_and_mu, mdn_get_sample

logger = None

use_cuda = torch.cuda.is_available()


def predict(config, model, device, feats, scaler):
    if model.prediction_type == "probabilistic":
    pi, sigma, mu = model(feats, [feats.shape[1]])
                
    if np.any(model_config.has_dynamic_features):
        max_sigma, max_mu = mdn_get_most_probable_sigma_and_mu(pi, sigma, mu)
                   
        # Apply denormalization
        # (B, T, D_out) -> (T, D_out)
        max_sigma = max_sigma.squeeze(0).cpu().data.numpy() * scaler.var_
        max_mu = scaler.inverse_transform(max_mu.squeeze(0).cpu().data.numpy())
        # Apply MLPG
        # (T, D_out) -> (T, static_dim)
        out = multi_stream_mlpg(max_mu, max_sigma, get_windows(model_config.num_windows),
                                model_config.stream_sizes, model_config.has_dynamic_features)

                else:
                    # (T, D_out)
                    out = mdn_get_sample(pi, sigma, mu).squeeze(0).cpu().data.numpy()
                    out = scaler.inverse_transform(out)
            else:
                out = model(feats, [feats.shape[1]]).squeeze(0).cpu().data.numpy()
                out = scaler.inverse_transform(out)

                # Apply MLPG if necessary
                if np.any(model_config.has_dynamic_features):
                    out = multi_stream_mlpg(
                        out, scaler.var_, get_windows(model_config.num_windows),
                        model_config.stream_sizes, model_config.has_dynamic_features)
    

def generate(config, models, device, in_feats, scaler, out_dir):
    with torch.no_grad():
        for idx in tqdm(range(len(in_feats))):
            feats = torch.from_numpy(in_feats[idx]).unsqueeze(0).to(device)

            if config.stream_wise_training and \
               type(model) is list and \
               len(model) == len(config.stream_sizes):
                # stream-wise trained model
                out = []
                for stream_id in range(len(config.stream_sizes)):
                    out.append(model[stream_id](feats, [feats.shape[1]]).squeeze(0).cpu().data.numpy())
                out = np.concatenate(out, -1)
            else:
                out = model(feats, [feats.shape[1]]).squeeze(0).cpu().data.numpy()
            
            out = scaler.inverse_transform(out)

            # Apply MLPG if necessary
            print(f"config.has_dynamic_features: {config.has_dynamic_features}")
            if np.any(config.has_dynamic_features):
                windows = get_windows(3)
                out = multi_stream_mlpg(
                    out, scaler.var_, windows, config.stream_sizes,
                    config.has_dynamic_features)

            name = basename(in_feats.collected_files[idx][0])
            out_path = join(out_dir, name)
            np.save(out_path, out, allow_pickle=False)
            


            name = basename(in_feats.collected_files[idx][0])
            out_path = join(out_dir, name)
            np.save(out_path, out, allow_pickle=False)

    



def resume(config, device, checkpoint, stream_id=None):
    if stream_id is not None and\
       len(config.stream_sizes) == len(checkpoint):
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
    if model_config.stream_wise_training and \
       len(model_config.models) == len(model_config.stream_sizes) and \
       len(config.model.checkpoint) == len(model_config.stream_sizes):
        for stream_id in range(len(model_config.stream_sizes)):
            models.append(resume(model_config, device, config.model.checkpoint, stream_id))
    else:
        models.append(resume(model_config, device, config.model.checkpoint, None))
        
    generate(model_config, models, device, in_feats, scaler, out_dir)
            
def entry():
    my_app()


if __name__ == "__main__":
    my_app()
