import torch
from torch import nn


class WavehaxWrapper(nn.Module):
    def __init__(self, config, generator):
        super().__init__()
        self.generator = generator
        self.config = config

    def inference(self, f0, aux_feats):
        """Inference for Wavehax

        Unlike USFGANWrapper, Wavehax needs no external excitation signal
        generator and no aux_context_window padding: the generator's own
        inference() synthesizes the harmonic prior internally from f0.

        Args:
            f0 (numpy.ndarray): F0 (T, 1)
            aux_feats (Tensor): Auxiliary features (T, C)

        """
        device = aux_feats.device

        cond = aux_feats.unsqueeze(0).transpose(2, 1).to(device)
        f0 = torch.FloatTensor(f0).unsqueeze(0).transpose(2, 1).to(device)

        return self.generator.inference(cond, f0)
