"""
model.py — Neural network for exoplanet atmospheric retrieval.

Architecture: TransitCNN
  - TemporalEncoder: 1D CNN along the time axis, separately for AIRS-CH0 and FGS1
  - MLP fusion block combining both encodings with auxiliary stellar/planet features
  - Two output heads: mean (283,) and log_variance (283,) per output wavelength

Loss: gaussian_nll_loss
  GLL(y, mu, log_var) = 0.5 * (log_var + (y - mu)^2 / exp(log_var))
  This trains the model to predict calibrated uncertainty alongside the mean.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    1D CNN encoder that collapses a multi-channel time series into a fixed-size embedding.

    Input shape : (batch, in_channels, time)
    Output shape: (batch, embed_dim)

    Three Conv1d layers with GELU activations followed by global average pooling.
    Kernel sizes decrease (5 → 5 → 3) to capture multi-scale temporal patterns.

    Parameters
    ----------
    in_channels : number of input channels (356 for AIRS, 1 for FGS1)
    embed_dim   : size of the output embedding vector
    dropout     : dropout probability applied after the last conv layer
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        mid = max(64, embed_dim // 2)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, mid, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(mid, mid * 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(mid * 2, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pooling over time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)       # (batch, embed_dim, time)
        x = self.pool(x)       # (batch, embed_dim, 1)
        return x.squeeze(-1)   # (batch, embed_dim)


class TransitCNN(nn.Module):
    """
    Full retrieval model for exoplanet transmission spectra.

    Takes raw (or preprocessed) light curves and stellar/planetary parameters,
    and outputs a probabilistic prediction of the transmission spectrum:
    one mean and one log-variance per output wavelength bin.

    Parameters
    ----------
    n_airs_channels : number of AIRS-CH0 wavelength channels  (default 356)
    n_aux           : number of auxiliary stellar/planet features (default 9)
    n_output_wl     : number of output wavelength bins           (default 283)
    embed_dim       : internal AIRS encoder embedding dimension  (default 128)
    dropout         : dropout probability used throughout        (default 0.1)
    """

    def __init__(
        self,
        n_airs_channels: int = 356,
        n_aux: int = 9,
        n_output_wl: int = 283,
        embed_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_output_wl = n_output_wl

        fgs1_dim = max(32, embed_dim // 4)
        self.airs_encoder = TemporalEncoder(n_airs_channels, embed_dim, dropout)
        self.fgs1_encoder = TemporalEncoder(1, fgs1_dim, dropout)

        fusion_in = embed_dim + fgs1_dim + n_aux
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 512),       nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),       nn.GELU(),
        )
        self.head_mean    = nn.Linear(256, n_output_wl)
        self.head_log_var = nn.Linear(256, n_output_wl)

    def forward(
        self,
        airs: torch.Tensor,  # (batch, n_airs_channels, time)
        fgs1: torch.Tensor,  # (batch, 1, time)
        aux:  torch.Tensor,  # (batch, n_aux)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        airs : (batch, n_airs_channels, time) — channel-first AIRS light curves
        fgs1 : (batch, 1, time)               — FGS1 light curve
        aux  : (batch, n_aux)                 — stellar/planetary parameters

        Returns
        -------
        mean    : (batch, n_output_wl) — predicted transmission spectrum
        log_var : (batch, n_output_wl) — log-variance of the prediction
        """
        z_airs = self.airs_encoder(airs)   # (batch, embed_dim)
        z_fgs1 = self.fgs1_encoder(fgs1)  # (batch, fgs1_dim)
        z = torch.cat([z_airs, z_fgs1, aux], dim=-1)
        z = self.fusion(z)
        return self.head_mean(z), self.head_log_var(z)


def gaussian_nll_loss(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss, averaged over all elements.

    NLL = 0.5 * (log_var + (target - mean)^2 / exp(log_var))

    Training with this loss encourages the model to minimise both prediction
    error (squared residual term) and uncertainty calibration (log_var term).
    Overconfident predictions (small var, large residual) are heavily penalised.

    Parameters
    ----------
    mean    : (batch, n_wl) — predicted mean
    log_var : (batch, n_wl) — predicted log-variance (unconstrained)
    target  : (batch, n_wl) — ground-truth transmission spectrum

    Returns
    -------
    scalar loss tensor
    """
    var = torch.exp(log_var)
    return 0.5 * (log_var + (target - mean).pow(2) / var).mean()
