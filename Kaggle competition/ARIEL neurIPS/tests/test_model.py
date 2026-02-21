"""Tests for src/model.py — TransitCNN forward pass and gaussian_nll_loss."""

import pytest
import torch

from src.model import TransitCNN, TemporalEncoder, gaussian_nll_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch(batch_size: int = 4, time: int = 100) -> dict:
    """Minimal random batch matching ArielDataset output shapes."""
    return {
        "airs": torch.randn(batch_size, 356, time),
        "fgs1": torch.randn(batch_size, 1,   time),
        "aux":  torch.randn(batch_size, 9),
    }


# ---------------------------------------------------------------------------
# TemporalEncoder
# ---------------------------------------------------------------------------

def test_temporal_encoder_output_shape():
    enc = TemporalEncoder(in_channels=356, embed_dim=128)
    x = torch.randn(4, 356, 100)
    out = enc(x)
    assert out.shape == (4, 128)


def test_temporal_encoder_variable_time():
    """Encoder must handle arbitrary time-axis lengths via adaptive pooling."""
    enc = TemporalEncoder(in_channels=1, embed_dim=32)
    for t in [50, 100, 300]:
        out = enc(torch.randn(2, 1, t))
        assert out.shape == (2, 32), f"failed for time={t}"


# ---------------------------------------------------------------------------
# TransitCNN
# ---------------------------------------------------------------------------

def test_transitcnn_output_shapes():
    model = TransitCNN()
    b = _batch()
    mean, log_var = model(b["airs"], b["fgs1"], b["aux"])
    assert mean.shape    == (4, 283), f"mean shape: {mean.shape}"
    assert log_var.shape == (4, 283), f"log_var shape: {log_var.shape}"


def test_transitcnn_no_nan():
    """Forward pass must not produce NaN outputs."""
    model = TransitCNN()
    b = _batch()
    mean, log_var = model(b["airs"], b["fgs1"], b["aux"])
    assert not torch.isnan(mean).any(),    "mean contains NaN"
    assert not torch.isnan(log_var).any(), "log_var contains NaN"


def test_transitcnn_custom_dims():
    """TransitCNN must respect non-default constructor arguments."""
    model = TransitCNN(n_airs_channels=64, n_aux=5, n_output_wl=100, embed_dim=64)
    airs = torch.randn(2, 64, 80)
    fgs1 = torch.randn(2, 1,  80)
    aux  = torch.randn(2, 5)
    mean, log_var = model(airs, fgs1, aux)
    assert mean.shape    == (2, 100)
    assert log_var.shape == (2, 100)


# ---------------------------------------------------------------------------
# gaussian_nll_loss
# ---------------------------------------------------------------------------

def test_nll_perfect_mean_zero_residual():
    """When mean == target, loss = 0.5 * mean(log_var). With log_var=0 → loss=0."""
    target  = torch.ones(4, 283)
    mean    = torch.ones(4, 283)
    log_var = torch.zeros(4, 283)   # var = 1.0
    loss = gaussian_nll_loss(mean, log_var, target)
    assert abs(loss.item()) < 1e-5, f"expected ~0, got {loss.item()}"


def test_nll_penalises_overconfidence():
    """
    Small variance (log_var << 0) with non-zero residual must give large loss.
    This verifies the model cannot cheat by predicting tiny uncertainty.
    """
    target  = torch.zeros(4, 283)
    mean    = torch.ones(4, 283)    # residual = 1.0 everywhere
    log_var_confident  = torch.full((4, 283), -5.0)   # var ≈ 0.0067
    log_var_uncertain  = torch.full((4, 283),  2.0)   # var ≈ 7.4
    loss_confident  = gaussian_nll_loss(mean, log_var_confident,  target)
    loss_uncertain  = gaussian_nll_loss(mean, log_var_uncertain,  target)
    assert loss_confident > loss_uncertain, (
        "overconfident predictions should have higher NLL than uncertain ones"
    )
