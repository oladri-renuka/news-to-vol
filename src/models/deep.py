"""Sequence models for log-RV forecasting, shared across tickers through a ticker embedding."""
from __future__ import annotations

import math

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """2-layer LSTM. The 8-dim ticker embedding is concatenated to every time step."""

    def __init__(self, n_features: int, n_tickers: int, hidden: int = 64, layers: int = 2,
                 emb_dim: int = 8, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(n_tickers, emb_dim)
        self.lstm = nn.LSTM(n_features + emb_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, tk: torch.Tensor) -> torch.Tensor:
        e = self.emb(tk).unsqueeze(1).expand(-1, x.size(1), -1)
        h, _ = self.lstm(torch.cat([x, e], -1))
        return self.head(h[:, -1]).squeeze(-1)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class TransformerForecaster(nn.Module):
    """Encoder-only Transformer over the window. The ticker embedding is added to every token,
    and the model reads out from the last token, i.e. the forecast origin."""

    def __init__(self, n_features: int, n_tickers: int, d_model: int = 64, nhead: int = 4, layers: int = 3,
                 ff: int = 128, emb_dim: int = 8, dropout: float = 0.2, max_len: int = 512):
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.emb = nn.Embedding(n_tickers, emb_dim)
        self.emb_proj = nn.Linear(emb_dim, d_model)
        self.pe = SinusoidalPE(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, tk: torch.Tensor) -> torch.Tensor:
        z = self.inp(x) + self.emb_proj(self.emb(tk)).unsqueeze(1)
        z = self.enc(self.pe(z))
        return self.head(self.norm(z[:, -1])).squeeze(-1)


def build_model(mcfg: dict, n_features: int, n_tickers: int) -> nn.Module:
    kind = mcfg["type"]
    if kind == "lstm":
        return LSTMForecaster(n_features, n_tickers, mcfg.get("hidden", 64), mcfg.get("layers", 2),
                              mcfg.get("emb_dim", 8), mcfg.get("dropout", 0.2))
    if kind == "transformer":
        return TransformerForecaster(n_features, n_tickers, mcfg.get("d_model", 64), mcfg.get("nhead", 4),
                                     mcfg.get("layers", 3), mcfg.get("ff", 128), mcfg.get("emb_dim", 8),
                                     mcfg.get("dropout", 0.2))
    raise ValueError(kind)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
