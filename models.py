from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Type

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from xlstm import (
        FeedForwardConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
        sLSTMBlockConfig,
        sLSTMLayerConfig,
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
    )
except Exception:
    xLSTMBlockStack = None
    xLSTMBlockStackConfig = None
    mLSTMBlockConfig = None
    mLSTMLayerConfig = None
    sLSTMBlockConfig = None
    sLSTMLayerConfig = None
    FeedForwardConfig = None


class FlattenMLPEncoder(BaseFeaturesExtractor):
    """Default extractor with no internal recurrence.

    RecurrentPPO already has a policy LSTM. Keeping the feature extractor
    feed-forward usually trains more stably on small financial datasets.
    """

    def __init__(self, observation_space, hidden_dim: int = 128, dropout: float = 0.05):
        window_size, n_features = observation_space.shape
        super().__init__(observation_space, features_dim=hidden_dim)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size * n_features, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, obs):
        return self.net(obs)


class LastTokenMLPEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim: int = 128, dropout: float = 0.05):
        _, n_features = observation_space.shape
        super().__init__(observation_space, features_dim=hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, obs):
        return self.net(obs[:, -1, :])


class LSTMEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim: int = 128, dropout: float = 0.05):
        super().__init__(observation_space, features_dim=hidden_dim)
        _, n_features = observation_space.shape
        self.input_proj = nn.Sequential(nn.Linear(n_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, obs):
        x = self.input_proj(obs)
        out, _ = self.lstm(x)
        return self.dropout(self.norm(out[:, -1, :]))


def build_xlstm_stack(embedding_dim: int, context_length: int, num_blocks: int = 3, use_slstm: bool = False):
    if xLSTMBlockStack is None:
        raise ImportError("Пакет xlstm не установлен. Установи: pip install git+https://github.com/NX-AI/xlstm.git")
    if use_slstm and (not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8):
        use_slstm = False
    if use_slstm:
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(backend="cuda", num_heads=4, conv1d_kernel_size=4, bias_init="powerlaw_blockdependent"),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=[1],
        )
    else:
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=[],
        )
    return xLSTMBlockStack(cfg)


class xLSTMEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, embedding_dim: int = 128, num_blocks: int = 3, use_slstm: bool = False, dropout: float = 0.05):
        super().__init__(observation_space, features_dim=embedding_dim)
        window_size, n_features = observation_space.shape
        self.input_proj = nn.Sequential(nn.Linear(n_features, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU())
        self.xlstm = build_xlstm_stack(embedding_dim, window_size, num_blocks, use_slstm)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, observations):
        x = self.input_proj(observations)
        x = self.xlstm(x)
        return self.dropout(self.norm(x[:, -1, :]))


class xLSTMEncoderWithAttention(BaseFeaturesExtractor):
    def __init__(self, observation_space, embedding_dim: int = 128, num_blocks: int = 3, num_heads: int = 4, use_slstm: bool = False, dropout: float = 0.05):
        super().__init__(observation_space, features_dim=embedding_dim)
        window_size, n_features = observation_space.shape
        self.input_proj = nn.Sequential(nn.Linear(n_features, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU())
        self.xlstm = build_xlstm_stack(embedding_dim, window_size, num_blocks, use_slstm)
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, observations):
        x = self.input_proj(observations)
        x = self.xlstm(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return self.dropout(x[:, -1, :])


class xLSTMEncoderLarge(BaseFeaturesExtractor):
    def __init__(self, observation_space, embedding_dim: int = 192, num_blocks: int = 5, use_slstm: bool = False, dropout: float = 0.05):
        super().__init__(observation_space, features_dim=embedding_dim)
        window_size, n_features = observation_space.shape
        self.input_proj = nn.Sequential(nn.Linear(n_features, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU())
        self.xlstm = build_xlstm_stack(embedding_dim, window_size, num_blocks, use_slstm)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, observations):
        x = self.input_proj(observations)
        x = self.xlstm(x)
        return self.dropout(self.norm(x[:, -1, :]))


class CausalTransformerEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, embedding_dim: int = 128, num_heads: int = 4, num_layers: int = 2, ffn_dim: int = 256, dropout: float = 0.05):
        super().__init__(observation_space, features_dim=embedding_dim)
        window_size, n_features = observation_space.shape
        self.input_proj = nn.Sequential(nn.Linear(n_features, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU())
        self.pos_emb = nn.Embedding(window_size, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        mask = torch.triu(torch.ones(window_size, window_size), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, observations):
        _, T, _ = observations.shape
        x = self.input_proj(observations)
        pos = torch.arange(T, device=observations.device)
        x = x + self.pos_emb(pos).unsqueeze(0)
        x = self.transformer(x, mask=self.causal_mask[:T, :T])
        return self.dropout(self.norm(x[:, -1, :]))


class _GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.05):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.sigmoid(self.gate(x)) * torch.nn.functional.elu(self.fc(x))
        h = self.drop(self.out(h))
        return self.norm(x + h)


class TFTLikeEncoder(BaseFeaturesExtractor):
    """TFT-inspired encoder: GRN + causal attention + GRN."""

    def __init__(self, observation_space, embedding_dim: int = 128, num_heads: int = 4, num_grn_layers: int = 2, dropout: float = 0.05):
        super().__init__(observation_space, features_dim=embedding_dim)
        window_size, n_features = observation_space.shape
        self.input_proj = nn.Sequential(nn.Linear(n_features, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU())
        self.pos_emb = nn.Embedding(window_size, embedding_dim)
        self.pre_grns = nn.Sequential(*[_GatedResidualNetwork(embedding_dim, embedding_dim, dropout) for _ in range(num_grn_layers)])
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(embedding_dim)
        self.post_grn = _GatedResidualNetwork(embedding_dim, embedding_dim, dropout)
        mask = torch.triu(torch.ones(window_size, window_size), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, observations):
        _, T, _ = observations.shape
        x = self.input_proj(observations)
        pos = torch.arange(T, device=observations.device)
        x = x + self.pos_emb(pos).unsqueeze(0)
        x = self.pre_grns(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=self.causal_mask[:T, :T])
        x = self.attn_norm(x + attn_out)
        x = self.post_grn(x)
        return self.dropout(self.norm(x[:, -1, :]))


def check_slstm_available() -> bool:
    import sys
    if sys.platform == "win32":
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 8


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelSpec:
    name: str
    extractor_class: Type[BaseFeaturesExtractor]
    extractor_kwargs: Dict[str, Any]
    short: bool = False


def get_model_registry(use_slstm: bool = False) -> Dict[str, ModelSpec]:
    registry: Dict[str, ModelSpec] = {
        "mlp": ModelSpec("mlp", FlattenMLPEncoder, {"hidden_dim": 128, "dropout": 0.05}, short=False),
        "last_mlp": ModelSpec("last_mlp", LastTokenMLPEncoder, {"hidden_dim": 128, "dropout": 0.05}, short=False),
        "lstm": ModelSpec("lstm", LSTMEncoder, {"hidden_dim": 128, "dropout": 0.05}, short=False),
        "transformer_causal": ModelSpec(
            "transformer_causal",
            CausalTransformerEncoder,
            {"embedding_dim": 128, "num_heads": 4, "num_layers": 2, "ffn_dim": 256, "dropout": 0.05},
            short=False,
        ),
        "tft_like": ModelSpec(
            "tft_like",
            TFTLikeEncoder,
            {"embedding_dim": 128, "num_heads": 4, "num_grn_layers": 2, "dropout": 0.05},
            short=False,
        ),
    }
    if xLSTMBlockStack is not None:
        registry.update(
            {
                "xlstm_base": ModelSpec("xlstm_base", xLSTMEncoder, {"embedding_dim": 128, "num_blocks": 3, "use_slstm": use_slstm, "dropout": 0.05}, short=False),
                "xlstm_attn": ModelSpec("xlstm_attn", xLSTMEncoderWithAttention, {"embedding_dim": 128, "num_blocks": 3, "num_heads": 4, "use_slstm": use_slstm, "dropout": 0.05}, short=False),
                "xlstm_large": ModelSpec("xlstm_large", xLSTMEncoderLarge, {"embedding_dim": 192, "num_blocks": 5, "use_slstm": use_slstm, "dropout": 0.05}, short=False),
            }
        )
    return registry


def build_policy_kwargs(spec: ModelSpec, pi_hidden=(128, 64), vf_hidden=(128, 64)) -> Dict[str, Any]:
    return {
        "features_extractor_class": spec.extractor_class,
        "features_extractor_kwargs": spec.extractor_kwargs,
        "net_arch": {"pi": list(pi_hidden), "vf": list(vf_hidden)},
    }


def smoke_test_encoder(encoder_cls, encoder_kwargs, observation_space, device=None):
    device = device or get_device()
    encoder = encoder_cls(observation_space, **encoder_kwargs).to(device)
    dummy = torch.zeros(2, *observation_space.shape, device=device)
    out = encoder(dummy)
    return tuple(out.shape)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
