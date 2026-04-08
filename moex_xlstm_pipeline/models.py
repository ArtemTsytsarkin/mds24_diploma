from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Type

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


class LSTMEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim: int = 128):
        super().__init__(observation_space, features_dim=hidden_dim)
        _, n_features = observation_space.shape
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, num_layers=1, batch_first=True)

    def forward(self, obs):
        out, _ = self.lstm(obs)
        return out[:, -1, :]


def build_xlstm_stack(embedding_dim: int, context_length: int, num_blocks: int = 4, use_slstm: bool = False):
    if xLSTMBlockStack is None:
        raise ImportError(
            "Пакет xlstm не установлен. Установи `pip install git+https://github.com/NX-AI/xlstm.git`."
        )

    # Fix #7: sLSTM requires CUDA Compute >= 8.0; fall back to mLSTM-only if unavailable.
    if use_slstm:
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            import warnings
            warnings.warn(
                "sLSTM requires CUDA Compute Capability >= 8.0, which is not available. "
                "Falling back to mLSTM-only stack.",
                RuntimeWarning,
                stacklevel=2,
            )
            use_slstm = False

    if use_slstm:
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=[1],
        )
    else:
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=[],
        )
    return xLSTMBlockStack(cfg)


class xLSTMEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, embedding_dim: int = 128, num_blocks: int = 4, use_slstm: bool = False):
        super().__init__(observation_space, features_dim=embedding_dim)
        window_size, n_features = observation_space.shape
        self.input_proj = nn.Sequential(nn.Linear(n_features, embedding_dim), nn.GELU())
        self.xlstm = build_xlstm_stack(embedding_dim, window_size, num_blocks, use_slstm)

    def forward(self, observations):
        x = self.input_proj(observations)
        x = self.xlstm(x)
        return x[:, -1, :]


class xLSTMEncoderWithAttention(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        embedding_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
        use_slstm: bool = False,
    ):
        super().__init__(observation_space, features_dim=embedding_dim)
        window_size, n_features = observation_space.shape
        self.input_proj = nn.Sequential(nn.Linear(n_features, embedding_dim), nn.GELU())
        self.xlstm = build_xlstm_stack(embedding_dim, window_size, num_blocks, use_slstm)
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, observations):
        x = self.input_proj(observations)
        x = self.xlstm(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return x[:, -1, :]


class xLSTMEncoderLarge(BaseFeaturesExtractor):
    def __init__(self, observation_space, embedding_dim: int = 256, num_blocks: int = 6, use_slstm: bool = False):
        super().__init__(observation_space, features_dim=embedding_dim)
        window_size, n_features = observation_space.shape
        self.input_proj = nn.Sequential(nn.Linear(n_features, embedding_dim), nn.GELU())
        self.xlstm = build_xlstm_stack(embedding_dim, window_size, num_blocks, use_slstm)

    def forward(self, observations):
        x = self.input_proj(observations)
        x = self.xlstm(x)
        return x[:, -1, :]


def check_slstm_available() -> bool:
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
    lstm_hidden_size: int
    short: bool = True


def get_model_registry(use_slstm: bool) -> Dict[str, ModelSpec]:
    return {
        "lstm": ModelSpec(
            name="lstm",
            extractor_class=LSTMEncoder,
            extractor_kwargs={"hidden_dim": 128},
            lstm_hidden_size=128,
        ),
        "xlstm_base": ModelSpec(
            name="xlstm_base",
            extractor_class=xLSTMEncoder,
            extractor_kwargs={"embedding_dim": 128, "num_blocks": 4, "use_slstm": use_slstm},
            lstm_hidden_size=128,
        ),
        "xlstm_attn": ModelSpec(
            name="xlstm_attn",
            extractor_class=xLSTMEncoderWithAttention,
            extractor_kwargs={"embedding_dim": 128, "num_blocks": 4, "num_heads": 4, "use_slstm": use_slstm},
            lstm_hidden_size=128,
        ),
        "xlstm_large": ModelSpec(
            name="xlstm_large",
            extractor_class=xLSTMEncoderLarge,
            extractor_kwargs={"embedding_dim": 256, "num_blocks": 6, "use_slstm": use_slstm},
            lstm_hidden_size=256,
        ),
    }


def build_policy_kwargs(spec: ModelSpec) -> Dict[str, Any]:
    return {
        "features_extractor_class": spec.extractor_class,
        "features_extractor_kwargs": spec.extractor_kwargs,
        "lstm_hidden_size": spec.lstm_hidden_size,
        "n_lstm_layers": 1,
        "enable_critic_lstm": True,
        "net_arch": [],
    }


def smoke_test_encoder(encoder_cls, encoder_kwargs, observation_space, device=None):
    device = device or get_device()
    encoder = encoder_cls(observation_space, **encoder_kwargs).to(device)
    dummy = torch.zeros(2, *observation_space.shape).to(device)
    out = encoder(dummy)
    return tuple(out.shape)
