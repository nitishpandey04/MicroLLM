from dataclasses import dataclass
import torch.nn.functional as F
import torch.nn as nn
import torch


@dataclass
class MicroLLMConfig:
    num_layers: int = 24
    hidden_dim: int = 128
    num_heads: int = 8
    attn_dropout: float = 0.1
    vocab_size: int = 32000
    label_smoothing_factor: float = 0.1
    init_std: float = 0.02


# gqa, swiglu, rope
class MLPLayer(nn.Module):
    def __init__(self, config: MicroLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.up_proj = nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False)
        self.act_fn = nn.ReLU()
        self.down_proj = nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(inputs)))


class AttentionLayer(nn.Module):
    def __init__(self, config: MicroLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, T, C = inputs.shape
        assert self.config.hidden_dim % self.config.num_heads == 0
        H, D = self.config.num_heads, self.config.hidden_dim // self.config.num_heads
        query = self.q_proj(inputs).view(B, T, H, D).transpose(1, 2)
        key = self.k_proj(inputs).view(B, T, H, D).transpose(1, 2)
        value = self.v_proj(inputs).view(B, T, H, D).transpose(1, 2)
        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.config.attn_dropout if self.training else 0.0,
            is_causal=True
        )
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(x)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, config: MicroLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_attn_norm = nn.RMSNorm(config.hidden_dim, elementwise_affine=config.norm_weight)
        self.attn_layer = AttentionLayer(config)
        self.pre_mlp_norm = nn.RMSNorm(config.hidden_dim, elementwise_affine=config.norm_weight)
        self.mlp_layer = MLPLayer(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        res = inputs
        x = self.attn_layer(self.pre_attn_norm(res))
        x += res
        res = x
        x = self.mlp_layer(self.pre_mlp_norm(res))
        x += res
        return x


class MicroLLM(nn.Module):
    def __init__(self, config: MicroLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.final_norm = nn.RMSNorm(config.hidden_dim, elementwise_affine=config.norm_weight)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor=None) -> tuple:
        x = self.wte(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if target_ids is not None:
            loss = F.cross_entropy(
                logits,
                target_ids,
                label_smoothing=self.config.label_smoothing_factor
            )
            return (logits, loss)
        return (logits,)
