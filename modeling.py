import torch.nn.functional as F
import torch.nn as nn
import torch


class MicroLLMConfig:
    def __init__(self):
        self.num_layers = 6
        self.hidden_dim = 64
        self.attn_dropout = 0.1
        self.attn_scale = self.hidden_dim ** -0.5
        self.vocab_size = 32000
        self.label_smoothing_factor = 0.1
        self.init_std = 0.02


# rope
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
        query = self.q_proj(inputs)
        key = self.k_proj(inputs)
        value = self.v_proj(inputs)
        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.config.attn_dropout,
            is_causal=True,
            scale=self.config.attn_scale
        )
        out = self.out_proj(x)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, config: MicroLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_attn_norm = nn.RMSNorm(config.hidden_dim)
        self.attn_layer = AttentionLayer(config)
        self.pre_mlp_norm = nn.RMSNorm(config.hidden_dim)
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
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.final_norm = nn.RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # initialization
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
        for layer in self.decoder_layers:
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
