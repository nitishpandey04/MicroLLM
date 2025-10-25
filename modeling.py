import torch.nn.functional as F
import torch.nn as nn
import torch



class MicroLLMConfig(dict):
    def __getattr__(self, attr):
        return self[attr]



class MLPLayer(nn.Module):
    def __init__(self, config: MicroLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.up_proj = nn.Linear(config.hidden_dim, 4 * config.hidden_dim)
        self.act_fn = nn.ReLU()
        self.down_proj = nn.Linear(4 * config.hidden_dim, config.hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(inputs)))


class AttentionLayer(nn.Module):
    def __init__(self, config: MicroLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = self.q_proj(inputs)
        key = self.k_proj(inputs)
        value = self.v_proj(inputs)
        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.1,
            is_causal=True,
            scale=self.config.hidden_dim**-0.5
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

        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.final_norm = nn.RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits






