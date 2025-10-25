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
        self.down_proj = nn.Linear(4 * config.hidden_dim, config.hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pass



class AttentionLayer(nn.Module):
    def __init__(self, config: MicroLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pass



class DecoderLayer(nn.Module):
    def __init__(self, config: MicroLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_attn_norm = nn.RMSNorm(config.hidden_dim)
        self.attn_layer = AttentionLayer(config)
        self.pre_mlp_norm = nn.RMSNorm(config.hidden_dim)
        self.mlp_layer = MLPLayer(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pass



class MicroLLM(nn.Module):
    def __init__(self, config: MicroLLMConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass