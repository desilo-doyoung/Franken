from torch import nn

from franken.models.llama.config import LlamaModelConfig
from franken.ops import build_activation


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaModelConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = build_activation(config.activation, **config.activation_kwargs)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
