from torch import nn

from franken.models.bert.config import BertModelConfig
from franken.models.bert.layer import BertLayer


class BertEncoder(nn.Module):
    def __init__(self, config: BertModelConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        all_hidden_states = [hidden_states]
        all_attentions = []

        for layer_module in self.layer:
            layer_outputs, attentions = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs
            all_hidden_states.append(hidden_states)
            all_attentions.append(attentions)

        return hidden_states, all_hidden_states, all_attentions
