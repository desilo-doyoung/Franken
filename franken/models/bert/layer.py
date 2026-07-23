from torch import nn

from franken.config import ModelConfig
from franken.models.bert.attention import BertAttention
from franken.models.bert.ffn import BertIntermediate, BertOutput


class BertLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output, probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, probs
