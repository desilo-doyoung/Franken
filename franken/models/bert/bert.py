import torch
from torch import nn

from franken.config import ModelConfig
from franken.models.bert.embeddings import BertEmbeddings
from franken.models.bert.encoder import BertEncoder


class BertPooler(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def _extend_mask(self, attention_mask, dtype):
        mask = attention_mask[:, None, None, :].to(dtype)  # (B, 1, 1, S)
        return (1.0 - mask) * torch.finfo(dtype).min

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        emb = self.embeddings(input_ids, token_type_ids)
        extended_mask = (
            self._extend_mask(attention_mask, emb.dtype) if attention_mask is not None else None
        )
        last, all_hidden, all_attn = self.encoder(emb, extended_mask)
        pooled = self.pooler(last)

        return dict(
            last_hidden_state=last,
            pooled_output=pooled,
            all_hidden_states=all_hidden,
            all_attentions=all_attn,
        )


class BertForClassification(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled = self.dropout(outputs["pooled_output"])
        logits = self.classifier(pooled)
        return dict(
            logits=logits,
            hidden_states=outputs["all_hidden_states"],
            attentions=outputs["all_attentions"],
        )
