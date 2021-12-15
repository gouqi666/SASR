from torch import nn

from utils import TextFeaturizer


class Transformer(nn.Module):
    def __init__(self,
                 am_featurizer: TextFeaturizer,
                 lm_featurizer: TextFeaturizer,
                 embedding_dim=512,
                 bert_dim=768,
                 d_model=256,
                 num_heads=4,
                 dff=1024,
                 drop_rate=0.1):
        super().__init__()
        self.lm_featurizer = lm_featurizer
        self.am_featurizer = am_featurizer
        self.embedding = nn.Embedding(am_featurizer.vocab_size, embedding_dim)
        self.embedding_linear = nn.Linear(embedding_dim, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, dim_feedforward=dff, dropout=drop_rate, nhead=num_heads),
            num_layers=5
        )
        self.bert_feature_layer = nn.Linear(d_model, bert_dim)
        self.feed_forward = nn.Linear(bert_dim, lm_featurizer.vocab_size)

    def forward(self, x, mask=None):
        enc_output = self.encoder(self.embedding_linear(self.embedding(x)), src_key_padding_mask=mask.transpose(-1, 0))
        bert_out = self.bert_feature_layer(enc_output)
        final_out = self.feed_forward(bert_out)
        return final_out, bert_out
