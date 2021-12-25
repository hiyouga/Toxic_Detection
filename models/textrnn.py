import torch
import torch.nn as nn
from .layers import DynamicLSTM
from .env import multienv


class TextRNN(nn.Module):

    def __init__(self, rnn_type, num_layers, hidden_dim, configs):
        super(TextRNN, self).__init__()

        WN, WD = configs['embedding_matrix'].shape
        HD = hidden_dim
        C = configs['num_classes']

        self.embed_layer = nn.Embedding.from_pretrained(torch.tensor(configs['embedding_matrix'], dtype=torch.float))
        self.rnn = DynamicLSTM(WD, HD, num_layers=num_layers, batch_first=True, bidirectional=True, rnn_type=rnn_type)
        self.linear = nn.Linear(HD * 2, C)
        self.dropout = nn.Dropout(0.1)

        self.output_token_hidden = configs['output_token_hidden'] if 'output_token_hidden' in configs else False
        self.use_env = configs['use_env'] if 'use_env' in configs else False
        if self.use_env:
            accumulator = configs['accumulator']
            self.env_model = multienv(WD, accumulator)

    def forward(self, text, mask=None, env=None):
        """
        text: (batch, seq_len)
        mask: (batch, seq_len)
        env_embeddings: (batch, hidden_dim)
        """
        # calculate text_len first to get proper mask during rnn
        if self.use_env and env is None:
            raise RuntimeWarning("build a env-enable model, but get no env input")
        if not self.use_env and env is not None:
            raise RuntimeError("build a env-free model, but get env input")

        text_len = torch.sum(text != 0, dim=-1)
        if mask is not None:
            text = torch.mul(text, mask)
        word_emb = self.embed_layer(text)
        if self.use_env and env is not None:
            # env_embeddings (batch, hidden_dim)
            env_embeddings = self.env_model(env)
            # env_embeddings (batch, seq_len, hidden_dim)
            env_embeddings = env_embeddings.unsqueeze(dim=1).expand_as(word_emb)
            word_emb += env_embeddings
        word_emb = self.dropout(word_emb)
        rnn_output, _ = self.rnn(word_emb, text_len.cpu())
        if self.output_token_hidden:
            output = self.linear(self.dropout(rnn_output))
            return output
        else:
            output = rnn_output.sum(dim=1).div(text_len.float().unsqueeze(-1))
            output = self.linear(self.dropout(output))
            return output


def textrnn(configs):
    return TextRNN('GRU', 1, 300, configs)
