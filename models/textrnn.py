import torch
import torch.nn as nn
from .layers import DynamicLSTM


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

    def forward(self, text):
        word_emb = self.dropout(self.embed_layer(text))
        text_len = torch.sum(text!=0, dim=-1)
        rnn_output = self.rnn((word_emb, text_len.cpu()))
        output = rnn_output.sum(dim=1).div(text_len.float().unsqueeze(-1))
        output = self.linear(self.dropout(output))
        return output


def textrnn(configs):
    return TextRNN('GRU', 1, 300, configs)
