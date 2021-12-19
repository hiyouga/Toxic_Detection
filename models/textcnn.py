import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self, kernel_num, kernel_sizes, configs):
        super(TextCNN, self).__init__()

        WN, WD = configs['embedding_matrix'].shape
        KN = kernel_num
        KS = kernel_sizes
        C = configs['num_classes']

        self.embed = nn.Embedding.from_pretrained(torch.tensor(configs['embedding_matrix'], dtype=torch.float))
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(WD, KN, K, padding=K//2, bias=True),
                nn.ReLU(inplace=True),
            ) for K in KS
        ])
        self.linear = nn.Linear(len(KS) * KN, C)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text):
        word_emb = self.dropout(self.embed(text))
        maxpool_out = list()
        for conv in self.conv:
            cnn_out_i = conv(word_emb.transpose(1, 2))
            maxpool_i = F.max_pool1d(cnn_out_i, cnn_out_i.size(-1)).squeeze(-1)
            maxpool_out.append(maxpool_i)
        maxpool_out = torch.cat(maxpool_out, dim=-1)
        output = self.linear(self.dropout(maxpool_out))
        return output


def textcnn(configs):
    return TextCNN(256, [3,4,5], configs)
