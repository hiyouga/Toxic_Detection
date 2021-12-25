import torch
import torch.nn as nn

from .env import multienv


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
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(len(KS) * KN, C)
        self.dropout = nn.Dropout(0.1)

        self.use_env = configs['use_env']
        if self.use_env:
            accumulator = configs['accumulator']
            self.env_model = multienv(WD, accumulator)

    def forward(self, text, mask=None, env=None):
        if self.use_env and env is None:
            raise RuntimeWarning("build a env-enable model, but get no env input")
        if not self.use_env and env is not None:
            raise RuntimeError("build a env-free model, but get env input")

        if mask is not None:
            text = torch.mul(text, mask)
        word_emb = self.embed(text)
        if self.use_env and env is not None:
            env_embeddings = self.env_model(env)
            env_embeddings = env_embeddings.unsqueeze(dim=1).expand_as(word_emb)
            word_emb += env_embeddings
        word_emb = self.dropout(word_emb)
        maxpool_out = list()
        for conv in self.conv:
            cnn_out_i = conv(word_emb.transpose(1, 2))
            maxpool_i = self.maxpool(cnn_out_i).squeeze(-1)
            maxpool_out.append(maxpool_i)
        maxpool_out = torch.cat(maxpool_out, dim=-1)
        output = self.linear(self.dropout(maxpool_out))
        return output


def textcnn(configs):
    return TextCNN(256, [3,4,5], configs)
