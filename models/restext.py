import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1x3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1x5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv1x7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        out1 = self.conv1x3(x)
        out2 = self.conv1x5(x)
        out3 = self.conv1x7(x)
        out = out1 + out2 + out3
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout):
        super(BasicBlock, self).__init__()
        self.residual = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            ConvBlock(out_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.residual(x) + x
        out = self.relu(out)
        return out


class ResText(nn.Module):

    def __init__(self, kernel_num, configs):
        super(ResText, self).__init__()

        WN, WD = configs['embedding_matrix'].shape
        KN = kernel_num
        C = configs['num_classes']

        self.embed = nn.Embedding.from_pretrained(torch.tensor(configs['embedding_matrix'], dtype=torch.float))
        self.conv1 = nn.Sequential(
            ConvBlock(WD, KN),
            nn.BatchNorm1d(KN),
            nn.ReLU(inplace=True)
        )
        self.layer1 = BasicBlock(KN, KN, dropout=0.1)
        self.layer2 = BasicBlock(KN, KN, dropout=0.1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(KN, C)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text):
        out = self.dropout(self.embed(text)).transpose(1, 2)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.maxpool(out).squeeze(-1)
        out = self.linear(self.dropout(out))
        return out


def restext(configs):
    return ResText(256, configs)
