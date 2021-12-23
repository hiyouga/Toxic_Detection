import torch
import torch.nn as nn
from transformers import BertModel


class BERT(nn.Module):

    def __init__(self, configs):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained(configs['bert_name'])
        self.dropout = nn.Dropout(configs['dropout'])
        self.dense = nn.Linear(768, configs['num_classes'])

    def forward(self, text):
        mask = torch.where(text > 0, torch.ones_like(text), torch.zeros_like(text))
        output = self.bert(input_ids=text, attention_mask=mask)
        cls_out = output[0][:, 0, :]
        output = self.dense(self.dropout(cls_out))
        return output


def bert(configs):
    return BERT(configs)
