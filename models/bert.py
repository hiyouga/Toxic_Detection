import torch
import torch.nn as nn
from transformers import BertModel

from .env import multienv


class BERT(nn.Module):

    def __init__(self, configs):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained(configs['bert_name'])
        self.dropout = nn.Dropout(configs['dropout'])
        self.dense = nn.Linear(768, configs['num_classes'])
        self.output_token_hidden = configs['output_token_hidden'] if 'output_token_hidden' in configs else False
        self.use_env = configs['use_env'] if 'use_env' in configs else False
        if self.use_env:
            accumulator = configs['accumulator']
            self.env_model = multienv(768, accumulator)

    def forward(self, text, mask=None, env=None):
        if self.use_env and env is None:
            raise RuntimeWarning("build a env-enable model, but get no env input")
        if not self.use_env and env is not None:
            raise RuntimeError("build a env-free model, but get env input")
        if mask is not None:
            mask = mask * torch.where(text > 0, torch.ones_like(text), torch.zeros_like(text))
        else:
            mask = torch.where(text > 0, torch.ones_like(text), torch.zeros_like(text))
        extended_attention_mask = get_extended_attention_mask(mask)
        # hack huggingface/BertModel to add token_embedding with env_embedding
        embedding_output = self.bert.embeddings(text)
        if self.use_env and env is not None:
            env_embeddings = self.env_model(env)
            env_embeddings = env_embeddings.unsqueeze(dim=1).expand_as(embedding_output)
            embedding_output += env_embeddings
        encoder_outputs = self.bert.encoder(embedding_output, attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]
        if self.output_token_hidden:
            output = self.dense(self.dropout(sequence_output))
            return output
        else:
            cls_out = sequence_output[:, 0]
            output = self.dense(self.dropout(cls_out))
            return output


def bert(configs):
    return BERT(configs)


def get_extended_attention_mask(attention_mask):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for attention_mask (shape {attention_mask.shape})"
        )
    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
