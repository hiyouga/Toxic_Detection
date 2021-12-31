import torch
from torch import nn


class MultiEnv(nn.Module):
    def __init__(self, num_group, num_env, hidden_dim, accumulator):
        super().__init__()
        self.num_group = num_group
        self.num_env = num_env
        self.hidden_dim = hidden_dim
        self.embeds = nn.ModuleList([nn.Embedding(num_env, hidden_dim) for _ in range(num_group)])
        self.accumulator = self._get_accumulator(accumulator)

    def _get_accumulator(self, name):
        """
        x (num_group, batch, hidden_dim)
        output (batch, hidden_dim))
        """
        if name == 'sum':
            return lambda x: torch.sum(x, dim=0)
        elif name == 'mean':
            return lambda x: torch.mean(x, dim=0)
        elif name == 'max':
            return lambda x: torch.max(x, dim=0)[0]
        elif name == 'linear':
            self.lin = nn.Linear(self.hidden_dim * self.num_group, self.hidden_dim)

            def _acc(x):
                num_group, batch, hidden_dim = x.shape
                x = x.permute(1, 0, 2).reshape(batch, -1)
                return self.lin(x)

            return _acc
        else:
            raise NotImplementedError(f"unsupported accumulator type {name}")

    def forward(self, envs):
        """
        envs: (batch, num_group)
        out: (batch, hidden_dim)
        """
        assert envs.shape[-1] == self.num_group
        embeddings = []
        for i in range(self.num_group):
            embeddings.append(self.embeds[i](envs[..., i]))
        # embeddings (num_group, batch, hidden_dim)
        embeddings = torch.stack(embeddings)
        out = self.accumulator(embeddings)
        return out


def multienv(hidden_dim, accumulator):
    return MultiEnv(5, 10, hidden_dim, accumulator)
