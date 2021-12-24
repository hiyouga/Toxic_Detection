import torch
import torch.nn as nn


class DefaultTrainer:

    def __init__(self, model, args):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self._clip_norm = args.clip_norm
        self.params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=args.lr, weight_decay=args.decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.num_epoch)

    def lr_scheduler_step(self):
        self.scheduler.step()

    def to(self, device):
        self.model.to(device)

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def save_state_dict(self):
        return self.model.state_dict()

    def train(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self._clip_norm)
        self.optimizer.step()
        return outputs, loss

    def evaluate(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return outputs, loss

    def predict(self, inputs):
        outputs = self.model(inputs)
        return outputs
