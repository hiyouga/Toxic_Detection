import torch
import torch.nn as nn


class DefaultTrainer:

    def __init__(self, model, writer, args):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self._clip_norm = args.clip_norm
        self.params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=args.lr, weight_decay=args.decay)
        self.device = args.device
        self.no_bar = args.no_bar
        self.writer = writer

    def lr_scheduler_step(self):
        pass

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

    def _train_step(self, inputs, masks, targets):
        self.optimizer.zero_grad()
        outputs = self.model(inputs, masks)
        loss = self.criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self._clip_norm)
        self.optimizer.step()
        return outputs, loss

    def _evaluate_step(self, inputs, masks, targets):
        outputs = self.model(inputs, masks)
        loss = self.criterion(outputs, targets)
        return outputs, loss

    def _predict_step(self, inputs, masks):
        outputs = self.model(inputs, masks)
        return outputs

    def train(self, dataloader, epoch, eva_update_func):
        train_loss, n_correct, n_train = 0, 0, 0
        n_batch = len(dataloader)
        self.train_mode()
        for i_batch, sample_batched in enumerate(dataloader):
            global_step = epoch * n_batch + i_batch
            inputs = sample_batched['text'].to(self.device)
            masks = (inputs > 0).to(inputs.dtype)
            targets = sample_batched['target'].to(self.device)
            outputs, loss = self._train_step(inputs, masks, targets)
            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_train += targets.size(0)
            if not self.no_bar:
                ratio = int((i_batch + 1) * 50 / n_batch)  # process bar
                print(f"[{'>' * ratio}{' ' * (50 - ratio)}] {i_batch + 1}/{n_batch} {(i_batch + 1) * 100 / n_batch:.2f}%", end='\r')
            if global_step % 300 == 0 or i_batch == n_batch - 1:
                _log = {"loss": train_loss / n_train if n_train > 0 else 'not_available',
                        "acc": n_correct / n_train if n_train > 0 else 'not_available',
                        }
                print(f"at step {global_step}")
                print(f"[train] {_log}")
                eva_update_func(global_step)
                train_loss, n_correct, n_train = 0, 0, 0
        if not self.no_bar:
            print()

    def evaluate(self, dataloader, epoch=None):
        val_loss, n_correct, n_val = 0, 0, 0
        all_cid, all_pred = list(), list()
        n_batch = len(dataloader)
        self.eval_mode()
        with torch.no_grad():
            all_pred = []
            for i_batch, sample_batched in enumerate(dataloader):
                inputs = sample_batched['text'].to(self.device)
                masks = (inputs > 0).to(inputs.dtype)
                targets = sample_batched['target'].to(self.device)
                outputs, loss = self._evaluate_step(inputs, masks, targets)
                val_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                all_cid.extend(sample_batched['id'])
                outputs = torch.softmax(outputs, dim=-1)
                all_pred.extend([outputs[x][1].item() for x in range(outputs.shape[0])])
                n_val += targets.size(0)
                if not self.no_bar:
                    ratio = int((i_batch + 1) * 50 / n_batch)  # process bar
                    print(f"[{'>' * ratio}{' ' * (50 - ratio)}] {i_batch + 1}/{n_batch} {(i_batch + 1) * 100 / n_batch:.2f}%", end='\r')
        if not self.no_bar:
            print()
        return val_loss / n_val, n_correct / n_val, all_cid, all_pred, {}

    def predict(self, dataloader, epoch=None):
        all_cid, all_pred = list(), list()
        n_batch = len(dataloader)
        self.eval_mode()
        with torch.no_grad():
            all_pred = []
            for i_batch, sample_batched in enumerate(dataloader):
                inputs = sample_batched['text'].to(self.device)
                masks = (inputs > 0).to(inputs.dtype)
                outputs, loss = self._predict_step(inputs, masks)
                all_cid.extend(sample_batched['id'])
                outputs = torch.softmax(outputs, dim=-1)
                all_pred.extend([outputs[x][1].item() for x in range(outputs.shape[0])])
                if not self.no_bar:
                    ratio = int((i_batch + 1) * 50 / n_batch)  # process bar
                    print(f"[{'>' * ratio}{' ' * (50 - ratio)}] {i_batch + 1}/{n_batch} {(i_batch + 1) * 100 / n_batch:.2f}%", end='\r')
        if not self.no_bar:
            print()
        return all_cid, all_pred, {}
