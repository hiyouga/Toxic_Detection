import sty
import torch
import torch.nn as nn
from collections import namedtuple

_Invrats = namedtuple("_invrats", ('generator', 'env_inv', 'env_enable'))


class InvratTrainer:

    def __init__(self, models, writer, args):
        self.sparsity_percentage = args.sparsity_percentage
        self.sparsity_lambda = args.sparsity_lambda
        self.continuity_lambda = args.continuity_lambda
        self.diff_lambda = args.diff_lambda
        self.predict_use_rationale = True
        self.args = args

        if isinstance(models, (tuple, list)):
            self.models = _Invrats(*models)
        elif isinstance(models, dict):
            self.models = _Invrats(**models)
        else:
            assert False, f"models must be tuple or dict, but got type({models}) is {type(models)}"

        params, optimizers, schedulers = [], [], []
        for model in self.models:
            param = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = torch.optim.AdamW(param, lr=args.lr, weight_decay=args.decay)
            params.append(param), optimizers.append(optimizer)
        self.params, self.optimizers = _Invrats(*params), _Invrats(*optimizers)

        self.criterion = nn.CrossEntropyLoss()
        self._clip_norm = args.clip_norm
        self.no_bar = args.no_bar
        self.device = args.device
        self.writer = writer

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def lr_scheduler_step(self):
        pass

    def to(self, device):
        for model in self.models:
            model.to(device)

    def train_mode(self):
        for model in self.models:
            model.train()

    def eval_mode(self):
        for model in self.models:
            model.eval()

    def load_state_dict(self, state_dict):
        state_dict = _Invrats(**state_dict)
        for model, state in zip(self.models, state_dict):
            model.load_state_dict(state)

    def save_state_dict(self):
        state_dict = [model.state_dict() for model in self.models]
        state_dict = _Invrats(*state_dict)
        state_dict = state_dict._asdict()
        return state_dict

    def show_binary_rationale(self, tgt, inv_logit, en_logit, ids, z):
        text = self.args.tokenizer.bert_tokenizer.convert_ids_to_tokens(ids)
        output = f'[[{str(tgt)}]] [[{inv_logit}]] [[{en_logit}]]' + ' '
        for i, word in enumerate(text):
            if word == self.args.tokenizer.bert_tokenizer.pad_token:
                break
            if z[i] > 0.5:
                output += sty.fg.red + word + sty.fg.rs + ' '
            else:
                output += word + ' '
        print(output)

    def _step(self, inputs, masks, envs=None, inv_only=False, not_use_rationale=False):
        """
        inputs: (batch, seq_length)
        masks: (batch, seq_length)
        envs: (batch)
        """
        # ################## generator ################## #
        # TODO: make model forward can deal with masks.
        #  note that: mask is not restricted to left-aligned!!!
        #  because of the existence of rationale.
        if not_use_rationale:
            rationale = masks
        else:
            gen_logits = self.models.generator(inputs, masks)
            rationale = self._independent_straight_through_sampling(gen_logits, inputs.shape)
            # rationale (batch, seq_length)
            rationale = masks * rationale
        # ################## env_inv predictor ################## #
        env_inv_logits = self.models.env_inv(inputs, rationale)
        if inv_only:
            return rationale, env_inv_logits, None
        # ################## env_enable predictor ################## #
        # TODO: make model forward can deal with envs
        env_enable_logits = self.models.env_enable(inputs, rationale, envs)
        return rationale, env_inv_logits, env_enable_logits

    def train(self, dataloader, epoch, eva_update_func):
        self.train_mode()
        g_loss, inv_loss, inv_n_correct, enable_loss, enable_n_correct, n_train = 0, 0, 0, 0, 0, 0
        n_batch = len(dataloader)
        for i_batch, sample_batched in enumerate(dataloader):
            global_step = epoch * n_batch + i_batch
            self.zero_grad()
            # do forward
            inputs = sample_batched['text'].to(self.device)
            masks = sample_batched['mask'].to(self.device)
            targets = sample_batched['target'].to(self.device)
            envs = sample_batched['env'].to(self.device)
            # print(f"inputs.shape: {inputs.shape}")
            # print(f"masks.shape: {masks.shape}")
            rationale, env_inv_logits, env_enable_logits = self._step(inputs, masks, envs)
            env_inv_loss, env_enable_loss, diff_loss = self.inv_rat_loss(env_inv_logits, env_enable_logits, targets)
            sparsity_loss = self.sparsity_lambda * self.cal_sparsity_loss(rationale, masks)
            continuity_loss = self.continuity_lambda * self.cal_continuity_loss(rationale)
            gen_loss = self.diff_lambda * diff_loss + env_inv_loss + sparsity_loss + continuity_loss
            losses = _Invrats(gen_loss, env_inv_loss, env_enable_loss)
            # do backward and step
            path = global_step % 7
            if path in [0]:
                step_type = 0  # train generator
            elif path in [1, 2, 3]:
                step_type = 1  # train env_inv
            elif path in [4, 5, 6]:
                step_type = 2  # train env_enable
            else:
                assert False
            loss = losses[step_type]
            param = self.params[step_type]
            optimizer = self.optimizers[step_type]
            loss.backward()
            if self._clip_norm:
                nn.utils.clip_grad_norm_(param, self._clip_norm)
            optimizer.step()
            # log sth necessary
            g_loss += gen_loss.item() * targets.size(0)
            inv_loss += env_inv_loss.item() * targets.size(0)
            inv_n_correct += (torch.argmax(env_inv_logits, -1) == targets).sum().item()
            enable_loss += env_enable_loss.item() * targets.size(0)
            enable_n_correct += (torch.argmax(env_enable_logits, -1) == targets).sum().item()
            n_train += targets.size(0)
            if self.writer:
                self.writer.add_scalar(f"train/g_loss", gen_loss.item(), global_step)
                self.writer.add_scalar(f"train/inv_loss", env_inv_loss.item(), global_step)
                self.writer.add_scalar(f"train/enable_loss", env_enable_loss.item(), global_step)
            if not self.no_bar:
                ratio = int((i_batch + 1) * 50 / n_batch)  # process bar
                print(f"[{'>' * ratio}{' ' * (50 - ratio)}] {i_batch + 1}/{n_batch} {(i_batch + 1) * 100 / n_batch:.2f}%", end='\r')
            if (global_step >= n_batch / 10) and (global_step % 500 == 0 or i_batch == n_batch - 1):
            # if (global_step >= 30) and (global_step % 30 == 0 or i_batch == n_batch - 1):
                if not self.no_bar:
                    print()
                _inv_logits = torch.softmax(env_inv_logits, -1).cpu().detach().numpy()
                _en_logits = torch.softmax(env_enable_logits, -1).cpu().detach().numpy()
                _tgs = targets.cpu().detach().numpy()
                _t_idx = 0
                for _i, _v in enumerate(_tgs):
                    if _v > 0.5:
                        _t_idx = _i
                        break
                _f_idx = 0
                for _i, _v in enumerate(_tgs):
                    if _v < 0.5:
                        _f_idx = _i
                        break
                self.show_binary_rationale(_tgs[_t_idx], _inv_logits[_t_idx], _en_logits[_t_idx], inputs[_t_idx].cpu().detach().numpy(),
                                           rationale[_t_idx].cpu().detach().numpy())
                self.show_binary_rationale(_tgs[_f_idx], _inv_logits[_f_idx], _en_logits[_f_idx], inputs[_f_idx].cpu().detach().numpy(),
                                           rationale[_f_idx].cpu().detach().numpy())
                g_loss = g_loss / n_train if n_train > 0 else 'not_available'
                inv_loss = inv_loss / n_train if n_train > 0 else 'not_available'
                enable_loss = enable_loss / n_train if n_train > 0 else 'not_available'
                inv_acc = inv_n_correct / n_train if n_train > 0 else 'not_available'
                enable_acc = enable_n_correct / n_train if n_train > 0 else 'not_available'
                _log = {"rationale_has_nan": torch.any(torch.isnan(rationale))>0.5,
                        "g_loss": g_loss,
                        "sparsity_loss": sparsity_loss,
                        "continuity_loss": continuity_loss,
                        "inv_loss": inv_loss,
                        "enable_loss": enable_loss,
                        "inv_acc": inv_acc,
                        "enable_acc": enable_acc,
                        }
                print(f"[step] {global_step}")
                print(f"[train] ", end='')
                for k, v in _log.items():
                    print(f" {k}: {v:.4f} ", end='')
                print()
                if global_step % 10000 == 0:
                    max_steps = None
                else:
                    max_steps = 2000
                eva_update_func(global_step, max_steps)
                g_loss, inv_loss, inv_n_correct, enable_loss, enable_n_correct, n_train = 0, 0, 0, 0, 0, 0
        if not self.no_bar:
            print()

    def evaluate(self, dataloader, max_steps=None, epoch=None):
        self.eval_mode()
        all_cid, inv_preds, enable_preds = [], [], []
        g_loss, inv_loss, inv_n_correct, enable_loss, enable_n_correct, n_train = 0, 0, 0, 0, 0, 0
        n_batch = len(dataloader)
        if max_steps is None:
            max_steps = n_batch
        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch > max_steps:
                break
            global_step = epoch * n_batch + i_batch if epoch else i_batch
            # do forward
            inputs = sample_batched['text'].to(self.device)
            masks = sample_batched['mask'].to(self.device)
            targets = sample_batched['target'].to(self.device)
            envs = sample_batched['env'].to(self.device)
            rationale, env_inv_logits, env_enable_logits = self._step(inputs, masks, envs)
            env_inv_loss, env_enable_loss, diff_loss = self.inv_rat_loss(env_inv_logits, env_enable_logits, targets)
            sparsity_loss = self.sparsity_lambda * self.cal_sparsity_loss(rationale, masks)
            continuity_loss = self.continuity_lambda * self.cal_continuity_loss(rationale)
            gen_loss = self.diff_lambda * diff_loss + env_inv_loss + sparsity_loss + continuity_loss
            losses = _Invrats(gen_loss, env_inv_loss, env_enable_loss)
            # log sth necessary
            g_loss += gen_loss.item() * targets.size(0)
            inv_loss += env_inv_loss.item() * targets.size(0)
            inv_n_correct += (torch.argmax(env_inv_logits, -1) == targets).sum().item()
            enable_loss += env_enable_loss.item() * targets.size(0)
            enable_n_correct += (torch.argmax(env_enable_logits, -1) == targets).sum().item()
            n_train += targets.size(0)
            all_cid.extend(sample_batched['id'])
            outputs = torch.softmax(env_inv_logits, dim=-1)
            inv_preds.extend([outputs[x][1].item() for x in range(outputs.shape[0])])
            outputs = torch.softmax(env_enable_logits, dim=-1)
            enable_preds.extend([outputs[x][1].item() for x in range(outputs.shape[0])])
            if self.writer:
                self.writer.add_scalar(f"train/g_loss", gen_loss.item(), global_step)
                self.writer.add_scalar(f"train/inv_loss", env_inv_loss.item(), global_step)
                self.writer.add_scalar(f"train/enable_loss", env_enable_loss.item(), global_step)
        #     if not self.no_bar:
        #         ratio = int((i_batch + 1) * 50 / n_batch)  # process bar
        #         print(f"[{'>' * ratio}{' ' * (50 - ratio)}] {i_batch + 1}/{n_batch} {(i_batch + 1) * 100 / n_batch:.2f}%", end='\r')
        # if not self.no_bar:
        #     print()
        g_loss = g_loss / n_train
        inv_loss = inv_loss / n_train
        enable_loss = enable_loss / n_train
        inv_acc = inv_n_correct / n_train
        enable_acc = enable_n_correct / n_train

        return inv_loss, inv_acc, all_cid, inv_preds, {"g_loss": g_loss,
                                                       "inv_loss": inv_loss,
                                                       "enable_loss": enable_loss,
                                                       "inv_acc": inv_acc,
                                                       "enable_acc": enable_acc,
                                                       "inv_preds": inv_preds,
                                                       "enable_preds": enable_preds,
                                                       }

    def predict(self, dataloader):
        self.eval_mode()
        all_cid, all_rationale, inv_preds = [], [], []
        n_batch = len(dataloader)
        for i_batch, sample_batched in enumerate(dataloader):
            # do forward
            inputs = sample_batched['text'].to(self.device)
            masks = sample_batched['mask'].to(self.device)
            rationale, env_inv_logits, _ = self._step(inputs, masks, inv_only=True, not_use_rationale=False)
            # log sth necessary
            all_cid.extend(sample_batched['id'])
            outputs = torch.softmax(env_inv_logits, dim=-1)
            inv_preds.extend([outputs[x][1].item() for x in range(outputs.shape[0])])
            if rationale is not None:
                all_rationale.extend(rationale.tolist())
        #     if not self.no_bar:
        #         ratio = int((i_batch + 1) * 50 / n_batch)  # process bar
        #         print(f"[{'>' * ratio}{' ' * (50 - ratio)}] {i_batch + 1}/{n_batch} {(i_batch + 1) * 100 / n_batch:.2f}%", end='\r')
        # if not self.no_bar:
        #     print()
        return all_cid, inv_preds, {'rationale': all_rationale}

    # ##############################
    #  some Auxiliary functions
    # ##############################

    @staticmethod
    def _independent_straight_through_sampling(rationale_logits, id_shape):
        assert rationale_logits.shape[-1] == 2, f"rationale_logits.shape should be f{(*id_shape, 2)} but got {rationale_logits.shape}"
        # rationale_logits (batch, seq_length, 2)
        z = torch.softmax(rationale_logits, dim=-1)
        z_hard = (z >= 0.5).type_as(z)
        z_output = z + (z_hard - z).detach()
        rationale = z_output[:, :, 1]
        rationale[:, 0] = 1.0
        return rationale
        # y_soft = torch.max(rationale_logits, dim=-1, keepdim=True)[0]
        # y_hard = (y_soft == rationale_logits).to(rationale_logits.dtype)
        # ret = y_hard - y_soft.detach() + y_soft
        # ret = ret[..., 1]
        # ret[:, 0] = 1.0
        # return ret

    def inv_rat_loss(self, env_inv_logits, env_enable_logits, targets):
        """
            Compute the loss for the invariant rationalization training.
            Inputs:
                env_inv_logits -- logits of the predictor without env index
                                  (batch_size, num_classes)
                env_enable_logits -- logits of the predictor with env index
                                  (batch_size, num_classes)
                targets -- the ground truth labels
                                  (batch_size)
            """
        env_inv_loss = self.criterion(env_inv_logits, targets)
        env_enable_loss = self.criterion(env_enable_logits, targets)
        diff_loss = torch.relu(env_inv_loss - env_enable_loss)
        return env_inv_loss, env_enable_loss, diff_loss

    def cal_sparsity_loss(self, rationale, masks):
        sparsity = torch.sum(rationale) / (torch.sum(masks) + 1e-6)
        return torch.abs(sparsity - self.sparsity_percentage)

    def cal_continuity_loss(self, rationale):
        return torch.mean(torch.abs(rationale[:, 1:] - rationale[:, :-1]))
