import os
import sys
import time
import torch
import random
import logging
import argparse
import warnings
import datetime
import numpy as np
from trainers import DefaultTrainer, InvratTrainer
from data_utils import load_data
from models import textcnn, textrnn, restext, bert
from evaluation import evaluate
from transformers import AutoModel
from copy import deepcopy
from pathlib import Path


def bert_backbone_config(name):
    backbone = AutoModel.from_pretrained(name)
    arch = str(name).split('-')[0]
    if arch in ['bert', 'roberta', "distilroberta"]:
        return {'embeddings': backbone.embeddings, 'encoder': backbone.encoder, 'num_hidden_layers': backbone.config.num_hidden_layers,
                'extend_attention': True}
    if arch in ["distilbert"]:
        return {'embeddings': backbone.embeddings, 'encoder': backbone.transformer, 'num_hidden_layers': backbone.config.num_hidden_layers}
    else:
        raise NotImplementedError(f"unsupported model {name}")


class Instructor:

    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
        self._print_args()
        dataloaders = load_data(args=args, batch_size=args.batch_size, bert_name=args.bert_name)
        self.train_dataloader, self.dev_dataloader, self.test_dataloader, self.tokenizer, embedding_matrix = dataloaders
        args.tokenizer = self.tokenizer
        self.logger.info('=> creating model')
        writer = None
        if args.do_invrat:
            rationale_configs = {'num_classes': 2, 'embedding_matrix': embedding_matrix, 'dropout': 0.1, 'output_token_hidden': True}
            inv_configs = {'num_classes': 2, 'embedding_matrix': embedding_matrix, 'dropout': 0.1}
            enable_configs = {'num_classes': 2, 'embedding_matrix': embedding_matrix, 'dropout': 0.1, 'use_env': True,
                              'accumulator': args.accumulator}
            if args.bert_name:
                update_config = bert_backbone_config(args.bert_name)
                if args.weight_sharing:
                    rationale_configs.update(update_config)
                    inv_configs.update(update_config)
                    enable_configs.update(update_config)
                else:
                    rationale_configs.update(deepcopy(update_config))
                    inv_configs.update(deepcopy(update_config))
                    enable_configs.update(deepcopy(update_config))
            models = [args.rationale_model_class(rationale_configs), args.model_class(inv_configs), args.model_class(enable_configs)]
            self.trainer = InvratTrainer(models, writer, args)
        else:
            configs = {'num_classes': 2, 'embedding_matrix': embedding_matrix, 'dropout': 0.1}
            if args.bert_name:
                configs.update(bert_backbone_config(args.bert_name))
            model = args.model_class(configs)
            self.trainer = DefaultTrainer(model, writer, args)
        if args.from_ckpt:
            self.trainer.load_state_dict(torch.load(args.from_ckpt))
        self.trainer.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info(f"=> cuda memory allocated: {torch.cuda.memory_allocated(self.args.device.index)}")
        self.best_record = {'epoch': 0, 'overall_auc': 0, 'model_state': None}
        print('=> build Instructor done')

    def _print_args(self):
        print('TRAINING ARGUMENTS:')
        for arg in vars(self.args):
            print(f">>> {arg}: {getattr(self.args, arg)}")

    def _update_record(self, global_step, overall_auc, best_record):
        if overall_auc > best_record['overall_auc']:
            best_record['global_step'] = global_step
            best_record['overall_auc'] = overall_auc
            best_record['model_state'] = self.trainer.save_state_dict()
            torch.save(self.trainer.save_state_dict(), os.path.join('state_dict', f"{self.args.timestamp}.pt"))
        return best_record

    def _train(self, dataloader, epoch, eva_update_func):
        self.trainer.train(dataloader, epoch, eva_update_func)

    def _validate(self, dataloader, max_steps=None, inference=False):
        torch.cuda.empty_cache()
        if inference:
            all_cid, all_pred, _ = self.trainer.predict(dataloader)
            return all_cid, all_pred
        else:
            val_loss, val_acc, all_cid, all_pred, _ = self.trainer.evaluate(dataloader, max_steps)
            ''' bias auc may be nan when a subgroup is empty '''
            overall_auc, bias_auc, final_score = evaluate(np.array(all_pred), np.array(all_cid), dev_json=args.dev_json)
            return val_loss, val_acc, overall_auc, bias_auc, final_score

    def evaluate_and_update(self, global_step, max_steps):
        val_loss, val_acc, overall_auc, bias_auc, final_score = self._validate(self.dev_dataloader, max_steps)
        self.best_record = self._update_record(global_step, overall_auc, self.best_record)
        self.logger.info(f"[val] loss: {val_loss:.4f}, acc: {val_acc * 100:.2f}")
        self.logger.info(f"[val] auc: {overall_auc * 100:.2f}, bias_auc: {bias_auc * 100:.2f}, score: {final_score * 100:.2f}")
        all_cid, all_pred = self._validate(self.test_dataloader, inference=True)
        fname = f"{self.args.model_name}_{self.args.bert_name}_{self.args.timestamp}/{global_step}_{self.best_record['overall_auc'] * 100:.2f}.txt"
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        with open(fname, 'w', encoding='utf-8') as f:
            f.write('\n'.join([f"{cid} {pred}" for cid, pred in zip(all_cid, all_pred)]))
        self.logger.info(f"submission result saved: {fname}")

    def run(self):
        for epoch in range(self.args.num_epoch):
            self._train(self.train_dataloader, epoch, self.evaluate_and_update)
            # self.logger.info(f"{epoch + 1}/{self.args.num_epoch} - {100 * (epoch + 1) / self.args.num_epoch:.2f}%")
            # self.logger.info(f"[train] loss: {train_loss:.4f}, acc: {train_acc * 100:.2f}")


if __name__ == '__main__':

    model_classes = {'textcnn': textcnn, 'textrnn': textrnn, 'restext': restext, 'bert': bert}
    parser = argparse.ArgumentParser(description='Trainer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ''' dataset '''
    parser.add_argument('--train_json', type=str, default='train.json')
    parser.add_argument('--dev_json', type=str, default='dev.json')
    parser.add_argument('--test_json', type=str, default='test.json')
    ''' model '''
    parser.add_argument('--model_name', type=str, default='textcnn', choices=model_classes.keys(), help='Classifier model architecture.')
    parser.add_argument('--bert_name', type=str, default=None, help='Bert name.')
    ''' optimization '''
    parser.add_argument('--num_epoch', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=1e-5, help='Weight decay (L2 penalty).')
    parser.add_argument('--clip_norm', type=int, default=20, help='Maximum norm of gradients.')
    ''' environment '''
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='Selected device.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--timestamp', type=str, default=None, help='Experiment timestamp.')
    parser.add_argument('--no_bar', default=False, action='store_true', help='Disable process bar.')
    parser.add_argument('--no_backend', default=False, action='store_true', help='Use frontend matplotlib.')
    ''' invrat '''
    trainer_classes = {'default': DefaultTrainer, 'invrat': InvratTrainer}
    parser.add_argument('--trainer_name', default='default', type=str, choices=trainer_classes.keys(), help='choose trainer')
    parser.add_argument('--rationale_name', type=str, default=None, help='rationale generator model class name')
    parser.add_argument('--accumulator', type=str, default='sum', help='multi env pooler')
    parser.add_argument('--sparsity_percentage', type=float, default=0.2, help='the sparsity percentage for rationale')
    parser.add_argument('--sparsity_lambda', type=float, default=1, help='the penalty coefficient for rationale sparsity loss')
    parser.add_argument('--continuity_lambda', type=float, default=2, help='the penalty coefficient for rationale continuity loss')
    parser.add_argument('--diff_lambda', type=float, default=10, help='the penalty coefficient for env_inv and env_enable model diff loss')
    parser.add_argument('--weight_sharing', default=False, action='store_true', help='sharing bert weight among models')
    parser.add_argument('--from_ckpt', type=str)

    args = parser.parse_args()
    args.model_class = model_classes[args.model_name]
    args.trainer_class = trainer_classes[args.trainer_name]
    args.do_invrat = (args.trainer_name == 'invrat')
    if args.do_invrat:
        args.rationale_model_class = model_classes[args.rationale_name]
    args.log_name = f"{args.model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:]}.log"
    args.timestamp = args.timestamp if args.timestamp else str(int(time.time())) + format(random.randint(0, 999), '03')
    args.seed = args.seed if args.seed else random.randint(0, 2 ** 32 - 1)
    args.device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ''' set seeds '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    ''' global settings '''
    for dir_name in ['dats', 'logs', 'state_dict']:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
    warnings.simplefilter("ignore")
    ins = Instructor(args)
    ins.run()
