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
from trainers import DefaultTrainer
from data_utils import load_data
from models import textcnn, textrnn, restext, bert
from evaluation import evaluate


class Instructor:

    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
        self._print_args()
        dataloaders = load_data(batch_size=args.batch_size, bert_name=args.bert_name)
        self.train_dataloader, self.dev_dataloader, self.test_dataloader, self.tokenizer, embedding_matrix = dataloaders
        configs = {'num_classes': 2, 'embedding_matrix': embedding_matrix, 'bert_name': args.bert_name, 'dropout': 0.1}
        self.logger.info('=> creating model')
        writer = None
        self.trainer = DefaultTrainer(args.model_class(configs), writer, args)
        self.trainer.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info(f"=> cuda memory allocated: {torch.cuda.memory_allocated(self.args.device.index)}")

    def _print_args(self):
        print('TRAINING ARGUMENTS:')
        for arg in vars(self.args):
            print(f">>> {arg}: {getattr(self.args, arg)}")

    def _update_record(self, epoch, overall_auc, best_record):
        if overall_auc > best_record['overall_auc']:
            best_record['epoch'] = epoch
            best_record['overall_auc'] = overall_auc
            best_record['model_state'] = self.trainer.save_state_dict()
            torch.save(self.trainer.save_state_dict(), os.path.join('state_dict', f"{self.args.timestamp}.pt"))
        return best_record

    def _train(self, dataloader, epoch):
        train_loss, train_acc, _ = self.trainer.train(dataloader, epoch)
        return train_loss, train_acc

    def _validate(self, dataloader, epoch=None, inference=False):
        if inference:
            all_cid, all_pred, _ = self.trainer.predict(dataloader)
            return all_cid, all_pred
        else:
            val_loss, val_acc, all_cid, all_pred, _ = self.trainer.evaluate(dataloader, epoch)
            ''' bias auc may be nan when a subgroup is empty '''
            overall_auc, bias_auc, final_score = evaluate(np.array(all_pred))
            return val_loss, val_acc, overall_auc, bias_auc, final_score

    def run(self):
        best_record = {'epoch': 0, 'overall_auc': 0, 'model_state': None}
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(self.train_dataloader, epoch)
            val_loss, val_acc, overall_auc, bias_auc, final_score = self._validate(self.dev_dataloader, epoch)
            self.trainer.lr_scheduler_step()
            best_record = self._update_record(epoch+1, overall_auc, best_record)
            self.logger.info(f"{epoch+1}/{self.args.num_epoch} - {100*(epoch+1)/self.args.num_epoch:.2f}%")
            self.logger.info(f"[train] loss: {train_loss:.4f}, acc: {train_acc*100:.2f}")
            self.logger.info(f"[val] loss: {val_loss:.4f}, acc: {val_acc*100:.2f}")
            self.logger.info(f"[val] auc: {overall_auc*100:.2f}, bias_auc: {bias_auc*100:.2f}, score: {final_score*100:.2f}")
        self.logger.info(f"best overall auc: {best_record['overall_auc']*100:.2f}")
        if best_record['model_state'] is not None:
            self.trainer.load_state_dict(best_record['model_state'])
        self.logger.info(f"model saved: {self.args.timestamp}.pt")
        all_cid, all_pred = self._validate(self.test_dataloader, inference=True)
        with open(f"{self.args.model_name}_{self.args.timestamp}_{best_record['overall_auc']*100:.2f}.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join([f"{cid} {pred}" for cid, pred in zip(all_cid, all_pred)]))
        self.logger.info(f"submission result saved: {self.args.model_name}_{self.args.timestamp}_{best_record['overall_auc']*100:.2f}.txt")


if __name__ == '__main__':

    model_classes = {'textcnn': textcnn, 'textrnn': textrnn, 'restext': restext, 'bert': bert}
    parser = argparse.ArgumentParser(description='Trainer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    args = parser.parse_args()
    args.model_class = model_classes[args.model_name]
    args.log_name = f"{args.model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:]}.log"
    args.timestamp = args.timestamp if args.timestamp else str(int(time.time())) + format(random.randint(0, 999), '03')
    args.seed = args.seed if args.seed else random.randint(0, 2**32-1)
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
