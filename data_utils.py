import os
import json
import pickle
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class Vocab:

    def __init__(self, vocab_list, add_pad=True, add_unk=True):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad: # pad_id should be zero (for mask)
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._vocab_dict[self.pad_word] = self.pad_id
            self._length += 1
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._vocab_dict[self.unk_word] = self.unk_id
            self._length += 1
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w

    def word_to_id(self, word):
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]

    def id_to_word(self, idx):
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(idx, self.unk_word)
        return self._reverse_vocab_dict[idx]

    def has_word(self, word):
        return word in self._vocab_dict

    def __len__(self):
        return self._length


class Tokenizer:

    def __init__(self, vocab, lower, bert_name):
        self.vocab = vocab
        self.maxlen = 256
        self.lower = lower
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_name) if bert_name else None

    @classmethod
    def from_files(cls, fnames, lower=True):
       all_tokens = set()
       for fname in fnames:
           fdata = json.load(open(os.path.join('data', fname), 'r', encoding='utf-8'))
           for data in fdata:
               all_tokens.update([token.lower() if lower else token for token in Tokenizer.split_text(data['text'])])
       return cls(vocab=Vocab(all_tokens), lower=lower, bert_name=None)

    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.bert_tokenizer is not None:
            sequence = [101] + self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(text))
        else:
            words = self.split_text(text.lower() if self.lower else text)
            sequence = [self.vocab.word_to_id(w) for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence.reverse()
        padding_id = self.vocab.pad_id if self.vocab else 0
        return self.pad_sequence(sequence, pad_id=padding_id, maxlen=self.maxlen, padding=padding, truncating=truncating)

    @staticmethod
    def split_text(text):
        for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":", "@", "$"]:
            text = text.replace(ch, " "+ch+" ")
        return text.strip().split()


class ToxicDataset(Dataset):

    def __init__(self, fname, tokenizer, split, bert_name):
        cache_file = os.path.join('dats', f"{split}_{bert_name}.dat")
        if os.path.exists(cache_file):
            print(f"loading dataset: {cache_file}")
            dataset = pickle.load(open(cache_file, 'rb'))
        else:
            print('building dataset...')
            dataset = list()
            fdata = json.load(open(os.path.join('data', fname), 'r', encoding='utf-8'))
            for data in fdata:
                data['text'] = tokenizer.to_sequence(data['text'])
                if split == 'test':
                    data['target'] = 0
                else:
                    data['target'] = 0 if data['target'] < 0.5 else 1
                if 'env' in data.keys():
                    data['env'] = np.asarray(data['env']).astype(data['text'].dtype)
                dataset.append(data)
            pickle.dump(dataset, open(cache_file, 'wb'))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


def _load_wordvec(embed_file, word_dim, vocab=None):
    with open(embed_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        word_vec['<pad>'] = np.zeros(word_dim).astype('float32')
        for line in f:
            tokens = line.rstrip().split()
            if (len(tokens)-1) != word_dim:
                continue
            if tokens[0] == '<pad>' or tokens[0] == '<unk>':
                continue
            if vocab is None or vocab.has_word(tokens[0]):
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        return word_vec


def build_embedding_matrix(vocab, word_dim=300):
    cache_file = os.path.join('dats', 'embedding_matrix.dat')
    embed_file = os.path.join('..', 'glove', 'glove.840B.300d.txt')
    if os.path.exists(cache_file):
        print(f"loading embedding matrix: {cache_file}")
        embedding_matrix = pickle.load(open(cache_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), word_dim)).astype('float32')
        word_vec = _load_wordvec(embed_file, word_dim, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(cache_file, 'wb'))
    return embedding_matrix


def build_tokenizer(fnames, bert_name):
    if bert_name is not None:
        return Tokenizer(vocab=None, lower=None, bert_name=bert_name)
    cache_file = os.path.join('dats', 'tokenizer.dat')
    if os.path.exists(cache_file):
        print(f"loading tokenizer: {cache_file}")
        tokenizer = pickle.load(open(cache_file, 'rb'))
    else:
        print('building tokenizer...')
        tokenizer = Tokenizer.from_files(fnames=fnames)
        pickle.dump(tokenizer, open(cache_file, 'wb'))
    return tokenizer


def load_data(batch_size, bert_name=None):
    tokenizer = build_tokenizer(fnames=['train.json', 'dev.json'], bert_name=bert_name)
    if bert_name is None:
        embedding_matrix = build_embedding_matrix(tokenizer.vocab)
    else:
        embedding_matrix = None
    trainset = ToxicDataset('train.json', tokenizer, split='train', bert_name=bert_name)
    devset = ToxicDataset('dev.json', tokenizer, split='dev', bert_name=bert_name)
    testset = ToxicDataset('test.json', tokenizer, split='test', bert_name=bert_name)
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    dev_dataloader = DataLoader(devset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_dataloader, dev_dataloader, test_dataloader, tokenizer, embedding_matrix
