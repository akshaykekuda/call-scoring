# -*- coding: utf-8 -*-
"""DataLoader_fns

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t2vWLuHjJ9f2XaKZ75UHjFK3AWDiJv2M
"""

import torch
from torchtext.data.utils import get_tokenizer
word_tokenizer = get_tokenizer('basic_english')

class Collate:
    def __init__(self, vocab, device):
        self.vocab = vocab
        self.device = device
        self.count = 0

    def pad_trans(self, trans, max_len):
        num_sents = len(trans)
        trans_pos_indices = [i+1 for i in range(num_sents)]
        for i in range(max_len - num_sents):
            trans.append('<pad>')
            trans_pos_indices.append(0)
        return trans, trans_pos_indices

    def get_indices(self, sentence, max_sent_len):
        tokens = word_tokenizer(sentence)
        indices = [self.vocab[token] for token in tokens]
        diff = max_sent_len - len(tokens)
        positional_indices = [i + 1 for i in range(len(tokens))]
        for i in range(diff):
            indices.append(self.vocab['<pad>'])  # padding idx=1
            positional_indices.append(0)
        return indices, positional_indices

    def collate(self, batch):
        max_num_sents = 0
        max_sent_len = 0
        trans_len = []
        for i, sample in enumerate(batch):
            sent_len = []
            trans = sample['text']
            for sent in trans:
                l = len(word_tokenizer(sent))
                sent_len.append(l)
                if l > max_sent_len:
                    max_sent_len = l
            num_sents = len(trans)
            if num_sents > max_num_sents:
                max_num_sents = num_sents
            trans_len.append(sent_len)

        for sample in batch:
            trans = sample['text']
            pad_trans, sample['trans_pos_indices'] = self.pad_trans(trans, max_num_sents)
            sample['indices'] = []
            sample['word_pos_indices'] = []
            for sent in pad_trans:
                indices, positional_indices = self.get_indices(sent, max_sent_len)
                sample['indices'].append(indices)
                sample['word_pos_indices'].append(positional_indices)

        batch_dict = {'text': [], 'indices': [], 'scores': [], 'id': [],
                      'trans_pos_indices': [], 'word_pos_indices': []}
        for sample in batch:
            batch_dict['text'].append(sample['text'])
            batch_dict['indices'].append(sample['indices'])
            batch_dict['scores'].append(sample['scores'])
            batch_dict['id'].append(sample['id'])
            batch_dict['trans_pos_indices'].append(sample['trans_pos_indices'])
            batch_dict['word_pos_indices'].append(sample['word_pos_indices'])

        batch_dict['indices'] = torch.tensor(batch_dict['indices'], device=self.device)
        batch_dict['trans_pos_indices'] = torch.tensor(batch_dict['trans_pos_indices'], device=self.device)
        batch_dict['word_pos_indices'] = torch.tensor(batch_dict['word_pos_indices'], device=self.device)
        batch_dict['lens'] = trans_len

        return batch_dict

