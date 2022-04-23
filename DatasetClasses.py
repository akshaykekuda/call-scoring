# -*- coding: utf-8 -*-
"""DatasetClasses

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WGuilwMSn2xsfnKuw4yy27BkIlL_8A-h
"""

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import torch
from collections import Counter
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from Preprocessing import preprocess_transcript
import pickle

# scoring_criteria = ['Greeting', 'Professionalism', 'Confidence',
#                     'Cross Selling', 'Retention', 'Creates Incentive', 'Product Knowledge',
#                     'Documentation', 'Education', 'Processes', 'Category', 'CombinedPercentileScore']


class CallDataset(Dataset):
    """Call transcript dataset."""

    def __init__(self, df, scoring_criteria):
        """
        Args:
            file_name: The json file to make the dataset from
        """
        self.df = df
        self.scoring_criteria = scoring_criteria
        self.fbk_str = [criterion + " fbk_vector" for criterion in scoring_criteria]

        word_tokenizer = get_tokenizer('basic_english')
        counter = Counter()
        # Build vocab from transcripts
        for transcript in df.text:
            for i in range(len(transcript)):
                words = word_tokenizer(transcript[i])
                counter.update(words)

        self.vocab = vocab(counter)
        self.vocab.insert_token('<cls>', 0)
        self.vocab.insert_token('<pad>', 0)
        self.vocab.insert_token('<UNK>', 0)
        self.vocab.set_default_index(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = self.df.iloc[idx]['text']
        id = self.df.index[idx]
        scores = self.df.iloc[idx][self.scoring_criteria]
        sample = {'text': text, 'id': id, 'scores': scores}
        # if self.use_feedback:
        #     fbk_str = [criterion + " fbk_vector" for criterion in scoring_criteria]
        #     fbk_vector = self.df.iloc[idx][fbk_str]
        #     sample['fbk_vector'] = fbk_vector
        return sample

    def get_vocab(self):
        return self.vocab

    def save_vocab(self, path):
        output = open(path, 'wb')
        pickle.dump(self.vocab, output)
        output.close()


class CallDatasetWithFbk(CallDataset):
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = self.df.iloc[idx]['text']
        id = self.df.index[idx]
        scores = self.df.iloc[idx][self.scoring_criteria]
        sample = {'text': text, 'id': id, 'scores': scores}
        fbk_vector = self.df.iloc[idx][self.fbk_str]
        # return series
        for fbk_category in self.fbk_str:
            sample[fbk_category] = fbk_vector[fbk_category]
        return sample

class InferenceCallDataSet(Dataset):
    def __init__(self, df) -> None:
        super().__init__()
        self.df = df
        self.df['text'] = self.df.file_name.apply(lambda x: preprocess_transcript(x))
    def __len__(self):
        return len(self.df)


class MLMDataSet(Dataset):
    def __init__(self, paths, tokenizer):
        self.sentences = []
        for file in paths:
            with open(file, 'r') as f:
                sent = f.readlines()
                self.sentences.extend([line.strip('\n') for line in sent])
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, idx):
        return self.tokenizer(self.sentences[idx])
