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
import nltk
from nltk.tokenize import sent_tokenize
from Preprocessing import preprocess_transcript
import pickle

scoring_criteria = ['Greeting', 'Professionalism', 'Confidence',
                    'Cross Selling', 'Retention', 'Creates Incentive', 'Product Knowledge',
                    'Documentation', 'Education', 'Processes', 'Category', 'CombinedPercentileScore']


class CallDataset(Dataset):
    """Call transcript dataset."""

    def __init__(self, df):
        """
        Args:
            file_name: The json file to make the dataset from
        """
        self.df = df
        word_tokenizer = get_tokenizer('basic_english')
        clean_files = []
        for f in df.file_name:
            clean_files.append(preprocess_transcript(f))

        counter = Counter()

        # Build vocab from transcripts
        for transcript in clean_files:
            for i in range(len(transcript)):
                words = word_tokenizer(transcript[i])
                counter.update(words)

        self.vocab = vocab(counter)
        self.vocab.insert_token('<pad>', 0)
        self.vocab.insert_token('<UNK>', 0)
        self.vocab.set_default_index(0)
        self.df['text'] = clean_files

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = self.df.iloc[idx]['text']
        id = self.df.index[idx]
        scores = self.df.iloc[idx][scoring_criteria]
        sample = {'text': text, 'id': id, 'scores': scores}
        return sample

    def get_vocab(self):
        return self.vocab

    def save_vocab(self, path):
        output = open(path, 'wb')
        pickle.dump(self.vocab, output)
        output.close()
