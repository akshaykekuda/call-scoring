# -*- coding: utf-8 -*-
"""Copy of Yelp_Experiment_Vocab_Embeddings_Experiment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16SujvXDehWuiOh4z_7QbPNQG7QP2IR3B

# Preprocessing
"""

# from google.colab import files

# uploaded = files.upload()

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import brown
nltk.download('punkt')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
from sklearn.model_selection import train_test_split
#np.random.seed(0)
#torch.manual_seed(0)

from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.vocab import GloVe
from Preprocessing import preprocess_transcript

word_tokenizer = get_tokenizer('basic_english')

class YelpDataset(Dataset):
    """Yelp dataset."""

    def __init__(self, file_name):
        """
        Args:
            file_name: The json file to make the dataset from
        """
        self.df = pd.read_json(file_name, lines=True)

        binary_cat = []
        counter = Counter()
        reviews = []

        #Create target class for each review, build vocab
        for index, row in self.df.iterrows():
            binary_cat.append(row['category'])

            sentences = sent_tokenize(row['text'])
            reviews.append(sentences)
            for i in range(len(sentences)):
              words = word_tokenizer(sentences[i])
              counter.update(words)

        self.vocab = Vocab(counter, min_freq=1)
        self.df['category'] = binary_cat
        self.df['text'] = reviews
        


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        category = self.df.iloc[idx, 0]
        text = self.df.iloc[idx, 1]
        sample = {'category': category, 'text': text}

        return sample

    def get_vocab(self):
      return self.vocab

class CallDataset(Dataset):
    """Call transcript dataset."""

    def __init__(self, files, classifications):
        """
        Args:
            file_name: The json file to make the dataset from
        """
        self.df = pd.DataFrame()
        word_tokenizer = get_tokenizer('basic_english')
        
        clean_files = []
        for f in files:
          clean_files.append(preprocess_transcript(f))

        counter = Counter()

        #Build vocab from transcripts
        for transcript in clean_files:
          for i in range(len(transcript)):
            words = word_tokenizer(transcript[i])
            counter.update(words)

        self.vocab = Vocab(counter)
        self.df['category'] = classifications
        self.df['text'] = clean_files

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        category = self.df.iloc[idx, 0]
        text = self.df.iloc[idx, 1]
        sample = {'category': category, 'text': text}

        return sample

    def get_vocab(self):
        return self.vocab

df = pd.read_pickle("/home/kekuak/ScoringDetail_viw_subscore.p")
df = df.sort_values(by= ['RecordingDate', 'QGroupSequence', 'QuestionSequence']).copy()
cols = ['QGroupSequence', 'QGroupName','InteractionIdKey', 'QuestionSequence', 'QuestionText', 'QuestionType', 
        'QuestionPromptType', 'QuestionWeight', 'QuestionMin', 'QuestionMax', 'AnswerScore', 'RawAnswer', 'DisplayAnswer', 
        'UserComments']
calls_df = df[(df.QuestionnaireName == 'Call Interaction') & (df['RecordingDate'] >= pd.Timestamp(2021,1,1,0))].copy()
q_df = calls_df[cols]
temp = q_df[0:10]
temp = temp.reset_index(drop=True)
q_text =[]
for index, row in temp.iterrows():
    q_text.append(row['QuestionText'])
score_df = pd.DataFrame()

for row in calls_df.itertuples():
    score_df.loc[row.InteractionIdKey,row.QuestionText] = row.AnswerScore
    score_df.loc[row.InteractionIdKey,'CombinedPercentileScore'] = row.CombinedPercentileScore
    if (row.QuestionText == 'Professionalism') and (row.AnswerScore == 5):
        score_df.loc[row.InteractionIdKey,'Category'] = 1
    elif (row.QuestionText == 'Professionalism') and (row.AnswerScore == 0):
        score_df.loc[row.InteractionIdKey,'Category'] = 0
score_df.Category = score_df.Category.astype('int64')

print("Dataframe creation done")
"""
score_comment_df = pd.DataFrame()
for row in calls_df.itertuples():
    score_comment_df.loc[row.InteractionIdKey,row.QuestionText] = row.UserComments
    score_comment_df.loc[row.InteractionIdKey,'CombinedPercentileScore'] = row.CombinedPercentileScore
    score_comment_df.loc[row.InteractionIdKey,'Category'] = score_df.loc[row.InteractionIdKey,'Category']
"""
dir = "/mnt/transcriber/manual_score_transcriber/transcriptions/output"
#dir = '/mnt/transcriber/manual_score_transcriber/transcriptions/output_mono_small'
df = pd.DataFrame(columns=['text', 'file_name'])
for file in os.listdir(dir):
    #if file.endswith('.txt') and file.startswith('r'):
    if file.endswith('.txt'):
        file_loc = dir + '//' + file
        f = open(file_loc, 'r')
        tscpt = f.read()
        f.close()
        arr = file.split("_")
        id = arr[1].split('.')[0]
        if id in score_df.index:
                df.loc[id, score_df.columns] = score_df.loc[id]
                df.loc[id, ['text', 'file_name']] = [tscpt, file_loc]
"""
for t in df.itertuples():
    idx = t[0]
    id = t[1]
    if id in score_df.index:
        df.loc[idx, score_df.columns] = score_df.loc[id]
"""
#df = df.dropna()
h = len(df[df.Professionalism == 5])
temp = df[df.Professionalism == 5].sample(h -200).index
df_sampled = df.drop(temp)
print('training and dev sample size = {}'.format(len(df_sampled)))
print('test sample size = {}'.format(len(df.loc[temp])))
remaining = df.index.difference(temp, sort=False)

train, dev = train_test_split(df_sampled, test_size=0.20)
#dev, test = train_test_split(test, test_size=0.50)
filenames_train = train.file_name
classifications_train = train.Category
filenames_dev = dev.file_name
classifications_dev = dev.Category
filenames_test = df.loc[temp].file_name
classifications_test = df.loc[temp].Category


### DO NOT APPEND ZEROS ###

dataset_transcripts_train = CallDataset(filenames_train, classifications_train)
dataset_transcripts_dev = CallDataset(filenames_dev, classifications_dev)
dataset_transcripts_test = CallDataset(filenames_test, classifications_test)


vocab = dataset_transcripts_train.get_vocab()

def get_indices(sentence, max_sent_len):
  tokens = word_tokenizer(sentence)
  indices = [vocab[token] for token in tokens]
  diff = max_sent_len - len(tokens)
  for i in range(diff):
    indices.append(1)
  return indices

def collate(batch):

  max_num_sents = 0
  max_sent_len = 0
  for sample in batch:
    num_sents = len(sample['text'])
    if num_sents > max_num_sents:
      max_num_sents = num_sents
    for sent in sample['text']:
      if len(word_tokenizer(sent)) > max_sent_len:
        max_sent_len = len(word_tokenizer(sent))
  
  for sample in batch:
    sample['text'] = pad_review(sample['text'], max_num_sents)
    sample['indices']= []
    for sent in sample['text']:
      sample['indices'].append(get_indices(sent, max_sent_len))

  batch_dict = {'text': [], 'indices': [], 'category': []}
  for sample in batch:
    batch_dict['text'].append(sample['text'])
    batch_dict['indices'].append(sample['indices'])
    batch_dict['category'].append(sample['category'])
  batch_dict['indices'] = torch.tensor(batch_dict['indices'])
  batch_dict['category'] = torch.tensor(batch_dict['category'], dtype = torch.long)

  return batch_dict


def pad_review(review, max_len):
  num_sents = len(review)
  for i in range(max_len - num_sents):
    review.append('<pad>')
  return review

batch_size = 1
dataloader_train = DataLoader(dataset_transcripts_train, batch_size=batch_size, shuffle=True, 
                              num_workers=0, collate_fn = collate)
dataloader_dev = DataLoader(dataset_transcripts_dev, batch_size=batch_size, shuffle=True, 
                              num_workers=0, collate_fn = collate)
dataloader_test = DataLoader(dataset_transcripts_test, batch_size=batch_size, shuffle=True, 
                              num_workers=0, collate_fn = collate)

count = 0
for data in enumerate(dataloader_dev):
        t = data[1]['category']
        count = count + t.size()[0] - np.count_nonzero(t)
print("0 category count={}".format(count))

"""# Updated Model"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, weights_matrix):
        super(EncoderRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=1)
        self.embedding.load_state_dict({'weight': weights_matrix})

        self.hidden_size = hidden_size
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs):

        embed_output = self.embedding(inputs)
        embed_output = torch.mean(embed_output, dim=2, keepdim=True).squeeze(2)
        output, hidden = self.gru(embed_output)

        return output, hidden

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.input_size = input_size
        
        self.fcn = nn.Sequential(
            nn.Linear(2*input_size, 10),
            nn.Tanh(),
            nn.Linear(10, 2),
            nn.Tanh()
        )


    def forward(self, x):
        output = self.fcn(x)
        
        return output

class GRUAttention(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, weights_matrix):
        super(GRUAttention, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=1)
        self.embedding.load_state_dict({'weight': weights_matrix})

        self.hidden_size = hidden_size
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=True)


        self.attn = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

        self.fcn = nn.Sequential(
            nn.Linear(2*hidden_size, 64),
            nn.Tanh(),
            nn.Dropout(0.8),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def forward(self, inputs):
        embed_output = self.embedding(inputs)
        embed_output = torch.mean(embed_output, dim=2, keepdim=True).squeeze(2)
        output, hidden = self.gru(embed_output)
        attn_weights = self.attn(output)
        attn_scores = F.softmax(attn_weights, 1)
        out = torch.bmm(output.transpose(1, 2), attn_scores).squeeze(2)
        logits = self.fcn(out)
        return logits, attn_scores.squeeze(2)



## Make weights matrix
vec_size = 300
vocab = dataset_transcripts_train.get_vocab()
vocab_size = len(vocab)

glove = Word2VecKeyedVectors.load_word2vec_format('glove.w2v.txt')

weights_matrix = np.zeros((vocab_size, vec_size))
i = 0
for word in vocab.itos:
  try:
    weights_matrix[i] = glove[word]
  except KeyError:
    weights_matrix[i] = np.random.normal(scale=0.6, size=(vec_size, ))
  i+=1
  
weights_matrix = torch.tensor(weights_matrix)

from tqdm import tqdm

encoder_output_size = 32
encoder = GRUAttention(vocab_size, vec_size, encoder_output_size, weights_matrix)
criterion = nn.CrossEntropyLoss()

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)

epochs = 10
total = 0
for n in range(epochs):
    epoch_loss = 0
    count = 0
    for batch in tqdm(dataloader_train):
        encoder.train()
        encoder.zero_grad()
        loss = 0

        output, scores = encoder(batch['indices'])
 
        target = batch['category']

        loss += criterion(output, target)
        epoch_loss+=loss.detach().item()
        loss.backward()
        # print("training loss: {}".format(loss))
        encoder_optimizer.step()
    if n:
        print("Average loss at epoch {}: {}".format(n, epoch_loss/len(dataloader_train)))


total_correct = 0
from sklearn.metrics import classification_report
classification_arr = []
target_arr =[]

encoder.eval()

for batch in tqdm(dataloader_train):

        output, scores = encoder(batch['indices'])
        
        for i in range(output.shape[0]):
          classification = torch.argmax(output[i]).item()
          target = batch['category'][i]
          classification_arr.append(classification)
          target_arr.append(target)
          if target == classification:
             total_correct+=1
"""
             try:
               print('\nTop 5 sentences of class {}'.format(target), file = open('scores.txt','a'))
               top_scores = sorted(enumerate(zip(batch['text'][0], scores.squeeze())), key=lambda x: x[1][1], reverse=True)[0:5]
              
               for idx, (a,b) in top_scores:
                 print(a,b.item(), file = open('scores.txt','a'))
 
             except ValueError:
               print(scores.shape)
               print(batch['text'])
"""
 
print("Training Accuracy: {}".format(total_correct/(len(dataloader_train) * batch_size)))
print(classification_report(target_arr, classification_arr))

total_correct = 0
classification_arr = []
target_arr =[]
print("\nDev evaluation start", file = open('scores.txt', 'a'))
for batch in tqdm(dataloader_dev):

        output, scores = encoder(batch['indices'])

        for i in range(output.shape[0]):
  
          classification = torch.argmax(output[i]).item()
          target = batch['category'][i]
          classification_arr.append(classification)
          target_arr.append(target)
          if target == classification:
             total_correct+=1
             print('\nTop 5 sentences of actual class={} predicted class={}'.format(target,classification), file = open('scores.txt','a'))
             top_scores = sorted(enumerate(zip(batch['text'][0], scores.squeeze())), key=lambda x: x[1][1], reverse=True)[0:5]
             for idx, (a,b) in top_scores:
                print(a,b.item(), file = open('scores.txt','a'))
 
print("Dev Accuracy: {}".format(total_correct/(len(dataloader_dev) * batch_size)))
print(classification_report(target_arr, classification_arr))

total_correct = 0
classification_arr = []
target_arr =[]

for batch in tqdm(dataloader_test):

        output, scores = encoder(batch['indices'])

        for i in range(output.shape[0]):
  
          classification = torch.argmax(output[i]).item()
          target = batch['category'][i]
          classification_arr.append(classification)
          target_arr.append(target)
          if target == classification:
             total_correct+=1

print("Test Accuracy: {}".format(total_correct/(len(dataloader_test) * batch_size)))
print(classification_report(target_arr, classification_arr))

torch.save(encoder, 'gru_attention.model')
