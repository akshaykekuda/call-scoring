# -*- coding: utf-8 -*-
"""Inference_fns

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gtEyI0DrfZVDxgsIGqETSfGBeIPuvqU4
"""
import pandas as pd
from tqdm import tqdm
import torch
from Preprocessing import preprocess_transcript
from torchtext.data.utils import get_tokenizer
from DataLoader_fns import get_indices
from sklearn.metrics import classification_report, f1_score, mean_squared_error
from PrepareDf import prepare_baseline_df

def get_metrics(dataloader, encoder, scoring_criterion, type):
    pred_arr = []
    target_arr =[]
    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            output, scores = encoder(batch['indices'])
            if type == 'mse':
                pred = output.numpy()
            elif type == 'bi_class':
                pred = torch.argmax(output, dim=1).numpy()
            target = [sample[scoring_criterion] for sample in batch['scores']]
            pred_arr.extend(pred)
            target_arr.extend(target)
    encoder.train()
    print("Sample predictions")
    print("target:", target_arr[:10])
    print("prediction:", pred_arr[:10])
    if type == 'mse':
        mse_error = mean_squared_error(target_arr, pred_arr)
        print("MSE Error = {}".format(mse_error))
        return mse_error

    elif type == 'bi_class':
        f1 = f1_score(target_arr, pred_arr)
        clr = classification_report(target_arr, pred_arr)
        print("F1: {}".format(f1))
        print(clr)
        metrics = {'f1': f1, 'clr': clr}
        return metrics


def predict_baseline_metrics(test_df, type):
    cs_df, sales_df = prepare_baseline_df()
    cs_df = cs_df[~cs_df.index.duplicated(keep='first')]
    sales_df = sales_df[~sales_df.index.duplicated(keep='first')]
    df = pd.DataFrame(columns=['Category', 'Predicted Category'])
    if type == 'bi_class':
      for call_id in test_df.index:
        if test_df.loc[call_id, 'WorkgroupQueue'] == 'Sales':
          if call_id in sales_df.index:
            df.loc[call_id, 'Predicted Category'] = int(sales_df.loc[call_id, 'score'] < 0.25)
            df.loc[call_id, 'Category'] = test_df.loc[call_id, 'Category']
        elif test_df.loc[call_id, 'WorkgroupQueue'] == 'Customer Service':
          if call_id in cs_df.index:
            df.loc[call_id, 'Predicted Category'] = int(cs_df.loc[call_id, 'score'] < 0.25)
            df.loc[call_id, 'Category'] = test_df.loc[call_id, 'Category']
      clr = classification_report(df['Category'].tolist(), df['Predicted Category'].tolist())
      print(clr)
      return clr

    elif type == 'mse':
      for call_id in test_df.index:
        if test_df.loc[call_id, 'WorkgroupQueue'] == 'Sales':
          if call_id in sales_df.index:
            df.loc[call_id, 'Predicted Score'] = (1 - sales_df.loc[call_id, 'score'])*100
            df.loc[call_id, 'Overall Score'] = test_df.loc[call_id, 'CombinedPercentileScore']
        elif test_df.loc[call_id, 'WorkgroupQueue'] == 'Customer Service':
          if call_id in cs_df.index:
            df.loc[call_id, 'Predicted Score'] = (1 - cs_df.loc[call_id, 'score'])*100
            df.loc[call_id, 'Overall Score'] = test_df.loc[call_id, 'CombinedPercentileScore']
      mse_error = mean_squared_error(df['Overall Score'].tolist(), df['Predicted Score'].tolist())
      print("MSE Error = {}".format(mse_error))
      return mse_error


def predict(transcript, encoder, vocab):
  
  word_tokenizer = get_tokenizer('basic_english')
  sents = preprocess_transcript(transcript)
  max_len = 0
  for sent in sents:
    tokens = word_tokenizer(sent)
    if len(tokens) > max_len:
      max_len = len(tokens)

  indices = []
  for sent in sents:
    indices.append(get_indices(sent, max_len, vocab))
  indices = torch.tensor(indices)

  indices = indices.unsqueeze_(0)

  output, hidden = encoder(indices)
  output = output[:, -1, :]

  output = classifier(output)
  classification = torch.argmax(output).item()

  return classification


def get_regression_metrics(dataloader, encoder, scoring_criterion):
  pred_arr = []
  target_arr =[]
  encoder.eval()
  total_correct = 0
  batch_size = dataloader.batch_size
  with torch.no_grad():
    for batch in tqdm(dataloader):
      output, scores = encoder(batch['indices'])
      for i in range(output.shape[0]):
        pred = output[i].item()
        target = batch['scores'][i][scoring_criterion]
        pred_arr.append(pred)
        target_arr.append(target)
  encoder.train()
  mse_error = mean_squared_error(target_arr, pred_arr)
  print("MSE Error = {}".format(mse_error))
  return mse_error

def predict_baseline_mse(test_df):
    cs_df, sales_df = prepare_baseline_df()
    cs_df = cs_df[~cs_df.index.duplicated(keep='first')]
    sales_df = sales_df[~sales_df.index.duplicated(keep='first')]
    df = pd.DataFrame(columns=['Overall Score', 'Predicted Score'])
    for call_id in test_df.index:
      if test_df.loc[call_id, 'WorkgroupQueue'] == 'Sales':
        if call_id in sales_df.index:
          df.loc[call_id, 'Predicted Score'] = 1 - sales_df.loc[call_id, 'score']
          df.loc[call_id, 'Overall Score'] = test_df.loc[call_id, 'CombinedPercentileScore']/100
      elif test_df.loc[call_id, 'WorkgroupQueue'] == 'Customer Service':
        if call_id in cs_df.index:
          df.loc[call_id, 'Predicted Score'] = 1- cs_df.loc[call_id, 'score']
          df.loc[call_id, 'Overall Score'] = test_df.loc[call_id, 'CombinedPercentileScore']/100
    mse_error = mean_squared_error(df['Overall Score'].tolist(), df['Predicted Score'].tolist())
    print("MSE Error = {}".format(mse_error))
    return mse_error

def get_accuracy(dataloader, encoder, classifier):
  total_correct = 0

  for batch in tqdm(dataloader):

    batch_size = len(batch['indices'])

    output, hidden = encoder(batch['indices'])
    output = output[:,-1,:]

    output = classifier(output)

    for i in range(batch_size):

      classification = torch.argmax(output[i]).item()
      target = batch['category'][i]
      if target == classification:
        total_correct+=1

  acc = total_correct/(len(dataloader) * batch_size)
  print("Accuracy: {}".format(acc))
  return acc