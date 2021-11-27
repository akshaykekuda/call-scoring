# -*- coding: utf-8 -*-
"""MainCalls

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13U7IVc0TcKMzzszxaQt-jxqM-2UEAvah
"""
import argparse
import datetime
import sys
import os

from DatasetClasses import CallDataset, CallDatasetWithFbk
from torch.utils.data import DataLoader
from DataLoader_fns import Collate
from FeedbackComments import FeedbackComments
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.models import KeyedVectors
from TrainModel import TrainModel
from Inference_fns import *
from PrepareDf import *
from sklearn.model_selection import train_test_split, KFold
from nltk.tokenize import word_tokenize

import numpy as np
import torch
import pandas as pd
import time
np.random.seed(0)
torch.manual_seed(0)

path_to_handscored_p = 'ScoringDetail_viw_all_subscore.p' 
word_embedding_pt = dict(glove='../word_embeddings/glove_word_vectors',
                         w2v='../word_embeddings/custom_w2v_100d',
                         fasttext='../word_embeddings/fasttext_300d.bin')
pd.set_option("display.max_rows", None, "display.max_columns", None)

global vocab
sub_score_categories = ['Cross Selling', 'Creates Incentive', 'Product Knowledge', 'Education', 'Processes', 'Greeting', 'Professionalism', 'Confidence',  'Retention',
                        'Documentation']

def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='MainCalls.py')

    # General system running and configuration options
    parser.add_argument('--model', type=str, default='AllSubScores', help='model to run')
    parser.add_argument('--workgroup', type=str, default='all', help='workgroup of calls to score')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--epochs', type=int, default=1, help='epochs to run')
    parser.add_argument('--train_samples', type=int, default=50, help='number of samples for training')
    parser.add_argument('--word_embedding', type=str, default='glove', help='word embedding to use')
    parser.add_argument('--attention', type=str, default='hsan', help='attention mechanism to use')
    parser.add_argument('--save_path', type=str, default='logs/test/', help='path to save checkpoints')
    parser.add_argument('--num_heads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--model_size', type=int, default=64, help='model size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--trans_path', type=str, default='transcriptions/text_only/', help='link to transcripts')
    parser.add_argument('--device', type=str, default='cpu', help='device to use')
    parser.add_argument('--loss', type=str, default='bce', help='optimizer to use')
    parser.add_argument('--k', type=int, default='20', help='number of top comments to use')
    parser.add_argument('--use_feedback', type=bool, default=False, help='use feedback comments')

    args = parser.parse_args()
    return args


def predict_baseline(test_df):
    print('Test Baseline Model Metrics:')
    predict_baseline_metrics(test_df, type='bi_class')


def predict_all_subscores(trainer, dataloader_transcripts_test):
    scoring_criteria = ['Cross Selling', 'Creates Incentive', 'Product Knowledge', 'Education', 'Processes']
    if args.loss == 'bce':
        model = trainer.train_multi_label_model(scoring_criteria)
    else:
        raise ValueError("Cannot use the {} for Multi Label Classification".format(args.loss))

    torch.save(model.state_dict(), args.save_path+"call_encoder_bi_class.model")
    print('Test Metrics for Call Transcripts dataset  is:')
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criteria, optim=args.loss)
    plot_roc(scoring_criteria, pred_df, args.save_path+'auc.png')
    pred_df.to_pickle(args.save_path+'all_subscores_pred_test.p')
    return metrics


def predict_product_knowledge(trainer, dataloader_transcripts_test):
    scoring_criterion = ['Product Knowledge']
    if args.loss == 'cel':
        model = trainer.train_biclass_model(scoring_criterion)
    elif args.loss == 'bce':
        model = trainer.train_multi_label_model(scoring_criterion)
    else:
        raise ValueError("Cannot use the {} for Binary Classification".format(args.loss))

    torch.save(model.state_dict(), args.save_path+"product_knowledge.model")
    print('Test Metrics for Product Knowledge  is:')
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criterion, optim=args.loss)
    plot_roc(scoring_criterion, pred_df, args.save_path+'auc.png')
    pred_df.to_pickle(args.save_path+'prod_knw_pred_test.p')
    return metrics


def predict_cross_selling(trainer, dataloader_transcripts_test):
    scoring_criterion = ['Cross Selling']
    if args.loss == 'cel':
        model = trainer.train_biclass_model(scoring_criterion)
    elif args.loss == 'bce':
        model = trainer.train_multi_label_model(scoring_criterion)
    else:
        raise "Cannot use the {} for Binary Classification".format(args.loss)
    torch.save(model.state_dict(), args.save_path+"cross_selling.model")
    print('Test Metrics for Call Transcripts dataset  is:')
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criterion, optim=args.loss)
    plot_roc(scoring_criterion, pred_df, args.save_path+'auc.png')
    pred_df.to_pickle(args.save_path+'cross_selling_pred_test.p')
    return metrics


def predict_overall_category(trainer, dataloader_transcripts_test):
    scoring_criterion = ['Category']
    if args.loss == 'cel':
        model = trainer.train_biclass_model(scoring_criterion)
    elif args.loss == 'bce':
        model = trainer.train_multi_label_model(scoring_criterion)
    else:
        raise ValueError("Cannot use the {} for Binary Classification".format(args.loss))
    torch.save(model.state_dict(), args.save_path+"overall_category.model")
    print('Test Metrics for Call Transcripts dataset  is:')
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criterion, optim=args.loss)
    plot_roc(scoring_criterion, pred_df, args.save_path+'auc.png')
    pred_df.to_pickle(args.save_path+'overall_cat_pred_test.p')
    return metrics


def predict_overall_score(trainer, dataloader_transcripts_test):
    scoring_criterion = ["CombinedPercentileScore"]
    if args.loss == 'mse':
        model = trainer.train_linear_regressor(scoring_criterion)
    else:
        raise ValueError("Cannot use the {} for Regression".format(args.loss))
    torch.save(model.state_dict(), "overall_score.model")
    print('Test MSE for Call Transcripts dataset  is:')
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criterion, optim=args.loss)
    pred_df.to_pickle(args.save_path+'overall_score_pred_test.p')
    print('Test MSE for Call Transcripts dataset  is:')
    return metrics


def predict_scores(trainer, dataloader_transcripts_test):
    if len(scoring_criteria) == 1:
        if scoring_criteria[0] == 'CombinedPercentileScore':
            model = trainer.train_linear_regressor()
        elif args.loss == 'cel':
            model = trainer.train_biclass_model()
        elif args.loss == 'bce':
            model = trainer.train_multi_label_model()
        else:
            raise ValueError("cannot run training")
    else:
        model = trainer.train_multi_label_model()
    torch.save(model.state_dict(), args.save_path+"call_score.model")
    print('Test Metrics for Call Transcripts dataset  is:')
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criteria, loss=args.loss)
    plot_roc(scoring_criteria, pred_df, args.save_path+'auc.png')
    pred_df.to_pickle(args.save_path+'call_score_pred_test.p')
    return metrics


def predict_scores_mtl(trainer, dataloader_transcripts_test):
    model = trainer.train_mtl_model()
    torch.save(model.state_dict(), args.save_path+"call_score_mtl.model")
    print('Test Metrics for Call Transcripts dataset  is:')
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criteria, loss=args.loss)
    plot_roc(scoring_criteria, pred_df, args.save_path+'auc.png')
    pred_df.to_pickle(args.save_path+'overall_cat_pred_test.p')
    return metrics


def get_max_len(df):
    def fun(sent):
        return len(sent.split())
    max_trans_len = np.max(df.text.apply(lambda x: len(x.split("\n"))))
    max_sent_len = np.max(df.text.apply(lambda x: max(map(fun, x.split('\n')))))
    return max_trans_len, max_sent_len


def run_cross_validation(train_df, test_df):
    kf = KFold(n_splits=5, shuffle=True)
    kfold_results = []
    for train, dev in kf.split(train_df):
        # train, dev = train_test_split(df_sampled, test_size=0.20)
        t_df = train_df.iloc[train].copy()
        # t_df = t_df.loc[t_df.text.apply(lambda x: len(x.split('\n'))).sort_values().index]
        dev_df = train_df.iloc[dev].copy()
        subscore_dist = t_df.loc[:, ['Greeting', 'Professionalism', 'Confidence',
                    'Cross Selling', 'Retention', 'Creates Incentive', 'Product Knowledge',
                    'Documentation', 'Education', 'Processes', 'Category']].apply(lambda x: x.value_counts())
        print("Subscore distribution count in Training set\n", subscore_dist)
        if args.use_feedback:
            dataset_transcripts_train = CallDatasetWithFbk(t_df, scoring_criteria)
        else:
            dataset_transcripts_train = CallDataset(t_df, scoring_criteria)
        dataset_transcripts_dev = CallDataset(dev_df,scoring_criteria)
        dataset_transcripts_test = CallDataset(test_df, scoring_criteria)

        max_trans_len, max_sent_len = get_max_len(train_df)
        vocab = dataset_transcripts_train.get_vocab()
        dataset_transcripts_train.save_vocab('vocab')

        batch_size = args.batch_size
        c = Collate(vocab, args.device)
        dataloader_transcripts_train = DataLoader(dataset_transcripts_train, batch_size=batch_size, shuffle=True,
                                                  num_workers=0, collate_fn=c.collate)
        dataloader_transcripts_dev = DataLoader(dataset_transcripts_dev, batch_size=batch_size, shuffle=False,
                                                num_workers=0, collate_fn=c.collate)
        dataloader_transcripts_test = DataLoader(dataset_transcripts_test, batch_size=batch_size, shuffle=False,
                                                 num_workers=0, collate_fn=c.collate)

        embedding_model = KeyedVectors.load(word_embedding_pt[args.word_embedding], mmap='r')
        vocab_size = len(vocab)
        embedding_size = embedding_model.vector_size
        weights_matrix = np.zeros((vocab_size, embedding_size))
        i = 2
        for word in vocab.get_itos()[2:]:
            try:
                weights_matrix[i] = embedding_model[word]  # model.wv[word] for trained word2vec
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_size,))
            i += 1
        weights_matrix[0] = np.mean(weights_matrix, axis=0)
        weights_matrix = torch.tensor(weights_matrix)
        trainer = TrainModel(dataloader_transcripts_train, dataloader_transcripts_dev, vocab_size, embedding_size,
                             weights_matrix, args, max_trans_len, max_sent_len, scoring_criteria)
        if args.use_feedback:
            metrics = predict_scores_mtl(trainer, dataloader_transcripts_test)
        else:
            metrics = predict_scores(trainer, dataloader_transcripts_test)
        """    
        if args.model == 'Category':
            metrics = predict_overall_category(trainer, dataloader_transcripts_test)
        elif args.model == 'CombinedPercentileScore':
            metrics = predict_overall_score(trainer, dataloader_transcripts_test)
        elif args.model == 'AllSubScores':
            metrics = predict_all_subscores(trainer, dataloader_transcripts_test)
        elif args.model == 'Cross_Selling':
            metrics = predict_cross_selling(trainer, dataloader_transcripts_test)
        elif args.model == 'Product_Knowledge':
            metrics = predict_product_knowledge(trainer, dataloader_transcripts_test)
        elif args.model == 'MultiTask':
            metrics = predict_scores_mtl(trainer, dataloader_transcripts_test)
        else:
            raise ValueError("Invalid Model Argument")
        """
        kfold_results.append(metrics)
        break
    return kfold_results


if __name__ == "__main__":
    args = _parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Arguments:", args)
    score_df, q_text = prepare_score_df(
        path_to_handscored_p, workgroup=args.workgroup)
    transcript_score_df = prepare_trancript_score_df(score_df, q_text, args.trans_path)
    train_df, test_df = train_test_split(transcript_score_df, test_size=0.15)
    if args.model == 'AllSubScores':
        scoring_criteria = sub_score_categories
    elif args.model == 'BestSubScores':
        scoring_criteria = sub_score_categories[:5]
    else:
        scoring_criteria = [args.model]
    if args.use_feedback:
        comment_obj = FeedbackComments(train_df)
        top_k_comments = comment_obj.extract_top_k_comments(args.k, scoring_criteria)
        train_df = comment_obj.df
    if args.train_samples > 0:
        train_df = balance_df(train_df, args.train_samples)

    kfold_results = run_cross_validation(train_df, test_df)

    # avg_tuple = [sum(y) / len(y) for y in zip(*kfold_results)]
    # print("Overall accuracy={} Overall F1 score={}".format(avg_tuple[0], avg_tuple[1]))
