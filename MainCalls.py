# -*- coding: utf-8 -*-
"""MainCalls

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13U7IVc0TcKMzzszxaQt-jxqM-2UEAvah
"""
import argparse
import datetime
import sys
import os, json
from pathlib import Path
from DatasetClasses import CallDataset, CallDatasetWithFbk, MLMDataSet
from torch.utils.data import DataLoader,random_split
from DataLoader_fns import Collate
from FeedbackComments import FeedbackComments
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.models import KeyedVectors
from TrainModel import TrainModel
from Inference_fns import *
from PrepareDf import *
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from nltk.tokenize import word_tokenize
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling
import random
from tokenizers import BertWordPieceTokenizer
import numpy as np
import torch
import pandas as pd
from pathlib import Path

import time
seed = 1000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

path_to_handscored_p = 'ScoringDetail_viw_all_subscore.p'
dataset_dir = "datasets/"

word_embedding_pt = dict(glove='../word_embeddings/glove_word_vectors',
                         w2v='../word_embeddings/custom_w2v_100d',
                         fasttext='../word_embeddings/fasttext_300d.bin')
pd.set_option("display.max_rows", None, "display.max_columns", None)

global vocab
sub_score_categories = ['Cross Selling', 'Creates Incentive', 'Education', 'Processes', 'Product Knowledge', 'Greeting', 'Professionalism', 'Confidence',  'Retention',
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
    parser.add_argument('--attention', type=str, default='hs2an', help='attention mechanism to use')
    parser.add_argument('--save_path', type=str, default='logs/test/', help='path to save checkpoints')
    parser.add_argument('--word_nh', type=int, default=1, help='number of attention heads for word attn')
    parser.add_argument('--sent_nh', type=int, default=1, help='number of attention heads for sent attn')
    parser.add_argument('--model_size', type=int, default=64, help='model size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--trans_path', type=str, default='transcriptions/text_only/', help='link to transcripts')
    parser.add_argument('--test_path', type=str, default='transcriptions/test_transcriptions/', help='link to test transcripts')
    parser.add_argument('--device', type=str, default='cpu', help='device to use')
    parser.add_argument('--loss', type=str, default='bce', help='optimizer to use')
    parser.add_argument('--k', type=int, default=20, help='number of top comments to use')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataset')
    parser.add_argument("--save_model", default=False, action="store_true")
    parser.add_argument("--use_feedback", default=False, action="store_true")
    parser.add_argument("--new_data", default=False, action="store_true")
    parser.add_argument("--num_layers", default=1, type=int, help="num of layers of sentence level self attention")
    parser.add_argument("--word_nlayers", default=1, type=int, help="num of layers of word level self attention")
    parser.add_argument("--reg", default=1e-5, type=float, help="l2 regularization")
    parser.add_argument("--acum_step", default=1, type=int, help="grad accumulation steps")
    parser.add_argument("--tok_path", default='sa_tokenizer/', type=str, help="path to trained tokenizer")
    parser.add_argument("--doc2vec_pt", default='../word_embeddings/trained_doc2vec', type=str, help="path to trained doc2vec model")

    args = parser.parse_args()
    return args

"""
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
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criteria, loss=args.loss)
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
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criterion, loss=args.loss)
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
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criterion, loss=args.loss)
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
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criterion, loss=args.loss)
    plot_roc(scoring_criterion, pred_df, args.save_path+'auc.png')
    pred_df.to_pickle(args.save_path+'overall_cat_pred_test.p')
    return metrics


def predict_overall_score(trainer, dataloader_transcripts_test):
    scoring_criterion = ["CombinedPercentileScore"]
    if args.loss == 'mse':
        model = trainer.train_linear_regressor(scoring_criterion)
    else:
        raise ValueError("Cannot use the {} for Regression".format(args.loss))
    torch.save(model.state_dict(), 'overall_score.model')
    print('Test MSE for Call Transcripts dataset  is:')
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criterion, loss=args.loss)
    pred_df.to_pickle(args.save_path+'overall_score_pred_test.p')
    print('Test MSE for Call Transcripts dataset  is:')
    return metrics
"""
"""
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
    if args.save_model:
        torch.save(model, args.save_path+"call_score.model")
    print('Test Metrics for Call Transcripts dataset  is:')
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criteria, loss=args.loss)
    plot_roc(scoring_criteria, pred_df, args.save_path+'auc.png')
    pred_df.to_pickle(args.save_path+'call_score_test.p')
    return metrics
"""


def predict_scores(trainer, dataloader_transcripts_test):
    if args.loss == 'cel':
        model = trainer.train_cel_model()
    elif args.loss == 'bce':
        model = trainer.train_bce_model()
    elif args.loss == 'mse':
        model = trainer.train_linear_regressor()
    else:
        raise ValueError("Invalid loss function {}".format(args.loss))
    if args.save_model:
        torch.save(model, args.save_path+"call_score.model")
    print('Test Metrics for Call Transcripts dataset  is:')
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criteria, loss=args.loss)
    plot_roc(scoring_criteria, pred_df, args.save_path + 'fold_'+ str(trainer.fold) +'_auc.png')
    pred_df.to_pickle(args.save_path+'fold_' + str(trainer.fold)+'_call_score_test.p')
    return metrics


def predict_scores_mtl(trainer, dataloader_transcripts_test):
    model = trainer.train_mtl_model()
    if args.save_model:
        torch.save(model, args.save_path+"call_score_mtl.model")
    print('Test Metrics  for Call Transcripts dataset  is:')
    metrics, pred_df = get_metrics(dataloader_transcripts_test, model, scoring_criteria, loss=args.loss)
    plot_roc(scoring_criteria, pred_df, args.save_path+'auc.png')
    pred_df.to_pickle(args.save_path+'call_score_mtl_test.p')
    return metrics


def get_max_len(df):
    def fun(sent):
        return len(sent.split())
    max_trans_len = np.max(df.text.apply(lambda x: len(x.split("\n"))))
    max_sent_len = np.max(df.text.apply(lambda x: max(map(fun, x.split('\n')))))
    return max_trans_len, max_sent_len


def get_dataset(type, dataset_dir, df, scoring_criteria):
    if type == 'train':
        if args.use_feedback:
            path = dataset_dir + 'train_fbck'
        else:
            path = dataset_dir + 'train'

    elif type == 'dev':
        path = dataset_dir + 'dev'
    else:
        path = dataset_dir + 'test'

    if Path(path).is_file():
        with open(path, 'rb') as f:
            ds = pickle.load(f)
    else:
        if args.use_feedback:
            ds = CallDatasetWithFbk(df, scoring_criteria)
        else:
            ds = CallDataset(df, scoring_criteria)

        with open(path, 'wb') as f:
            pickle.dump(ds, f)
    return ds


def run_cross_validation(train_df, test_df):
    # kf = KFold(n_splits=2, shuffle=True)
    kf = ShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    kfold_results = []
    fold = 0
    for train, dev in kf.split(train_df):
        # train, dev = train_test_split(df_sampled, test_size=0.20)
        t_df = train_df.iloc[train].copy()
        # t_df = t_df.loc[t_df.text.apply(lambda x: len(x.split('\n'))).sort_values().index]
        dev_df = train_df.iloc[dev].copy()
        subscore_dist = t_df.loc[:, scoring_criteria].apply(lambda x: x.value_counts())
        print("Subscore distribution count in Training set\n", subscore_dist)
        subscore_dist = dev_df.loc[:, scoring_criteria].apply(lambda x: x.value_counts())
        print("Subscore distribution count in Dev set\n", subscore_dist)

        dataset_transcripts_train = CallDataset(t_df, scoring_criteria)
        dataset_transcripts_dev = CallDataset(dev_df, scoring_criteria)
        dataset_transcripts_test = CallDataset(test_df, scoring_criteria)
        max_trans_len, max_sent_len = 512, 128
        vocab = dataset_transcripts_train.get_vocab()
        dataset_transcripts_train.save_vocab('vocab')

        batch_size = args.batch_size
        c = Collate(vocab, args.device)
        dataloader_transcripts_train = DataLoader(dataset_transcripts_train, batch_size=batch_size, shuffle=True,
                                                  num_workers=args.num_workers, collate_fn=c.collate)
        dataloader_transcripts_dev = DataLoader(dataset_transcripts_dev, batch_size=batch_size, shuffle=False,
                                                num_workers=args.num_workers, collate_fn=c.collate)
        dataloader_transcripts_test = DataLoader(dataset_transcripts_test, batch_size=batch_size, shuffle=False,
                                                 num_workers=args.num_workers, collate_fn=c.collate)

        embedding_model = KeyedVectors.load(word_embedding_pt[args.word_embedding], mmap='r')
        vocab_size = len(vocab)
        embedding_size = embedding_model.vector_size
        weights_matrix = np.zeros((vocab_size, embedding_size))
        i = 2 # ignore <UNK> and <pad> tokens
        for word in vocab.get_itos()[2:]:
            try:
                weights_matrix[i] = embedding_model[word]  # model.wv[word] for trained word2vec
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_size,))
            i += 1
        weights_matrix[0] = np.mean(weights_matrix, axis=0)
        weights_matrix = torch.tensor(weights_matrix)
        trainer = TrainModel(dataloader_transcripts_train, dataloader_transcripts_dev, vocab_size, embedding_size,
                             weights_matrix, args, max_trans_len, max_sent_len, scoring_criteria, fold)
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
        fold+=1
        kfold_results.append(metrics)
    return kfold_results


def run_cross_validation_mlm(tokenizer):
    print("training mlm")

    paths = [str(x) for x in Path(args.trans_path).glob("**/*.txt")]
    print(args.train_samples)
    if args.train_samples>0:
        mini_paths = random.sample(paths, args.train_samples)
    else:
        mini_paths = paths
    mlm_ds = MLMDataSet(mini_paths, tokenizer)
    train_size = int(0.8 * len(mlm_ds))
    test_size = len(mlm_ds) - train_size
    train_dataset, dev_dataset = random_split(mlm_ds, [train_size, test_size])
    all_test_paths = [str(x) for x in Path(args.test_path).glob("**/*.txt")]
    test_paths = random.sample(all_test_paths, args.train_samples//10)
    test_dataset = MLMDataSet(test_paths, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer)
    dataloader_transcripts_train = DataLoader(train_dataset, batch_size=4, shuffle=True,
                                              collate_fn=collator.torch_call)
    dataloader_transcripts_dev = DataLoader(dev_dataset, batch_size=4, shuffle=True,
                                              collate_fn=collator.torch_call)
    dataloader_transcripts_test = DataLoader(test_dataset, batch_size=1, shuffle=True,
                                             collate_fn=collator.torch_call)
    max_trans_len, max_sent_len = 512, 128
    fold = 0
    trainer = TrainModel(dataloader_transcripts_train, dataloader_transcripts_dev, None, None,
                         None, args, max_trans_len, max_sent_len, None, fold)
    trainer.train_mlm_model(tokenizer)
    model = torch.load(args.save_path + 'fold_' + str(fold) + '_best_mlm_model')
    test_error, pred_df = get_mlm_metrics(dataloader_transcripts_test, model, tokenizer)
    pred_df.to_pickle(args.save_path+'fold_' + str(trainer.fold)+'_mlm_test.p')
    print("Test set loss = {}".format(test_error))


def train_tokenizer(file_path, save_path):
    print("training tokenizer")
    paths = [str(x) for x in Path(file_path).glob("**/*.txt")]
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        "[CLS]",
        "[PAD]",
        "[UNK]",
        "[MASK]",
        "[SEP]",
    ], show_progress=True)
    os.mkdir(save_path)
    tokenizer.save_model(save_path)
    with open(os.path.join(save_path, "config.json"), "w") as f:
        tokenizer_cfg = {
            "do_lower_case": True,
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "model_max_length": 512,
            "max_len": 512,
        }
        json.dump(tokenizer_cfg, f)


if __name__ == "__main__":
    args = _parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Arguments:", args)
    if args.model == 'mlm':
        if not os.path.isdir(args.tok_path):
            train_tokenizer(args.trans_path, args.tok_path)
        tokenizer = BertTokenizerFast.from_pretrained(args.tok_path, add_special_tokens=True)
        run_cross_validation_mlm(tokenizer)
    else:
        train_ds_path = dataset_dir+'train'
        test_ds_path = dataset_dir+'test'
        if args.new_data:
            score_df, q_text = prepare_score_df(
                path_to_handscored_p, workgroup=args.workgroup)
            train_df = prepare_trancript_score_df(score_df, q_text, args.trans_path)
            with open(train_ds_path, 'wb') as f:
                pickle.dump(train_df, f)
            test_df = prepare_trancript_score_df(score_df, q_text, args.test_path)
            with open(test_ds_path, 'wb') as f:
                pickle.dump(test_df, f)
        else:
            with open(train_ds_path, 'rb') as f:
                train_df = pickle.load(f)
            with open(test_ds_path, 'rb') as f:
                test_df = pickle.load(f)

        if args.model == 'AllSubScores':
            scoring_criteria = sub_score_categories
        elif args.model == 'BestSubScores':
            scoring_criteria = sub_score_categories[:4]
        else:
            scoring_criteria = [args.model]
        if args.use_feedback:
            comment_obj = FeedbackComments(train_df, args.k)
            top_k_comments = comment_obj.extract_top_k_comments(scoring_criteria)
            train_df = comment_obj.df
        if args.train_samples > 0:
            train_df = balance_df(train_df, args.train_samples)

        subscore_dist = test_df.loc[:, scoring_criteria].apply(lambda x: x.value_counts())
        print("Subscore distribution count in Test set\n", subscore_dist)
        kfold_results = run_cross_validation(train_df, test_df)

    # avg_tuple = [sum(y) / len(y) for y in zip(*kfold_results)]
    # print("Overall accuracy={} Overall F1 score={}".format(avg_tuple[0], avg_tuple[1]))
