# -*- coding: utf-8 -*-
"""Inference_fns

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gtEyI0DrfZVDxgsIGqETSfGBeIPuvqU4
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import classification_report, f1_score, mean_squared_error
from PrepareDf import prepare_baseline_df
from matplotlib import pyplot as plt
from sklearn import metrics
import torch.nn as nn

def get_score_target(batch, loss, scoring_criteria):
    if loss == 'mse':
        # put device in gpu may have to be modified
        target = torch.tensor([sample[scoring_criteria] for sample in batch['scores']],
                              dtype=torch.float).view(-1, 1)
    elif loss == 'bce':
        target = torch.tensor([sample[scoring_criteria] for sample in batch['scores']],
                              dtype=torch.float)
    elif loss == 'cel':
        target = torch.tensor([sample[scoring_criteria] for sample in batch['scores']],
                              dtype=torch.long)
    return target


def get_mlm_metrics(dataloader, model, tokenizer):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0
    target_arr = []
    pred_arr = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            output = model(item)
            labels = item['labels'][item['labels'] != -100]
            if len(labels) == 0:
                continue
            loss += criterion(output, labels).detach().item()
            pred_ids = torch.argmax(output, dim=-1)
            loss += criterion(output, labels)
            target_arr.append(tokenizer.convert_ids_to_tokens(labels))
            pred_arr.append(tokenizer.convert_ids_to_tokens(pred_ids))
    df = pd.DataFrame()
    df['Target Tokens'] = target_arr
    df['Pred Tokens'] = pred_arr
    return loss/len(dataloader), df


def val_get_metrics(dataloader, model, scoring_criterion, loss, loss_fn):
    id_arr = []
    text_arr = []
    attn_score_arr = []
    pred_arr = []
    target_arr = []
    raw_pred_arr = []
    model.eval()
    thresh = 0.5
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs, scores, _ = model(batch['indices'], batch['lens'], batch['trans_pos_indices'],
                                    batch['word_pos_indices'])
            output = outputs[0]
            targets = get_score_target(batch, loss, scoring_criterion)
            l = 0
            if loss == 'cel':
                for i in range(len(scoring_criterion)):
                    l += loss_fn[i](outputs[0][:, 2 * i:2 * (i + 1)], targets[:, i])
                l /= 2 * len(scoring_criterion)
            else:
                l += loss_fn(outputs[0], targets)
            val_loss += l.detach().item()

            target = [sample[scoring_criterion].tolist() for sample in batch['scores']]
            if loss == 'mse':
                raw_proba = [None for i in range(len(scoring_criterion))]
                pred = output.numpy()
            elif loss == 'bce':
                raw_proba = torch.sigmoid(output)
                pred = ((raw_proba > thresh).long()).tolist()
                raw_proba = raw_proba.tolist()
            elif loss == 'cel':
                output = output.reshape(-1, len(scoring_criterion), 2)
                probs = torch.softmax(output, dim=-1)
                max_vals = torch.max(probs, dim=-1)
                raw_proba = probs[:, :, 1].tolist()
                pred = max_vals[1].tolist()
            else:
                raise ValueError("Cannot do inference")
            raw_pred_arr.extend(raw_proba)
            pred_arr.extend(pred)
            target_arr.extend(target)
            id_arr.extend(batch['id'])
            text_arr.extend(batch['text'])
            attn_score_arr.extend(scores.numpy())
        raw_pred_df = pd.DataFrame(raw_pred_arr, columns=['RawProba ' + category for category in scoring_criterion])
        pred_df = pd.DataFrame(pred_arr, columns=['Pred ' + category for category in scoring_criterion])
        target_df = pd.DataFrame(target_arr, columns=['True ' + category for category in scoring_criterion])
        df = pd.concat((pred_df, target_df, raw_pred_df), axis=1)
        df['id'] = id_arr
        df['text'] = text_arr
        df['scores'] = attn_score_arr
    model.train()
    print("Sample predictions")
    print("target:", target_arr[:10])
    print("prediction:", pred_arr[:10])

    if loss == 'mse':
        mse_error = mean_squared_error(target_arr, pred_arr)
        print("MSE Error = {}".format(mse_error))
        return mse_error, val_loss/len(dataloader)
    else:
        # f1 = f1_score(target_arr, pred_arr)
        auc_arr = []
        for i in range(len(scoring_criterion)):
            crit = scoring_criterion[i]
            print("Category:", crit)
            print(classification_report(target_df.iloc[:, i], pred_df.iloc[:, i]))
            y_pred_proba = df['RawProba ' + crit]
            y_true = df['True ' + crit]
            if not y_pred_proba.isna().sum():
                fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_proba)
                auc = metrics.roc_auc_score(y_true, y_pred_proba).round(2)
                auc_arr.append(auc)
                # print("AUC for {} = {}".format(crit, auc))
            else:
                print("raw scores may be NAN for {}".format(crit))
        clr = classification_report(target_arr, pred_arr)
        return clr, val_loss/len(dataloader), auc_arr


def get_metrics(dataloader, model, scoring_criterion, loss):
    id_arr = []
    text_arr = []
    attn_score_arr = []
    pred_arr = []
    target_arr = []
    raw_pred_arr = []
    model.eval()
    thresh = 0.5
    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs, scores, _ = model(batch['indices'], batch['lens'], batch['trans_pos_indices'],
                                    batch['word_pos_indices'])
            output = outputs[0]
            target = [sample[scoring_criterion].tolist() for sample in batch['scores']]
            if loss == 'mse':
                raw_proba = [None for i in range(len(scoring_criterion))]
                pred = output.numpy()
            elif loss == 'bce':
                raw_proba = torch.sigmoid(output)
                pred = ((raw_proba > thresh).long()).tolist()
                raw_proba = raw_proba.tolist()
            elif loss == 'cel':
                output = output.reshape(-1, len(scoring_criterion), 2)
                probs = torch.softmax(output, dim=-1)
                max_vals = torch.max(probs, dim=-1)
                raw_proba = probs[:, :, 1].tolist()
                pred = max_vals[1].tolist()
            else:
                raise ValueError("Cannot do inference")
            raw_pred_arr.extend(raw_proba)
            pred_arr.extend(pred)
            target_arr.extend(target)
            id_arr.extend(batch['id'])
            text_arr.extend(batch['text'])
            attn_score_arr.extend(scores.numpy())
        raw_pred_df = pd.DataFrame(raw_pred_arr, columns=['RawProba ' + category for category in scoring_criterion])
        pred_df = pd.DataFrame(pred_arr, columns=['Pred ' + category for category in scoring_criterion])
        target_df = pd.DataFrame(target_arr, columns=['True ' + category for category in scoring_criterion])
        df = pd.concat((pred_df, target_df, raw_pred_df), axis=1)
        df['id'] = id_arr
        df['text'] = text_arr
        df['scores'] = attn_score_arr
    model.train()
    print("Sample predictions")
    print("target:", target_arr[:10])
    print("prediction:", pred_arr[:10])

    if loss == 'mse':
        mse_error = mean_squared_error(target_arr, pred_arr)
        print("MSE Error = {}".format(mse_error))
        return mse_error, df
    else:
        # f1 = f1_score(target_arr, pred_arr)
        for i in range(len(scoring_criterion)):
            crit = scoring_criterion[i]
            print("Category:", crit)
            print(classification_report(target_df.iloc[:, i], pred_df.iloc[:, i]))
            y_pred_proba = df['RawProba ' + crit]
            y_true = df['True ' + crit]
            if not y_pred_proba.isna().sum():
                fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_proba)
                auc = metrics.roc_auc_score(y_true, y_pred_proba).round(2)
                print("AUC for {} = {}".format(crit, auc))
            else:
                print("raw scores may be NAN for {}".format(crit))
        clr = classification_report(target_arr, pred_arr)
        return clr, df


def plot_roc(scoring_criteria, df, path):
    plt.clf()
    for crit in scoring_criteria:
        y_pred_proba = df['RawProba ' + crit]
        y_true = df['True ' + crit]
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_proba)
        auc = metrics.roc_auc_score(y_true, y_pred_proba).round(2)
        print("AUC for {} = {}".format(crit, auc))
        plt.plot(fpr, tpr, label="{}, auc={}".format(crit, auc))
        plt.xlabel("False Positivity Rate, bad calls class 1 ")
        plt.ylabel("True Positivity Rate, bad calls class 1")
        plt.legend(loc=4)
    plt.savefig(path)
    plt.show()


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
                    df.loc[call_id, 'Predicted Score'] = (1 - sales_df.loc[call_id, 'score'])
                    df.loc[call_id, 'Overall Score'] = test_df.loc[call_id, 'CombinedPercentileScore']
            elif test_df.loc[call_id, 'WorkgroupQueue'] == 'Customer Service':
                if call_id in cs_df.index:
                    df.loc[call_id, 'Predicted Score'] = (1 - cs_df.loc[call_id, 'score'])
                    df.loc[call_id, 'Overall Score'] = test_df.loc[call_id, 'CombinedPercentileScore']
        mse_error = mean_squared_error(df['Overall Score'].tolist(), df['Predicted Score'].tolist())
        print("MSE Error = {}".format(mse_error))
        return mse_error


def get_regression_metrics(dataloader, model, scoring_criterion):
    pred_arr = []
    target_arr = []
    model.eval()
    total_correct = 0
    batch_size = dataloader.batch_size
    with torch.no_grad():
        for batch in tqdm(dataloader):
            output, scores = model(batch['indices'])
            for i in range(output.shape[0]):
                pred = output[i].item()
                target = batch['scores'][i][scoring_criterion]
                pred_arr.append(pred)
                target_arr.append(target)
    model.train()
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
                df.loc[call_id, 'Overall Score'] = test_df.loc[call_id, 'CombinedPercentileScore'] / 100
        elif test_df.loc[call_id, 'WorkgroupQueue'] == 'Customer Service':
            if call_id in cs_df.index:
                df.loc[call_id, 'Predicted Score'] = 1 - cs_df.loc[call_id, 'score']
                df.loc[call_id, 'Overall Score'] = test_df.loc[call_id, 'CombinedPercentileScore'] / 100
    mse_error = mean_squared_error(df['Overall Score'].tolist(), df['Predicted Score'].tolist())
    print("MSE Error = {}".format(mse_error))
    return mse_error


def get_accuracy(dataloader, model, classifier):
    total_correct = 0

    for batch in tqdm(dataloader):

        batch_size = len(batch['indices'])

        output, hidden = model(batch['indices'])
        output = output[:, -1, :]

        output = classifier(output)

        for i in range(batch_size):

            classification = torch.argmax(output[i]).item()
            target = batch['category'][i]
            if target == classification:
                total_correct += 1

    acc = total_correct / (len(dataloader) * batch_size)
    print("Accuracy: {}".format(acc))
    return acc
