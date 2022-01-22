# -*- coding: utf-8 -*-
"""TrainModel

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JkW85J7ZhjzUcKk7sSf8mSfFK6svNtND
"""

from __future__ import unicode_literals, print_function, division
from matplotlib import pyplot as plt
from tqdm import tqdm
from Models import *
from Inference_fns import get_metrics
from sklearn.utils import class_weight
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim


class TrainModel:
    def __init__(self, dataloader_train, dataloader_dev, vocab_size, vec_size, weights_matrix, args, max_trans_len,
                 max_sent_len, scoring_criteria):
        self.dataloader_train = dataloader_train
        self.dataloader_dev = dataloader_dev
        self.vocab_size = vocab_size
        self.vec_size = vec_size
        self.weights_matrix = weights_matrix
        self.args = args
        self.max_trans_len = max_trans_len
        self.max_sent_len = max_sent_len
        self.scoring_criteria = scoring_criteria

    def train(self):
        raise "To be Implemented"

    def get_model(self):
        if self.args.attention == 'baseline':
            encoder = EncoderRNN(self.vocab_size, self.vec_size, self.args.model_size, self.weights_matrix,
                                 self.args.dropout)
        elif self.args.attention == 'gru_attention':
            encoder = GRUAttention(self.vocab_size, self.vec_size, self.args.model_size, self.weights_matrix,
                                   self.args.dropout)
        elif self.args.attention == 'han':
            encoder = HAN(self.vocab_size, self.vec_size, self.args.model_size, self.weights_matrix, self.args.dropout)
        elif self.args.attention == 'hsan':
            encoder = HSAN(self.vocab_size, self.vec_size, self.args.model_size, self.weights_matrix,
                           self.max_trans_len, self.max_sent_len, self.args.num_heads, self.args.dropout)
        elif self.args.attention == 'hs2an':
            encoder = HS2AN(self.vocab_size, self.vec_size, self.args.model_size, self.weights_matrix,
                            self.max_trans_len, self.max_sent_len, self.args.num_heads, self.args.dropout, self.args.num_layers)
        else:
            raise ValueError("Invalid Attention Model argument")

        if self.args.loss == 'mse':
            fcn = FCN_ReLu(6 * self.args.model_size, len(self.scoring_criteria), self.args.dropout)
        elif self.args.loss == 'cel':
            fcn = FCN_Tanh(6 * self.args.model_size, len(self.scoring_criteria)*2, self.args.dropout)
        elif self.args.loss == 'bce':
            fcn = FCN_Tanh(6 * self.args.model_size, len(self.scoring_criteria), self.args.dropout)
        else:
            raise ValueError("Invalid Optimizer argument")

        mtl_head = FCN_MTL(6 * self.args.model_size, self.args.k, self.args.dropout)

        if self.args.use_feedback:
            model = EncoderMTL(encoder, fcn, mtl_head, len(self.scoring_criteria))
        else:
            model = EncoderMTL(encoder, fcn, mtl_head, 0)

        return model

    def get_class_weights(self):
        x = [batch['scores'] for batch in self.dataloader_train]
        arr = []
        pos_wt = []
        for b in x:
            arr.extend(b)
        for sub_category in self.scoring_criteria:
            y_train = [sample[sub_category].item() for sample in arr]
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train),
                                                              y=y_train)
            pos_wt.append(class_weights)
        positive_weights = torch.tensor(pos_wt, dtype=torch.float)
        return positive_weights

    def train_biclass_model(self):
        print("train using cross entropy loss without feedback")
        class_weights = self.get_class_weights()
        model = self.get_model()
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.squeeze())
        model_optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = MultiStepLR(model_optimizer, milestones=[10, 20], gamma=0.1)
        model = self.train_model(self.args.epochs, model, loss_fn, model_optimizer, scheduler)
        return model

    def train_multi_label_model(self):
        print("train using BCE loss without feedback")
        class_weights = self.get_class_weights()
        positive_weights = class_weights[:, 1]
        model = self.get_model()
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
        model_optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = MultiStepLR(model_optimizer, milestones=[10, 20], gamma=0.1)
        epochs = self.args.epochs
        model = self.train_model(epochs, model, loss_fn, model_optimizer, scheduler)
        return model

    def train_mtl_model(self):
        print("MTL")
        class_weights = self.get_class_weights()
        model = self.get_model()
        if self.args.loss == 'cel':
            if len(self.scoring_criteria) == 1:
                loss_fn1 = nn.CrossEntropyLoss(weight=class_weights.squeeze())
            else:
                raise ValueError("Cannot use CEL in multilabel setting")
        elif self.args.loss == 'bce':
            positive_weights = class_weights[:, 1]
            loss_fn1 = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
        else:
            raise ValueError("invalid loss function for score model")
        loss_fn = [loss_fn1]
        for i in range(len(self.scoring_criteria)):
            loss_fn.append(copy.deepcopy(nn.BCEWithLogitsLoss()))
        model_optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = MultiStepLR(model_optimizer, milestones=[10, 20], gamma=0.1)
        model = self.train_mtl(self.args.epochs, model, loss_fn, model_optimizer, scheduler)
        return model

    def train_linear_regressor(self, scoring_criteria):
        print("train linear regressor")
        model = self.get_model()
        loss_fn = nn.MSELoss()
        model_optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = MultiStepLR(model_optimizer, milestones=[10, 20], gamma=0.1)
        epochs = self.args.epochs
        model = self.train_model(epochs, model, loss_fn, model_optimizer, scheduler)
        return model

    def train_model(self, epochs, model, loss_fn, model_optimizer, scheduler):
        print("inside train model without feedback")
        train_acc = []
        dev_acc = []
        model = model.to(self.args.device)
        model.train()
        loss_arr = []
        print(model)
        print(loss_fn)
        print(model_optimizer)
        for n in range(epochs):
            epoch_loss = 0
            for batch in tqdm(self.dataloader_train):
                loss = 0
                outputs, scores, _ = model(batch['indices'], batch['lens'], batch['trans_pos_indices'],
                                        batch['word_pos_indices'])
                targets = self.get_score_target(batch)
                loss += loss_fn(outputs[0], targets)
                model_optimizer.zero_grad()
                epoch_loss += loss.detach().item()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                model_optimizer.step()
            scheduler.step()
            avg_epoch_loss = epoch_loss / len(self.dataloader_train)
            print("Average loss at epoch {}: {}".format(n, avg_epoch_loss))
            loss_arr.append(avg_epoch_loss)
            if n % 5 == 4:
                print("Training metric at end of epoch {}:".format(n))
                train_metrics, _ = get_metrics(self.dataloader_train, model, self.scoring_criteria, self.args.loss)
                print("Dev metric at end of epoch {}:".format(n))
                dev_metrics, _ = get_metrics(self.dataloader_dev, model, self.scoring_criteria, self.args.loss)
                train_acc.append(train_metrics)
                dev_acc.append(dev_metrics)
        print("Epoch Losses:", loss_arr)
        plt.plot(loss_arr)
        plt.show()
        plt.savefig(self.args.save_path + 'loss.png')
        print("Training Evaluation Metrics: ", train_acc)
        print("Dev Evaluation Metrics: ", dev_acc)
        return model

    def get_score_target(self, batch):
        if self.args.loss == 'mse':
            # put device in gpu may have to be modified
            target = torch.tensor([sample[self.scoring_criteria] for sample in batch['scores']],
                                  dtype=torch.float, device=self.args.device).view(-1, 1)
        elif self.args.loss == 'bce':
            target = torch.tensor([sample[self.scoring_criteria] for sample in batch['scores']],
                                  dtype=torch.float, device=self.args.device)
        elif self.args.loss == 'cel':
            target = torch.tensor([sample[self.scoring_criteria] for sample in batch['scores']],
                                  dtype=torch.long, device=self.args.device).squeeze(dim=-1)
        else:
            raise "invalid loss fn"
        return target

    def train_mtl(self, epochs, mtl_model, loss_fn, model_optimizer, scheduler):
        print("inside train mtl model")
        train_acc = []
        dev_acc = []
        mtl_model = mtl_model.to(self.args.device)
        mtl_model.train()
        loss_arr = []
        print(mtl_model)
        print(loss_fn)
        print(model_optimizer)
        for n in range(epochs):
            epoch_loss = 0
            for batch in tqdm(self.dataloader_train):
                loss = 0
                outputs, scores, _ = mtl_model(batch['indices'], batch['lens'], batch['trans_pos_indices'],
                                            batch['word_pos_indices'])
                targets = [self.get_score_target(batch)]
                target2 = [torch.tensor(batch[criterion+" fbk_vector"], dtype=float, device=self.args.device)
                           for criterion in self.scoring_criteria]
                targets.extend(target2)

                for i, fn in enumerate(loss_fn):
                    loss += loss_fn[i](outputs[i], targets[i])
                model_optimizer.zero_grad()
                epoch_loss += loss.detach().item()
                loss.backward()
                model_optimizer.step()
            scheduler.step()
            avg_epoch_loss = epoch_loss / len(self.dataloader_train)
            print("Average loss at epoch {}: {}".format(n, avg_epoch_loss))
            loss_arr.append(avg_epoch_loss)
            if n % 5 == 4:
                print("Training metric at end of epoch {}:".format(n))
                train_metrics, _ = get_metrics(self.dataloader_train, mtl_model, self.scoring_criteria, self.args.loss)
                print("Dev metric at end of epoch {}:".format(n))
                dev_metrics, _ = get_metrics(self.dataloader_dev, mtl_model, self.scoring_criteria, self.args.loss)
                train_acc.append(train_metrics)
                dev_acc.append(dev_metrics)
        print("Epoch Losses:", loss_arr)
        plt.plot(loss_arr)
        plt.savefig(self.args.save_path + 'loss.png')
        print("Training Evaluation Metrics: ", train_acc)
        print("Dev Evaluation Metrics: ", dev_acc)
        return mtl_model
