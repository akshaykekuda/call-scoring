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
from Inference_fns import val_get_metrics, get_mlm_metrics
from sklearn.utils import class_weight
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim


class TrainModel:
    def __init__(self, dataloader_train, dataloader_dev, vocab_size, vec_size, weights_matrix, args, max_trans_len,
                 max_sent_len, scoring_criteria, fold):
        self.dataloader_train = dataloader_train
        self.dataloader_dev = dataloader_dev
        self.vocab_size = vocab_size
        self.vec_size = vec_size
        self.weights_matrix = weights_matrix
        self.args = args
        self.max_trans_len = max_trans_len
        self.max_sent_len = max_sent_len
        self.scoring_criteria = scoring_criteria
        self.fold = fold

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
        elif self.args.attention == 'hsan1':
            encoder = HSAN1(self.vocab_size, self.vec_size, self.args.model_size, self.weights_matrix,
                           self.max_sent_len, self.args.word_nh, self.args.dropout, self.args.num_layers, 
                           self.args.word_nlayers)
        elif self.args.attention == 'hs2an':
            encoder = HS2AN(self.vocab_size, self.vec_size, self.args.model_size, self.weights_matrix,
                            self.max_trans_len, self.max_sent_len, self.args.word_nh, self.args.sent_nh, self.args.dropout, self.args.num_layers,
                            self.args.word_nlayers)
        else:
            raise ValueError("Invalid Attention Model argument")

        if self.args.loss == 'mse':
            fcn = FCN_ReLu(self.args.model_size, len(self.scoring_criteria), self.args.dropout)
        elif self.args.loss == 'cel':
            fcn = FCN_Tanh(self.args.model_size, len(self.scoring_criteria)*2, self.args.dropout)
        elif self.args.loss == 'bce':
            fcn = FCN_Tanh(self.args.model_size, len(self.scoring_criteria), self.args.dropout)
        else:
            raise ValueError("Invalid Optimizer argument")

        mtl_head = FCN_MTL(self.args.model_size, self.args.k, self.args.dropout)

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

    def train_cel_model(self):
        print("train using cross entropy loss without feedback")
        class_weights = self.get_class_weights()
        model = self.get_model()
        loss_fn_arr = []
        for i in range(len(self.scoring_criteria)):
            loss_fn_arr.append(nn.CrossEntropyLoss(weight=class_weights[i], reduction='mean'))
        model_optimizer = optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.reg)
        scheduler = MultiStepLR(model_optimizer, milestones=[15, 35], gamma=0.1)
        model = self.train_model(self.args.epochs, model, loss_fn_arr, model_optimizer, scheduler)
        return model

    def train_bce_model(self):
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

    def train_multi_label_model(self):
        print("train using BCE loss without feedback")
        class_weights = self.get_class_weights()
        positive_weights = class_weights[:, 1]
        model = self.get_model()
        if self.args.loss == 'bce':
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
        elif self.args.loss == 'cel':
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError("Cannot use {} in multilabel setting".format(self.args.loss))

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

    def train_linear_regressor(self):
        print("train linear regressor")
        model = self.get_model()
        loss_fn = nn.MSELoss()
        model_optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = MultiStepLR(model_optimizer, milestones=[10, 20], gamma=0.1)
        epochs = self.args.epochs
        model = self.train_model(epochs, model, loss_fn, model_optimizer, scheduler)
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
                                  dtype=torch.long, device=self.args.device)
        return target

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
        train_loss = []
        val_loss = []
        best_val_error = float(inf)
        for n in range(epochs):
            epoch_loss = 0
            for idx, batch in enumerate(tqdm(self.dataloader_train)):
                loss = 0
                outputs, scores, _ = model(batch['indices'], batch['lens'], batch['trans_pos_indices'],
                                        batch['word_pos_indices'])
                targets = self.get_score_target(batch)
                if self.args.loss == 'cel':
                    for i in range(len(self.scoring_criteria)):
                        loss += loss_fn[i](outputs[0][:, 2*i:2*(i+1)], targets[:, i])
                    # loss /= 2*len(self.scoring_criteria)*self.args.acum_step
                else:
                    loss += loss_fn(outputs[0], targets)
                epoch_loss += loss.detach().item()
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                if (idx%self.args.acum_step == self.args.acum_step-1) or idx == len(self.dataloader_train)-1:
                    model_optimizer.step()
                    model_optimizer.zero_grad()
            # scheduler.step()
            avg_epoch_loss = epoch_loss / len(self.dataloader_train)
            loss_arr.append(avg_epoch_loss)
            print("start of val on train set")
            train_metrics, train_error, train_auc = val_get_metrics(self.dataloader_train, model, self.scoring_criteria, self.args.loss, loss_fn)
            print("start of val on dev set")
            dev_metrics, val_error, val_auc = val_get_metrics(self.dataloader_dev, model, self.scoring_criteria, self.args.loss, loss_fn)
            if val_error < best_val_error:
                model.eval()
                torch.save(model, self.args.save_path + 'fold_' + str(self.fold) + '_best_model')
                best_val_error = val_error
            print("Average training loss at epoch {}: {}".format(n, avg_epoch_loss))
            print("Average val loss at epoch {}: {}".format(n, val_error))
            
            print("Training metric at end of epoch {}:".format(n))
            print(train_metrics)
            for i, crit in enumerate(self.scoring_criteria):
                print("Train AUC for {} = {}".format(crit, train_auc[i]))
            
            print("Dev metric at end of epoch {}:".format(n))
            print(dev_metrics)
            for i, crit in enumerate(self.scoring_criteria):
                print("Dev AUC for {} = {}".format(crit, val_auc[i]))            
            
            train_acc.append(train_metrics)
            dev_acc.append(dev_metrics)
            train_loss.append(train_error)
            val_loss.append(val_error)
            model.train()
        fig, axs = plt.subplots(1, 2, figsize=(16,6))
        fig.suptitle('Losses')
        axs[0].plot(loss_arr)
        axs[1].plot(val_loss, label = 'dev set')
        axs[1].plot(train_loss, label='train set')
        axs[0].title.set_text('training loss')
        axs[1].title.set_text('val loss')
        axs[1].legend()
        plt.show()
        plt.savefig(self.args.save_path + 'fold_'+ str(self.fold) + '_loss.png')
        print("Epoch Losses:", loss_arr)
        print("Training Evaluation Metrics: ", train_acc)
        print("Dev Evaluation Metrics: ", dev_acc)
        model = torch.load(self.args.save_path + 'fold_' + str(self.fold) + '_best_model')
        return model

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
                train_metrics, _ = val_get_metrics(self.dataloader_train, mtl_model, self.scoring_criteria, self.args.loss)
                print("Dev metric at end of epoch {}:".format(n))
                dev_metrics, _ = val_get_metrics(self.dataloader_dev, mtl_model, self.scoring_criteria, self.args.loss)
                train_acc.append(train_metrics)
                dev_acc.append(dev_metrics)
        print("Epoch Losses:", loss_arr)
        plt.plot(loss_arr)
        plt.savefig(self.args.save_path + 'loss.png')
        print("Training Evaluation Metrics: ", train_acc)
        print("Dev Evaluation Metrics: ", dev_acc)
        return mtl_model

    def train_mlm_model(self, tokenizer):
        model = MLMNetwork(self.args.model_size, tokenizer, self.args.dropout, self.args.word_nh)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.args.lr)
        loss_arr = []
        train_loss = []
        val_loss = []
        best_val_error = float('inf')
        for n in range(self.args.epochs):
            epoch_loss = 0
            model.train()
            for batch in tqdm(self.dataloader_train):
                output = model(batch)
                labels = batch['labels'][batch['labels']!=-100]
                if len(labels) == 0:
                    continue
                loss = criterion(output, labels)
                optimizer.zero_grad()
                epoch_loss += loss.detach().item()
                loss.backward()
                optimizer.step()
                loss_arr.append(loss.detach().item())
            avg_epoch_loss = epoch_loss / len(self.dataloader_train)
            print("Average loss at epoch {}: {}".format(n, avg_epoch_loss))
            print("start of val on train set")
            train_error, _ = get_mlm_metrics(self.dataloader_train, model, tokenizer)
            print("start of val on dev set")
            val_error, _ = get_mlm_metrics(self.dataloader_dev, model, tokenizer)
            if val_error < best_val_error:
                model.eval()
                torch.save(model, self.args.save_path + 'fold_' + str(self.fold) + '_best_mlm_model')
                best_val_error = val_error
            print("Average training loss at epoch {}: {}".format(n, avg_epoch_loss))
            print("Average val loss at epoch {}: {}".format(n, val_error))
            train_loss.append(train_error)
            val_loss.append(val_error)
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Losses')
        axs[0].plot(loss_arr)
        axs[1].plot(val_loss, label='dev set')
        axs[1].plot(train_loss, label='train set')
        axs[0].title.set_text('training loss per token')
        axs[1].title.set_text('val loss per epoch')
        axs[1].legend()
        plt.show()
        plt.savefig(self.args.save_path + 'fold_' + str(self.fold) + '_loss.png')




