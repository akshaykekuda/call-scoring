# -*- coding: utf-8 -*-
"""Models

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J7fMYaCzIFXjdod6uObHyVH34szYMIXb
"""

from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Custom_Multihead_Attention import custom_multi_head_attention_forward

F.multi_head_attention_forward = custom_multi_head_attention_forward

class FCN_Tanh(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate):
        super(FCN_Tanh, self).__init__()
        self.fcn = nn.Sequential(
            nn.Linear(input_size, num_classes)
            # nn.Tanh()
            # nn.Dropout(dropout_rate),
            # nn.Linear(64, num_classes),
        )

    def forward(self, x):
        output = self.fcn(x)
        return output


class FCN_ReLu(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate):
        super(FCN_ReLu, self).__init__()
        self.fcn = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        output = self.fcn(x)
        return output


class FCN_MTL(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate):
        super(FCN_MTL, self).__init__()
        self.fcn = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        output = self.fcn(x)
        return output


class EncoderRNN(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, weights_matrix, dropout_rate):
        super(EncoderRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=1)
        self.embedding.load_state_dict({'weight': weights_matrix})

        self.hidden_size = hidden_size
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs, *_):
        embed_output = self.embedding(inputs)
        embed_output = torch.mean(embed_output, dim=2, keepdim=True).squeeze(2)
        output, hidden = self.gru(embed_output)
        output = output[:, -1, :]
        return output, None


class LSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, weights_matrix, dropout_rate):
        super(LSTMAttention, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=1)
        self.embedding.load_state_dict({'weight': weights_matrix})
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)

        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, inputs, *_):
        embed_output = self.embedding(inputs)
        embed_output = torch.mean(embed_output, dim=2, keepdim=True).squeeze(2)
        output, hidden = self.lstm(embed_output)
        attn_weights = self.attn(output)
        attn_scores = F.softmax(attn_weights, 1)
        out = torch.bmm(output.transpose(1, 2), attn_scores).squeeze(2)
        return out, attn_scores.squeeze(2)


class GRUAttention(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, weights_matrix, dropout_rate):
        super(GRUAttention, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=1)
        self.embedding.load_state_dict({'weight': weights_matrix})
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, inputs, lens, trans_pos_indices, _):
        embed_output = self.embedding(inputs)
        # print(embed_output)
        attn_mask = trans_pos_indices == 0
        trans_lens = (~attn_mask).sum(dim=1).cpu()
        embed_output = torch.mean(embed_output, dim=2, keepdim=True).squeeze(2)
        pck_seq = torch.nn.utils.rnn.pack_padded_sequence(embed_output, trans_lens, batch_first=True,
                                                          enforce_sorted=False)
        output_pckd, hidden = self.gru(pck_seq)
        output, trans_lens = torch.nn.utils.rnn.pad_packed_sequence(output_pckd, batch_first=True, padding_value=0)
        # output, hidden = self.gru(embed_output)
        attn_weights = self.attn(output)
        ## mask weights
        attn_weights_masked = attn_weights.masked_fill(attn_mask.unsqueeze(2), value=-np.inf)
        # attn_weights = attn_mask.unsqueeze(2) * attn_weights
        attn_scores = F.softmax(attn_weights_masked, 1)
        out = torch.bmm(output.transpose(1, 2), attn_scores).squeeze(2)
        return out, attn_scores.squeeze(2)


class HAN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, weights_matrix, dropout_rate):
        super(HAN, self).__init__()
        self.word_attention = WordAttention(vocab_size, embedding_size, hidden_size, weights_matrix)
        self.sentence_attention = SentenceAttention(2 * hidden_size, hidden_size)

    def forward(self, inputs, lens, trans_pos_indices, word_pos_indices):
        att1 = self.word_attention.forward(inputs, word_pos_indices)
        att2, sentence_att_scores = self.sentence_attention.forward(att1, trans_pos_indices)
        return att2, sentence_att_scores


class SentenceAttention(nn.Module):
    def __init__(self, sentence_embedding_size, hidden_size):
        super(SentenceAttention, self).__init__()
        self.lstm = nn.LSTM(sentence_embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, inputs, positional_indices):
        padding_mask = positional_indices == 0
        trans_lens = (~padding_mask).sum(dim=1).cpu()
        pck_seq = torch.nn.utils.rnn.pack_padded_sequence(inputs, trans_lens, batch_first=True, enforce_sorted=False)
        output_pckd, hidden = self.lstm(pck_seq)
        output, trans_lens = torch.nn.utils.rnn.pad_packed_sequence(output_pckd, batch_first=True, padding_value=0)
        # output, hidden = self.lstm(inputs)
        attn_weights = self.attn(output)
        attn_weights_masked = attn_weights.masked_fill(padding_mask.unsqueeze(2), value=-np.inf)
        attn_scores = F.softmax(attn_weights_masked, 1)
        out = torch.bmm(output.transpose(1, 2), attn_scores).squeeze(2)
        return out, attn_scores.squeeze(2)


class WordAttention(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, weights_matrix, ):
        super(WordAttention, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=1)
        self.embedding.load_state_dict({'weight': weights_matrix})
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, inputs, positional_indices):
        embed_output = self.embedding(inputs)
        embed_output_cat = embed_output.view(-1, *embed_output.size()[2:])
        padding_mask = positional_indices == 0
        sent_lens = (~padding_mask).sum(dim=-1).view(-1).cpu()
        pck_seq = torch.nn.utils.rnn.pack_padded_sequence(embed_output_cat, sent_lens, batch_first=True,
                                                          enforce_sorted=False)
        word_out_pckd, word_hidden = self.lstm(pck_seq)
        word_out, sent_lens = torch.nn.utils.rnn.pad_packed_sequence(word_out_pckd, batch_first=True, padding_value=0)
        # word_out, hidden = self.lstm(embed_output_cat)
        attn_weights = self.attn(word_out)
        ## mask weights
        ##change to use pos indices
        attn_weights = attn_weights.masked_fill(padding_mask.view(-1, *padding_mask.size()[2:]).unsqueeze(dim=2),
                                                value=-np.inf)
        attn_scores = F.softmax(attn_weights, 1)
        # attn_scores = torch.nan_to_num(attn_scores)
        sentence_embedding = torch.sum(word_out * attn_scores, 1)
        return sentence_embedding.reshape(*inputs.size()[0:2], -1)


class HSAN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, weights_matrix, max_trans_len,
                 max_sent_len, num_heads, dropout_rate):
        super(HSAN, self).__init__()
        self.word_attention = WordAttention(vocab_size, embedding_size, hidden_size, weights_matrix)
        self.sentence_self_attention = SentenceSelfAttention(2 * hidden_size, num_heads, max_trans_len, dropout_rate)

    def forward(self, inputs, lens, trans_pos_indices, word_pos_indices):
        att1 = self.word_attention.forward(inputs, word_pos_indices)
        att2, sentence_att_scores = self.sentence_self_attention.forward(att1, trans_pos_indices)
        return att2, sentence_att_scores


class HS2AN(nn.Module):
    def __init__(self, vocab_size, embedding_size, model_size, weights_matrix, max_trans_len,
                 max_sent_len, word_nh, sent_nh, dropout_rate, num_layers, word_num_layers):
        super(HS2AN, self).__init__()
        self.word_self_attention = WordSelfAttention(vocab_size, embedding_size, model_size, weights_matrix,
                                                     max_sent_len, word_nh, dropout_rate, word_num_layers)
        self.sentence_self_attention = SentenceSelfAttention(model_size, sent_nh, max_trans_len, dropout_rate, num_layers)
        self.layerNorm = nn.LayerNorm(model_size)
    def forward(self, inputs, lens, trans_pos_indices, word_pos_indices):
        att1 = self.word_self_attention.forward(inputs, word_pos_indices)
        # att1 = self.layerNorm(att1)
        att2, sentence_att_scores, value = self.sentence_self_attention.forward(att1, trans_pos_indices)
        # print(sentence_att_scores.shape)
        return att2, sentence_att_scores, value


class SentenceSelfAttention(nn.Module):
    def __init__(self, model_size, num_heads, max_trans_len, dropout_rate, num_layers):
        super(SentenceSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(model_size, num_heads, dropout=dropout_rate, batch_first=True)
        # self.multihead_attn = CustomMultiHeadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.position_encoding = nn.Embedding(max_trans_len, model_size, padding_idx=0)
        self.num_layers = num_layers
        self.cls_token = torch.rand(size=(1, model_size), requires_grad=True)

    def forward(self, inputs, positional_indices):
        bs = inputs.size()[0]
        inputs = torch.cat((self.cls_token.repeat(bs, 1).unsqueeze(1), inputs), dim=1)
        positional_encoding = self.position_encoding(positional_indices)
        att_in = inputs + positional_encoding
        padding_mask = positional_indices == 0
        for i in range(self.num_layers):
            query = key = value = att_in
            att_in, attn_output_weights = self.multihead_attn(query, key, value, key_padding_mask=padding_mask)
        attn_output = att_in
        # mask_for_pads = (~padding_mask).unsqueeze(-1).expand(-1, -1, attn_output.size(-1))
        # attn_output *= mask_for_pads
        # attn_output_inter = torch.mean(attn_output[:, 1:-1, :], dim=1, keepdim=False)
        # attn_output = torch.cat((attn_output[:, 0, :], attn_output_inter, attn_output[:, -1, :]), dim=-1)
        attn_output = attn_output[:, 0, :]
        # attn_output = torch.mean(attn_output, dim=1, keepdim=False)
        
        return attn_output, attn_output_weights.squeeze(2), value


class WordSelfAttention(nn.Module):
    def __init__(self, vocab_size, embedding_size, out_dim, weights_matrix, max_sent_len, num_heads, dropout_rate, num_layers):
        super(WordSelfAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=1)
        self.embedding.load_state_dict({'weight': weights_matrix})
        self.multihead_attn = nn.MultiheadAttention(embedding_size, dropout=dropout_rate, num_heads=num_heads, batch_first=True)
        # self.multihead_attn = CustomMultiHeadAttention(embedding_size, dropout=dropout_rate, num_heads=num_heads, batch_first=True)

        self.ffn = nn.Linear(embedding_size, out_dim)
        self.position_encoding = nn.Embedding(max_sent_len, embedding_size, padding_idx=0)
        self.num_layers=num_layers

    def forward(self, inputs, positional_indices):
        embed_output = self.embedding(inputs)
        position_encoding = self.position_encoding(positional_indices)
        attn_in = embed_output + position_encoding
        padding_mask = (positional_indices == 0)
        bs = len(attn_in)
        # sent_embedding = torch.empty(size=(bs, embed_output.size()[1], embed_output.size()[3]))
        # for i in range(bs):
        #     query = key = value = attn_in[i]
        #     attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=padding_mask[i])
        #     sent_embedding[i] = attn_output[:, 0, :]

        attn_in = attn_in.flatten(0, 1)
        padding_mask = padding_mask.flatten(0, 1)
        for i in range(self.num_layers):
            query = key = value = attn_in
            attn_in, _ = self.multihead_attn(query, key, value, key_padding_mask=padding_mask)
        attn_output = attn_in
        sent_embedding = attn_output[:, 0, :]

        # force pad attention outputs
        # padding_mask = (inputs == 1).view(-1, *inputs.size()[2:])
        # mask_for_pads = (~padding_mask).unsqueeze(-1).expand(-1, -1, attn_output.size(-1))
        # attn_output *= mask_for_pads
        # sent_embedding = torch.mean(attn_output, dim=1, keepdim=False)
        sent_embedding = sent_embedding.reshape(*inputs.size()[0:2], -1)
        # sent_embedding = torch.na n_to_num(sent_embedding)
        sent_embedding = self.ffn(sent_embedding)
        return sent_embedding


class EncoderFCN(nn.Module):
    def __init__(self, encoder, fcn):
        super(EncoderFCN, self).__init__()
        self.encoder = encoder
        self.fcn = fcn

    def forward(self, inputs, lens, trans_pos_indices, word_pos_indices):
        encoder_out, attn_scores = self.encoder.forward(inputs, lens, trans_pos_indices, word_pos_indices)
        output = self.fcn.forward(encoder_out)
        return output, attn_scores


class EncoderMTL(nn.Module):
    def __init__(self, encoder, fcn, head, N):
        super(EncoderMTL, self).__init__()
        self.encoder = encoder
        self.fcn = fcn
        self.mtl_head = get_clones(head, N)
        self.N = N

    def forward(self, inputs, lens, trans_pos_indices, word_pos_indices):
        outputs = []
        encoder_out, attn_scores, value = self.encoder.forward(inputs, lens, trans_pos_indices, word_pos_indices)
        fcn_output = self.fcn.forward(encoder_out)
        outputs.append(fcn_output)
        for i in range(self.N):
            outputs.append(self.mtl_head[i].forward(encoder_out))
        return outputs, attn_scores, value


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


