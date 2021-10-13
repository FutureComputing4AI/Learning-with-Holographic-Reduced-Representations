#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/9
@author yrh

"""

import torch
import torch.nn as nn

from deepxml.modules import *
from deepxml.lib.embeddings import get_vectors
from deepxml.lib.mathops import get_appx_inv, circular_conv, complexMagProj


__all__ = ['AttentionRNN', 'FastAttentionRNN']


class Network(nn.Module):
    """

    """
    def __init__(self, emb_size, vocab_size=None, emb_init=None,
                 emb_trainable=True, padding_idx=0, emb_dropout=0.2,
                 **kwargs):
        super(Network, self).__init__()
        self.emb = Embedding(vocab_size, emb_size, emb_init, emb_trainable, padding_idx, emb_dropout)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class AttentionRNN(Network):
    """

    """
    def __init__(self, labels_num, emb_size, hidden_size, layers_num,
                 linear_size, dropout, use_spn, spn_dim, no_grad,
                 without_negative, **kwargs):
        super(AttentionRNN, self).__init__(emb_size, **kwargs)
        self.use_spn = use_spn
        if self.use_spn:
            self.label_size = spn_dim
            self.no_grad = no_grad
            self.without_negative = without_negative
        else:
            self.label_size = labels_num

        self.num_labels = labels_num
        self.lstm = LSTMEncoder(emb_size, hidden_size, layers_num, dropout)
        self.attention = MLAttention(self.label_size, hidden_size * 2)
        self.linear = MLLinear([hidden_size * 2] + linear_size, self.label_size)

        if self.use_spn:
            self.create_label_embedding() # Create the labels.

    def create_label_embedding(self):
        # Class labels. # +1 for the END of LIST Label.
        self._class_vectors = get_vectors(self.num_labels + 1, self.label_size)

        # Initialize embedding layer.
        self.class_vec = nn.Embedding(self.num_labels + 1, self.label_size)
        self.class_vec.load_state_dict({'weight': self._class_vectors})
        self.class_vec.weight.requires_grad = False

        # Initialize weights vector.
        weights = torch.ones((self.num_labels + 1, 1), dtype=torch.int8)
        weights[self.num_labels] = 0 # Padding vector is made 0.
        self.class_weights = nn.Embedding(self.num_labels + 1, 1)
        self.class_weights.load_state_dict({'weight': weights})
        self.class_weights.weight.requires_grad = False

        # P & N vectors.
        p_n_vec = get_vectors(2, self.label_size, ortho=True)
        if self.no_grad:
            print("P & N vectors WILL NOT be updated while training...")
            self.p = nn.Parameter(p_n_vec[0], requires_grad=False)
            self.n = nn.Parameter(p_n_vec[1], requires_grad=False)
        else:
            print("P & N vectors WILL be updated while training...")
            self.p = nn.Parameter(p_n_vec[0], requires_grad=True)
            self.n = nn.Parameter(p_n_vec[1], requires_grad=True)


    def inference(self, s, batch_size, positive=True):
        #(batch, dims)
        if positive:
            vec = self.p.unsqueeze(0).expand(batch_size, self.label_size)
        else:
            vec = self.n.unsqueeze(0).expand(batch_size, self.label_size)

        # vec = complexMagProj(vec)
        inv_vec = get_appx_inv(vec)
        y = circular_conv(inv_vec, s) #(batch, dims)
        y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
        return y

    def spp_loss(self, s, target):
        """
        Train with SPP.
        """
        pos_classes = self.class_vec(target)   #(batch, no_label, dims)
        pos_classes = pos_classes * self.class_weights(target)        # exit(0)

        # Normalize the class vectors.
        # tgt_shape = pos_classes.shape
        # pos_classes = torch.reshape(pos_classes, (tgt_shape[0] * tgt_shape[1],
        #                                           tgt_shape[2]))
        # pos_classes = torch.reshape(complexMagProj(pos_classes), (tgt_shape[0], tgt_shape[1],
        #                                            tgt_shape[2]))

        # Remove the padding idx vectors.
        # pos_classes = pos_classes.to(device)

        # Positive prediction loss
        convolve = self.inference(s, target.size(0))
        cosine = torch.matmul(pos_classes, convolve.unsqueeze(1).transpose(-1, -2)).squeeze(-1)
        J_p = torch.mean(torch.sum(1 - torch.abs(cosine), dim=-1))

        # Negative prediction loss.
        J_n = 0.0
        if self.without_negative is False:
            convolve = self.inference(s, target.size(0), positive=False)
            cosine = torch.matmul(pos_classes, convolve.unsqueeze(1).transpose(-1, -2)).squeeze(-1)
            J_n = torch.mean(torch.sum(torch.abs(cosine), dim=-1))

        # Total Loss.
        loss = J_n + J_p
        return loss


    def forward(self, inputs, **kwargs):
        emb_out, lengths, masks = self.emb(inputs, **kwargs)
        rnn_out = self.lstm(emb_out, lengths)   # N, L, hidden_size * 2 (Bidirectional RNN)
        attn_out = self.attention(rnn_out, masks)      # N, labels_num, hidden_size * 2
        return self.linear(attn_out)

class FastAttentionRNN(Network):
    """

    """
    def __init__(self, labels_num, emb_size, hidden_size, layers_num, linear_size, dropout, parallel_attn, **kwargs):
        super(FastAttentionRNN, self).__init__(emb_size, **kwargs)
        self.lstm = LSTMEncoder(emb_size, hidden_size, layers_num, dropout)
        self.attention = FastMLAttention(labels_num, hidden_size * 2, parallel_attn)
        self.linear = MLLinear([hidden_size * 2] + linear_size, 1)

    def forward(self, inputs, candidates, attn_weights: nn.Module, **kwargs):
        emb_out, lengths, masks = self.emb(inputs, **kwargs)
        rnn_out = self.lstm(emb_out, lengths)   # N, L, hidden_size * 2
        attn_out = self.attention(rnn_out, masks, candidates, attn_weights)     # N, sampled_size, hidden_size * 2
        return self.linear(attn_out)
