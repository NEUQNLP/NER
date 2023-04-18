#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号AI壹号堂
#个人微信 wibrce
#Author 杨博
from BruceNRE.models import BasicModule, Embedding
import torch.nn as nn
import torch.nn.functional as F

class BruceBiLSTM(BasicModule):
    def __init__(self, vocab_size, config):
        super(BruceBiLSTM, self).__init__()
        self.model_name = "BruceBiLSTM"
        self.word_dim = config.word_dim
        self.pos_size = config.pos_size
        self.pos_dim = config.pos_dim
        self.lstm_layers = config.lstm_layers
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        self.out_dim = config.relation_type

        self.embedding = Embedding(vocab_size, self.word_dim, self.pos_size, self.pos_dim)

        self.input_dim = self.word_dim + self.pos_dim * 2

        self.bilstm = nn.LSTM(
            input_size = self.input_dim,
            hidden_size = self.hidden_size,
            num_layers = self.lstm_layers,
            dropout = self.dropout,
            bidirectional=True,
            bias=True,
            batch_first=True
        )

        liner_input_dim = self.hidden_size * 2
        self.fc1 = nn.Linear(liner_input_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.out_dim)


    def forward(self, input):
        """
                :param self:
                :param input: word_ids, headpos, tailpos, mask
                :return:
        """
        *x, mask = input
        x = self.embedding(x)
        out_put, _ = self.bilstm(x)
        out_put = out_put[:,-1,:]
        y = F.leaky_relu(self.fc1(out_put))
        y = F.leaky_relu(self.fc2(y))
        return y
