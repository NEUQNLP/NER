#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号AI壹号堂
#个人微信 wibrce
#Author 杨博
import torch
import torch.nn as nn
import torch.nn.functional as F
from BruceNRE.models import BasicModule, Embedding

class BruceCNN(BasicModule):
    def __init__(self, vocab_size, config):
        super(BruceCNN, self).__init__()
        self.model_name = 'BruceCNN'
        self.out_channels = config.out_channels
        self.kernel_size = config.kernel_size
        self.word_dim = config.word_dim
        self.pos_size = config.pos_size
        self.pos_dim = config.pos_dim
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        self.out_dim = config.relation_type

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size]
        for k in self.kernel_size:
            assert k % 2 == 1, 'k 必须是奇数'

        self.embedding = Embedding(vocab_size, self.word_dim, self.pos_size, self.pos_dim)

        self.input_dim = self.word_dim + self.pos_dim * 2

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.input_dim,
                      out_channels=self.out_channels,
                      kernel_size=k,
                      padding=k//2,
                      bias=None
                      ) for k in self.kernel_size
        ])

        self.conv_dim = len(self.kernel_size) * self.out_channels
        self.fc1 = nn.Linear(self.conv_dim, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.hidden_size, self.out_dim)

    def forward(self, input):
        """
        :param self:
        :param input: word_ids, headpos, tailpos, mask
        :return:
        """
        *x, mask = input

        x = self.embedding(x)
        x = torch.transpose(x, 1, 2)

        x = [F.leaky_relu(conv(x)) for conv in self.convs]
        x = torch.cat(x, dim=1)
        s_len = x.size(-1)
        x = F.max_pool1d(x, s_len)
        x = x.squeeze(-1)

        x = self.dropout(x)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x


