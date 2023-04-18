#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号AI壹号堂
#个人微信 wibrce
#Author 杨博
import torch
import torch.nn as nn



class Embedding(nn.Module):
    def __init__(self, vocab_size, word_dim, pos_size, pos_dim):
        super(Embedding, self).__init__()
        self.word_embed = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.head_pos_embed = nn.Embedding(pos_size, pos_dim, padding_idx=0)
        self.tail_pos_embed = nn.Embedding(pos_size, pos_dim, padding_idx=0)


    def forward(self, x):
        words, head_pos, tail_pos = x
        word_embed = self.word_embed(words)
        head_embed = self.head_pos_embed(head_pos)
        tail_embed = self.tail_pos_embed(tail_pos)
        feature_embed = torch.cat([word_embed, head_embed, tail_embed], dim=-1)

        return feature_embed

