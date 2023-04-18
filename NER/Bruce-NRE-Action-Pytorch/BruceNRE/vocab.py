#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号AI壹号堂
#个人微信 wibrce
#Author 杨博

init_tokens = ['PAD', 'UNK']

class Vocab(object):
    def __init__(self, name, init_tokens = init_tokens):
        self.name = name
        self.init_tokens = init_tokens
        self.trimed = False
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {}
        self.count = 0
        self.add_init_tokens()

    def add_init_tokens(self):
        for token in self.init_tokens:
            self.add_word(token)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.count
            self.word2count[word] = 1
            self.idx2word[self.count] = word
            self.count = self.count + 1
        else:
            self.word2count[word] = self.word2count[word] + 1

    def add_sentences(self, sentences):
        for word in sentences:
            self.add_word(word)

    def trim(self, min_freq = 2):
        """
        当词频低于2的时候要从词库中删除
        :param min_freq:
        :return:
        """
        if self.trimed:
            return
        self.trimed = True

        keep_words = []
        new_words = []
        for k, v in self.word2count.items():
            if v >= min_freq:
                keep_words.append(k)
                new_words.extend([k]*v)

        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {}
        self.count = 0
        self.add_init_tokens()
        for word in new_words:
            self.add_word(word)









