#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号AI壹号堂
#个人微信 wibrce
#Author 杨博

# 系统相关
import os
import codecs
import csv
import json

#第三方
import jieba
# 自定义
from BruceNRE.utils import load_csv, load_json, ensure_dir, sava_pkl
from BruceNRE.config import config
from BruceNRE.vocab import Vocab

# 判断数据集中是否存在关系字段
def exist_realtion(file, file_type):
    with codecs.open(file, encoding='utf-8') as f:
        if file_type == 'csv':
            f = csv.DictReader(f)
        for line in f:
            keys = list(line.keys())
            try:
                num = keys.index('relation')
            except:
                num = -1
            return int(num)
# 分词
def split_sentences(raw_data):
    new_data_list = []
    jieba.add_word("HEAD")
    jieba.add_word("TAIL")
    for data in raw_data:
        head, tail = data[2], data[5]
        new_sent = data[0].replace(data[1], 'HEAD', 1)
        new_sent = new_sent.replace(data[4], 'TAIL', 1)
        new_sent = jieba.lcut(new_sent)
        head_pos, tail_pos = new_sent.index("HEAD"), new_sent.index("TAIL")
        new_sent[head_pos] = head
        new_sent[tail_pos] = tail
        data.append(new_sent)
        data.append([head_pos, tail_pos])
        new_data_list.append(data)
    return new_data_list

# 构建词典
def bulid_vocab(raw_data, out_path):
    if config.word_segment:
        vocab = Vocab('word')
        for data in raw_data:
            vocab.add_sentences(data[-2])
    else:
        vocab = Vocab('char')
        for data in raw_data:
            vocab.add_sentences(data[0])
    vocab.trim(config.min_freq)

    ensure_dir(out_path)

    vocab_path = os.path.join(out_path, 'vocab.pkl')
    vocab_txt = os.path.join(out_path, 'vocab.txt')
    sava_pkl(vocab_path, vocab, 'vocab')

    with codecs.open(vocab_txt, 'w', encoding='utf-8') as f:
        f.write(os.linesep.join([word for word in vocab.word2idx.keys()]))
    return vocab, vocab_path

# 获取位置编码特征
def get_pos_feature(sent_len, entity_pos, entity_len, pos_limit):
    """
    :param sent_len:
    :param entity_pos:
    :param entity_len:
    :param pos_limit:
    :return:
    """
    left = list(range(-entity_pos, 0))
    middle = [0] * entity_len
    right = list(range(1, sent_len - entity_pos - entity_len + 1))
    pos = left + middle + right
    for i, p in enumerate(pos):
        if p > pos_limit:
            pos[i]  = pos_limit
        if p < -pos_limit:
            pos[i] = -pos_limit
    pos = [p + pos_limit + 1 for p in pos]
    return pos

def get_mask_feature(entities_pos, sen_len):
    """
    获取mask编码
    :param entities_pos:
    :param sen_len:
    :return:
    """
    left = [1] * (entities_pos[0] + 1)
    middle = [2] * (entities_pos[1] - entities_pos[0] - 1)
    right = [3] * (sen_len - entities_pos[1])
    return left + middle + right

def bulid_data(raw_data, vocab):
    sents = []
    head_pos = []
    tail_pos = []
    mask_pos = []

    if vocab.name == 'word':
        for data in raw_data:
            sent = [vocab.word2idx.get(w,1) for w in data[-2]]
            pos = list(range(len(sent)))
            head, tail = int(data[-1][0]), int(data[-1][-1])
            entities_pos = [head, tail] if tail > head else [tail, head]
            head_p = get_pos_feature(len(sent), head, 1, config.pos_limit)
            tail_p = get_pos_feature(len(sent), tail, 1, config.pos_limit)
            mask_p = get_mask_feature(entities_pos, len(sent))
            sents.append(sent)
            head_pos.append(head_p)
            tail_pos.append(tail_p)
            mask_pos.append(mask_p)
    else:
        for data in raw_data:
            sent = [vocab.word2idx.get(w, 1) for w in data[0]]
            head, tail = int(data[3]), int(data[6])
            head_len, tail_len = len(data[1]), len(data[4])
            entities_pos = [head, tail] if tail > head else [tail, head]
            head_p = get_pos_feature(len(sent), head, head_len, config.pos_limit)
            tail_p = get_pos_feature(len(sent), tail, tail_len, config.pos_limit)
            mask_p = get_mask_feature(entities_pos, len(sent))
            sents.append(sent)
            head_pos.append(head_p)
            tail_pos.append(tail_p)
            mask_pos.append(mask_p)
    return sents, head_pos, tail_pos, mask_pos

def relation_tokenize(relations, file):
    """

    :param relations:
    :param file:
    :return:
    """
    relations_list = []
    relations_dict = {}
    out = []
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            relations_list.append(line.strip())
    for i, rel in enumerate(relations_list):
        relations_dict[rel] = i
    for rel in relations:
        out.append(relations_dict[rel])
    return out


# 数据预处理
def process(data_path, out_path, file_type):
    print("*****数据预处理开始*****")
    file_type = file_type.lower()
    assert file_type in ['csv', 'json']

    print("*****加载原始数据*****")

    train_fp = os.path.join(data_path, 'train.' + file_type)
    test_fp = os.path.join(data_path, 'test.' + file_type)
    relation_fp = os.path.join(data_path, 'relation.txt')

    relation_place = exist_realtion(train_fp, file_type)

    if relation_place > -1:
        if file_type == 'csv':
            train_raw_data = load_csv(train_fp)
            test_raw_data = load_csv(test_fp)
        else:
            train_raw_data = load_json(train_fp)
            test_raw_data = load_json(test_fp)

        train_relations = []
        test_relations = []

        for data in train_raw_data:
            train_relations.append(data.pop(relation_place))
        for data in test_raw_data:
            test_relations.append(data.pop(relation_place))

        if config.is_chinese and config.word_segment:
            train_raw_data = split_sentences(train_raw_data)
            test_raw_data = split_sentences(test_raw_data)

        print("构建词典")
        vocab, vocab_path = bulid_vocab(train_raw_data, out_path)

        print("构建train模型数据")
        train_sents, train_head_pos, train_tail_pos, train_mask_pos = bulid_data(train_raw_data, vocab);
        print("构建test模型数据")
        test_sents, test_head_pos, test_tail_pos, test_mask_pos = bulid_data(test_raw_data, vocab)

        print("构建关系数据")
        train_relations_token = relation_tokenize(train_relations, relation_fp)
        test_relations_token = relation_tokenize(test_relations, relation_fp)

        ensure_dir(out_path)
        train_data = list(
            zip(train_sents, train_head_pos, train_tail_pos, train_mask_pos, train_relations_token)
        )
        test_data = list(
            zip(test_sents, test_head_pos, test_tail_pos, test_mask_pos, test_relations_token)
        )

        train_data_path = os.path.join(out_path, 'train.pkl')
        test_data_path = os.path.join(out_path, 'test.pkl')

        sava_pkl(train_data_path, train_data, 'train data')
        sava_pkl(test_data_path, test_data, 'test data')

        print("*****数据预处理完成*****")



