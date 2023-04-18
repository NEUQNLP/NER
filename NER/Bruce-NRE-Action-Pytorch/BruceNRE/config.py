#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号AI壹号堂
#个人微信 wibrce
#Author 杨博

"""
模型配置类
"""
class Config(object):
    model_name = "BruceCNN"
    data_path = 'data/origin' #原始数据
    out_path = 'data/out' #数据处理的最终结果保存

    is_chinese = True #是否是中文

    word_segment = True #是否分词

    relation_type = 10 #关系种类

    min_freq = 2 #低频词处理

    #位置编码
    pos_limit = 50   #[-50,50]
    pos_size = 102  # 2*pos_limit + 2

    word_dim = 300
    pos_dim = 10

    hidden_size = 100 # FC 连接数
    dropout = 0.5

    batch_size = 128

    learning_rate = 0.001

    decay_rate = 0.3

    decay_patience = 5

    epoch = 50

    train_log = True
    log_interval = 10

    f1_norm = ['macro', 'micro']

    # CNN
    out_channels = 100
    kernel_size = [3,5]

    # bilstm
    lstm_layers = 2


    #初始化种子
    seed = 1

    use_gpu = True
    gpu_id = 0

config = Config()