#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号AI壹号堂
#个人微信 wibrce
#Author 杨博

#系统相关
import argparse
import os

#框架相关
import torch
import torch.optim as optim
import torch.nn as nn

#自定义
from BruceNRE.config import config
from BruceNRE.utils import make_seed, load_pkl, get_result
from BruceNRE.process import split_sentences, bulid_data
from BruceNRE import models


__Models__ = {
    "BruceCNN":models.BruceCNN,
    "BruceBiLSTM":models.BruceBiLSTM,
    "BruceBiLSTMPro":models.BruceBiLSTMPro
}

parser = argparse.ArgumentParser(description='关系抽取')
parser.add_argument('--model_name', type=str, default='BruceCNN', help='model name')
args = parser.parse_args()

if __name__ == "__main__":
    model_name = args.model_name if args.model_name else config.model_name

# 初始化随机数设置
make_seed(config.seed)

# 计算设备设置
if config.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda', config.gpu_id)
else:
    device = torch.device('cpu')
print(torch.cuda.is_available())

vocab_path = os.path.join(config.out_path, 'vocab.pkl')
train_data_path = os.path.join(config.out_path, 'train.pkl')

vocab = load_pkl(vocab_path, 'vocab')
vocab_size = len(vocab.word2idx)

model = __Models__[model_name](vocab_size, config)
print(model)

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', factor=config.decay_rate, patience=config.decay_patience)
loss_fn = nn.CrossEntropyLoss()

model.load("checkpoints/BruceCNN_epoch14_0104_14_15_03.pth")
print("******************开始预测*********************")


while True:
    text = input("input:")
    data = text.split("#")
    entity1 = data[1]
    entity2 = data[3]
    head_index = data[0].index(entity1)
    tail_index = data[0].index(entity2)
    data.insert(3, head_index)
    data.insert(6, tail_index)
    raw_data = []
    raw_data.append(data)
    new_test = split_sentences(raw_data)
    sents, head_pos, tail_pos, mask_pos = bulid_data(new_test, vocab)
    x = [torch.tensor(sents), torch.tensor(head_pos), torch.tensor(tail_pos), torch.tensor(mask_pos)]
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        y_pred = y_pred.argmax(dim=-1)
        result = get_result(entity1, entity2, y_pred.numpy()[0])
        print(result)


