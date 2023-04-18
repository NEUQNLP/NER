from random import random

import torch
import torch.nn as nn
from torch.version import cuda

from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])#deepcopy是真正意义上的复制，深拷贝，被复制对象完全复制一遍作为独立的新个体，新开辟一块空间。
                                                                   #N = 6


class Encoder_L2R(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()  # batch，第几个句子（行）batch_size，每个单词对应的数字≤vocab_size
        self.N = N  # (1,2,4)->(1,2,4,embedding_size=d_model)  d_model是词向量的纬度
        self.embed = Embedder(vocab_size, d_model)  # (batch_size,seq_len,embedding_size)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)  # N代表几个encoder block层
        self.norm = Norm(d_model)
    def forward(self, src, mask):  # src（batch_size,max_seqlen）
        x = self.embed(src)  # （batch_size,max_seqlen）->（batch_size,max_seqlen,embedding_size）
        x = self.pe(x)  # 对x进行了位置编码x = x + pe
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Encoder_R2L(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()                        #batch，第几个句子（行）batch_size，每个单词对应的数字≤vocab_size
        self.N = N                                #(1,2,4)->(1,2,4,embedding_size=d_model)  d_model是词向量的纬度
        self.embed = Embedder(vocab_size, d_model)#(batch_size,seq_len,embedding_size)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)#N代表几个encoder block层
        self.norm = Norm(d_model)
    def forward(self, src, mask):#src（batch_size,max_seqlen）
        x = self.embed(src)#（batch_size,max_seqlen）->（batch_size,max_seqlen,embedding_size）
        x = self.pe(x)#对x进行了位置编码x = x + pe
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)  # N代表几个解码器层
        self.norm = Norm(d_model)

    def forward(self, trg, f_e_outputs, l_e_outputs, f_src_mask,l_src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)  # x = x + pe
        for i in range(self.N):
            x = self.layers[i](x, f_e_outputs, l_e_outputs, f_src_mask,l_src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder_l2r = Encoder_L2R(src_vocab, d_model, N, heads, dropout)#继承，定义类Encoder
        self.encoder_r2l = Encoder_R2L(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, f_src, l_src, trg,  f_src_mask,l_src_mask, trg_mask):#实例化类函数Transformer时需要传入的参数
        f_e_outputs = self.encoder_l2r(f_src, f_src_mask)#实例化类函数encoder
        l_e_outputs = self.encoder_r2l(l_src, l_src_mask)  # 实例化类函数encoder
        d_output = self.decoder(trg, f_e_outputs, l_e_outputs, f_src_mask,l_src_mask,trg_mask)
        output = self.out(d_output)
        # we don't perform softmax on the output as this will be handled 
        # automatically by our loss function
        return output


def get_model(opt, src_vocab, trg_vocab):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
       
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/best_model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) # 初始化参数
    model = model.to(opt.device)
    # if opt.device == 0:
    #     model = model.cuda()
    
    return model

