import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Norm(nn.Module):
    """Normalisation is highly important in deep neural networks. 
    It prevents the range of values in the layers changing too much, meaning the model trains faster and has better ability to generalise.
    """
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):#dim=-1 代表对最后一个维度进行运算  也就是词向量维度
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    # (batch_size * H * seq_len * d_k) * (batch_size * H * d_k *seq_len)    -->   (batch_size * H * seq_len * seq_len)  
                  #矩阵相乘      #将k矩阵的最后两个维度做转置  形成相似度矩阵 seq_len * seq_len  这里进行pad mask 将原来pad（0）的位置作为-无穷 让他的分数趋近于0
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    #相似度矩阵的含义 5行5列  第一行第一个 代表第一个词向量与第一个词向量的相似度得分，第一行第二列，代表第一个词向量和第二个词向量的相似度得分，最后softmax是对行进行操作
    if mask is not None:
        mask = mask.unsqueeze(1)#在第一个维度0 后面增加一个维度h->head
        # 若有mask机制， 在mask值为 0 的地方填充 -1e9 , 让他得分最少
        # print("mask",mask.size())
        # print("scores",scores.size())
        scores = scores.masked_fill(mask == 0, -1e9)#原来被pad过的 比如5*5 最后一维是被pad过的词向量  然后最后一列都被化为-无穷（soft后=0），代表其他词向量与pad的
                                                    #相似度为零
    scores = F.softmax(scores, dim=-1)#对词向量所在的维度进行softmax  也就是矩阵的最后一个维度  按行来做softmax
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)#(batch_size * H * seq_len * seq_len) *(batch_size * H * seq_len * d_k)=(batch_size * H * seq_len * d_k)
    return output                   #与最初的q一样的维度

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        # In multi-head attention we split the embedding vector into N heads,  这里的意思是把词向量维度割成N份，然后并行的并行的进行attention操作
        # so they will then have the dimensions  ->  batch_size * N * seq_len * (d_model / N).
        # This final dimension (d_model / N ) we will refer to as d_k.

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)  # batch_size
        
        # perform linear operation and split into N heads
        # batch_size * seq_len * N * (d_model / N).
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)#qv与k的维度是一样的
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * seq_len * (d_model/N)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
