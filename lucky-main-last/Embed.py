import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class Embedder(nn.Module):
    """
    Embedding layer
    input is indices, e.g.[1,2,3,4] or a batch of x samples, output is [[(dim is d_model)],[],[],[]] or a batch of the latter example
    Args:
        vocab_size-->d_model
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    """
    what does the word mean? And what is its position in the sentence?

    Args:
        nn ([max_seq_len]): [200]
    """
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)#生成一个位置矩阵
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):#对每一个词向量每两个位置做一次操作
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))  # batch_size就是一个batch中句子总量的维度 #所有维度都是从0开始
        pe = pe.unsqueeze(0)    #0代表第一个维度 从0开始 在第0维增加维度（ max_seq_len ， d_model） -->(batch_size ， max_seq_len ， d_model)
        self.register_buffer('pe', pe)  # 在内存中设置一个常量，同时，模型保存和加载的时候可以写入和读出
 
    
    def forward(self, x):#注意这里x的维度
        # make embeddings relatively larger
        """
        The reason we increase the embedding values before addition is to make the positional encoding relatively smaller. 
        This means the original meaning in the embedding vector won't be lost when we add them together.
        x:  batch_size * seq_len * d_model
        """
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        # if x.is_cuda:
        #     pe.cuda()
        x = x + pe                       #x加上位置编码
        return self.dropout(x)