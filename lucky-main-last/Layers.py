import torch
import torch.nn as nn
from Sublayers import FeedForward, MultiHeadAttention, Norm
#layer norm即在一个单词的不同维度间进行norm操作，而非不同单词之间。
#LayerNorm（X+MultiHeadAttention（X））或者LayerNorm（X+FeedForward（X））  体现ResNet的思想
#FeedForward max(0,XW1+b1)W2+b2  两层全连接层，第一层的激活函数为relu 第二层没有激活函数  输出结果与x一个维度
#经过多头注意力机制后输出的结果与输入的x的维度相同
#Multi-Head Attention 输出的矩阵Z与其输入的矩阵X的维度是一样的。
class EncoderLayer(nn.Module):#一个encoder层 之后会有六个的组合
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)#在单个词向量的维度进行layernorm
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 由于层与层之间要做norm，所以上来先做norm
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.norm_4 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_3 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)#FeedForward max(0,XW1+b1)W2+b2 两层全连接层，第一层的激活函数为relu 第二层没有激活函数  输出结果与x一个维度

    def forward(self, x, f_e_outputs,l_e_outputs, f_src_mask, l_src_mask, trg_mask):#src_mask [BatchSize, seqLen, seqLen] trg_mask[BatchSize, seqLen, seqLen] 上三角的都要被mask
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))#前三个参数代表将要乘WQ WK WV的三个矩阵  第一个多头注意力需要mask
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, f_e_outputs, f_e_outputs, f_src_mask))#第二个多头注意力机制不同的是  KV阵是来自编码器的输出  第二个多头注意力不需要mask
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.attn_2(x2, l_e_outputs, l_e_outputs, l_src_mask))  # 第二个多头注意力机制不同的是  KV阵是来自编码器的输出  第二个多头注意力不需要mask
        x2 = self.norm_4(x)
        x = x + self.dropout_4(self.ff(x2))
        return x