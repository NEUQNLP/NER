import torch
from Batch import nopeak_mask
import torch.nn.functional as F
import math


def init_vars(f_c_sentence, l_c_sentence, model, SRC, TRG, opt):
    
    init_tok = TRG.vocab.stoi['<sos>']
    # 升维 从 [[example1],[example2]] 变为 [[[example1],[example2]]]
    f_src_mask = (f_c_sentence != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    l_src_mask = (l_c_sentence != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    # 形状是啥？
    # e_output : batch-size * d_model
    # e_output = model.encoder(src, src_mask)
    f_e = model.encoder_l2r(f_c_sentence,f_src_mask)
    l_e = model.encoder_r2l(l_c_sentence,l_src_mask)
    
    # 先传入<sos>
    outputs = torch.LongTensor([[init_tok]]).to(opt.device)
    # if opt.device == 0:
    #     outputs = outputs.cuda()
    
    trg_mask = nopeak_mask(1, opt)
    
    out = model.out(model.decoder(outputs, f_e, l_e, f_src_mask,l_src_mask, trg_mask))
    out = F.softmax(out, dim=-1)

    # print("out:",out)
    
    # topk函数求tensor中某个dim的前k大的index
    probs, ix = out[:, -1].data.topk(opt.k)
    # print("pre_probs", probs.size())
    # print("ix[0]", ix[0].size())
    # 这里probs.data[0]选择的是最大的那个,返回[[max1,max2,...,maxn]]
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    # print("pre_log_scores",log_scores.size())
    outputs = torch.zeros(opt.k, opt.max_len).long().to(opt.device)
    # if opt.device == 0:
    #     outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    # print("outputs:", outputs)
    # print("outputs_size", outputs.size())
    f_e_outputs = torch.zeros(opt.k, f_e.size(-2),f_e.size(-1)).to(opt.device)
    l_e_outputs = torch.zeros(opt.k, l_e.size(-2), l_e.size(-1)).to(opt.device)
    # if opt.device == 0:
    #     e_outputs = e_outputs.cuda()
    # 降维度
    f_e_outputs[:, :] = f_e[0]
    l_e_outputs[:, :] = l_e[0]
    # print("e_output", e_output.size())
    # print("e_outputs", e_outputs.size())
    
    # outputs是beam search的输出，k*max_len
    # e_outputs是encoder的输出，但size是 k*batchsize*d_model
    return outputs, f_e_outputs, l_e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    # print("i",i)
    #out[:,-1] k句话中，每句话最后一个单词概率分布
    probs, ix = out[:, -1].data.topk(k)
    # print("out",out.size()) # i=2 out:3*2*21634
    # print("out[:,-1]",out[:,-1].size()) # i=2,....  out[:,-1]:3*21593
    # print("probs",probs.size()) #i=2,...    probs:3*3    i=1 probs:1*3
    # 此时的scores加上之前的scores
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    # print("log_probs",log_probs.size()) #i=2,.... log_probs:3*3  i=1 log_probs=1*3
    # 最终选出当前时间步的topk
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]
    # print("outputs[]",outputs[:,i])
    log_scores = k_probs.unsqueeze(0)
    # print("log_scores",log_scores.size())
    return outputs, log_scores

def beam_search(f_c_sentence, l_c_sentence, model, SRC, TRG, opt, beam_index=False):
    
    outputs,f_e_outputs,l_e_outputs ,log_scores = init_vars(f_c_sentence, l_c_sentence, model, SRC, TRG, opt)
    eos_tok = TRG.vocab.stoi['<eos>']
    f_src_mask = (f_c_sentence != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    l_src_mask = (l_c_sentence != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    ind = None
    for i in range(2, opt.max_len):
    
        trg_mask = nopeak_mask(i, opt)

        # 这里输入decoder的e_outputs升维了，第一维是k
        # 为什么要用升维之后的

        # print("outputs:", outputs.size())
        # print("e_outputs", e_outputs)
        # print("src_mask", src_mask.size())
        # print("trg_mask", trg_mask.size())

        out = model.out(model.decoder(outputs[:,:i],f_e_outputs, l_e_outputs,f_src_mask,l_src_mask, trg_mask))

        out = F.softmax(out, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k)
        # print("outputs",outputs.size())
        # 找出输出中<eos>的位置(index)
        # 返回是这个样子的：
        # [ [false, false, false, true, true....],
        #   [false, false, false, false, false, true....]  
        # ]
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        # sentence_lengths存放每一个句子的长度
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).to(opt.device)
        # 二维的nonzero()返回每一个非零（True）元素的坐标位置
        # 对于每一个坐标而言：vec[0]代表第几个句子，vec[1]代表哪个位置是<eos>，vec[0]相同的情况下，
        # 最先出现的vec[1]就是这个句子的长度
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        # 存当前已经有的句子数量
        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        # 如果已经满足topk的要求 
        if num_finished_sentences == opt.k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    
    if beam_index is True:
        try:
            length = (outputs[0]==eos_tok).nonzero()[0]
            return [tok.item() for tok in outputs[0][1:length]]
        # 防止没有eos标志
        except IndexError:
            length = len(outputs[0])
            return [tok.item() for tok in outputs[0][1:length]]

    else:
        print("outputs", outputs)
        if ind is None:
            length = (outputs[ind]==eos_tok).nonzero()[0]
            print("return", ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]]))
            return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
        
        else:
            length = (outputs[ind]==eos_tok).nonzero()[0]
            print("return", ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]]))
            return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])
