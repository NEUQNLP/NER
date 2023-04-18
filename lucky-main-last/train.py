import argparse
import time
from numpy import dtype
import torch
from Beam import beam_search
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import dill as pickle
import sys
# from translate import get_synonym
from sacrebleu.metrics import BLEU, CHRF, TER
# from translate import multiple_replace
from torch.autograd import Variable
import itertools


def train_model(model, opt):
    
    with open('prediction.txt', 'a', encoding='utf-8') as f:
        f.write("training model...\n")
    
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()
    
    best_model_param = model.state_dict()
    best_bleu = 0.0
    wait_count=1
    best_is_saved = False
    total_loss=0
    
    for epoch in range(opt.epochs):

        model.train()
        total_loss = 0
        # if opt.floyd is False:
        #     print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
        #     ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')

        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')

        for i, batch in enumerate(opt.train):
            # batch从: src:[[1,2,3],    [1,4]是一个样本, [2,5]是一个样本
            #               [4,5,6]]
            #          trg:[[1,8,7]     [1,2]是[1,4]的对应标签, [8,5]是[2,5]对应的标签
            #               [2,5,3]]
            # transpose变成横的（和转置一样，这里可以改成batch.src.T）
            src = batch.src.transpose(0,1).to(opt.device)
            trg = batch.trg.transpose(0,1).to(opt.device)

            trg_input = trg[:, :-1]

            f_src = src
            l_src = torch.flip(src, [1])

            # 构造mask
            f_src_mask, trg_mask = create_masks(f_src, trg_input, opt)
            # print("trgmask  1",trg_mask.size())
            l_src_mask, trg_mask = create_masks(l_src, trg_input, opt)
            # print("trgmask   2",trg_mask.size())
            # 得到预测值
            # print("f_src",f_src.size())
            # print("l-src",l_src.size())
            # print("trg",trg.size())
            # print("f_src_mask",f_src_mask.size())
            # print("l_src_mask",l_src_mask.size())
            # print("trg_mask",trg_mask.size())
            preds = model(f_src, l_src, trg_input,  f_src_mask,l_src_mask, trg_mask)

            # trg[:,1:]的目的是去掉<sos>
            # contiguous()方法语义上是“连续的”，经常与torch.permute()、
            # torch.transpose()、torch.view()方法一起使用
            # 可以看出contiguous方法改变了多维数组在内存中的存储顺序，以便配合view方法使用
            # torch.contiguous()方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列。
            # view把数据展平
            ys = trg[:, 1:].contiguous().view(-1)

            # print("the model output size is ", preds.size())
            # print("the label size is ", ys.size())
            # 梯度清零
            opt.optimizer.zero_grad()
            # 求得损失函数
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            # 求梯度
            loss.backward()
            # 更新参数
            opt.optimizer.step()
            if opt.SGDR == True:
                opt.sched.step()

            total_loss += loss.item()

            if (i + 1) % opt.printevery == 0:
                 p = int(100 * (i + 1) / opt.train_len)
                 avg_loss = total_loss/opt.printevery

                 print("loss:",avg_loss)
                 if opt.floyd is False:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.6f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss))
                 else:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.6f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss))
                 total_loss = 0

            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()
    
        # 在val集上面验证效果，早停
        model.eval()
        with torch.no_grad():
            sys=[]
            refs=[]
            temp_refs=[]
            for i, batch in enumerate(opt.val): 
                src = batch.src.transpose(0,1).to(opt.device)
                trg = batch.trg.transpose(0,1).to(opt.device)
                
                # trg_input = trg[:, :-1]
                
                # # 在构造mask的时候需要有<sos>，不需要<eos>
                # src_mask, trg_mask = create_masks(src, trg_input, opt)

                # preds = model(src, trg_input, src_mask, trg_mask)
                
                # _, preds_index = preds.max(-1)
                preds_index=[]
                # print("src_size",src.size())
                f_src = src
                l_src = torch.flip(src, [1])

                for f,l in zip(f_src,l_src):
                    # print("src_sentence_size",src_sentence.size())
                    f_src_sentence = f.tolist()
                    l_src_sentence = l.tolist()
                    f_c_sentence = Variable(torch.LongTensor([f_src_sentence])).to(opt.device)
                    l_c_sentence = Variable(torch.LongTensor([l_src_sentence])).to(opt.device)
                    preds_index.append(beam_search(f_c_sentence, l_c_sentence,model, opt.SRC, opt.TRG, opt, beam_index=True))
                
                # print("preds_index:",preds_index)
                ys = trg[:, 1:]

                
                prediction_list, true_list = batch_ids_to_sentences(preds_index, ys, opt)
                
                for sentence in prediction_list:
                    sys.append(sentence)
                for sentence in true_list[0]:
                    temp_refs.append(sentence)
                
                # # 展开
                # preds_index = list(itertools.chain(*preds_index))
                # preds_index = torch.Tensor(preds_index).type(torch.float32)
                
                # ys = ys.contiguous().view(-1)
                # print(preds_index)
                # print(ys)
                
                # loss = F.cross_entropy(preds_index.view(-1, 1), ys, ignore_index=opt.trg_pad)
                
                # total_loss += loss.item()

            refs.append(temp_refs)

            # print("sys:", sys)
            # print("refs:", refs)

            score = bleu_score(sys, refs)
            
            with open('prediction.txt', 'a', encoding='utf-8') as f:
                f.write("score: " + str(score) + '\n')
            
            if score>best_bleu:
                best_bleu=score
                best_model_param=model.state_dict()
                wait_count=1
            else:
                wait_count+=1
                if wait_count >= opt.waiting_count:
                    torch.save(best_model_param, 'weights/best_model_weights')
                    best_is_saved = True
                    # break for epoch in range(opt.epochs)
                    break

    if best_is_saved is False:
        torch.save(best_model_param, 'weights/best_model_weights')

def test_model(model, opt):
    
    with open('prediction.txt', 'a', encoding='utf-8') as f:
        f.write("testing model...\n")
        
    model = get_model(opt, len(opt.SRC.vocab), len(opt.TRG.vocab))
    model.eval()  
    with torch.no_grad():
        sys=[]
        refs=[]
        temp_refs=[]
        for i, batch in enumerate(opt.test): 
            src = batch.src.transpose(0,1).to(opt.device)
            trg = batch.trg.transpose(0,1).to(opt.device)
            
            preds_index=[]
            
            # for src_sentence in src:
            #     src_sentence = src_sentence.tolist()
            #     src_sentence = Variable(torch.LongTensor([src_sentence])).to(opt.device)
            #     preds_index.append(beam_search(src_sentence, model, opt.SRC, opt.TRG, opt, beam_index=True))

            f_src = src
            l_src = torch.flip(src, [1])

            for f, l in zip(f_src, l_src):
                # print("src_sentence_size",src_sentence.size())
                f_src_sentence = f.tolist()
                l_src_sentence = l.tolist()
                f_c_sentence = Variable(torch.LongTensor([f_src_sentence])).to(opt.device)
                l_c_sentence = Variable(torch.LongTensor([l_src_sentence])).to(opt.device)
                preds_index.append(
                    beam_search(f_c_sentence, l_c_sentence, model, opt.SRC, opt.TRG, opt, beam_index=True))

            ys = trg[:, 1:]
            prediction_list, true_list = batch_ids_to_sentences(preds_index, ys, opt)
            
            for sentence in prediction_list:
                sys.append(sentence)
            for sentence in true_list[0]:
                temp_refs.append(sentence)
            

        refs.append(temp_refs)

        # print("sys:", sys)
        # print("refs:", refs)
        with open('sys.txt', 'w', encoding='utf-8') as f:
            for k in sys:
                f.write("sys: " + str(k) + '\n')
        with open('refs.txt', 'w', encoding='utf-8') as f:
            for k in refs:
                f.write("refs: " + str(k) + '\n')
        score = bleu_score(sys, refs)

        print("score:", score)
        with open('prediction.txt', 'a', encoding='utf-8') as f:
            f.write("Test bleu score: " + str(score) + '\n')

def batch_ids_to_sentences(preds_index, ys, opt):
    
    prediction=[]
    true_list=[]
    
    for sentence in preds_index:
        prediction_string=""
        # print("sentence",sentence)
        for tok in sentence:
            if(opt.TRG.vocab.itos[tok]=='<eos>'):
                break
            prediction_string = prediction_string + opt.TRG.vocab.itos[tok] + " "
        prediction.append(prediction_string)
    
    temp_pre=[]
    for sentence in ys:
        true_string=""
        for tok in sentence:
            if(opt.TRG.vocab.itos[tok]=='<eos>'):
                break
            true_string = true_string + opt.TRG.vocab.itos[tok] + " "
        temp_pre.append(true_string)
    true_list.append(temp_pre)
    
    # with open('prediction.txt', 'a', encoding='utf-8') as f:
    #     f.write("prediction:{}\n".format(prediction))
    #     f.write("true:{}\n".format(true_list))
            
    return prediction, true_list


def bleu_score(sys, refs):
    bleu = BLEU()
    x = bleu.corpus_score(sys, refs)
    return x.score

def main():


    parser = argparse.ArgumentParser()
    # 原始的参数：
    parser.add_argument('-src_data', required=False)
    parser.add_argument('-trg_data', required=False)
    parser.add_argument('-src_lang', required=False)
    parser.add_argument('-trg_lang', required=False)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batch-size', type=int, default=32)
    parser.add_argument('-printer', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valet', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=100)
    opt = parser.parse_args()
    
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt.src_train = "data/train_en.txt"
    opt.trg_train = "data/train_de.txt"
    opt.src_test = "data/test_en.txt"
    opt.trg_test = "data/test_de.txt"
    opt.src_val = "data/validation_en.txt"
    opt.trg_val = "data/validation_de.txt"
    opt.src_lang = "en_core_web_lg"
    opt.trg_lang = "de_core_news_lg"
    # opt.device = 0 if opt.no_cuda is False else -1
    # # device为0表示可以使用cuda
    # if opt.device == 0:
    #     assert torch.cuda.is_available()

    read_data(opt)

    SRC, TRG = create_fields(opt)
    
    opt.SRC = SRC
    opt.TRG = TRG
    
    opt.train, opt.test, opt.val = create_dataset(opt, SRC, TRG)
    
    # instantiate the Transformer
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        # 添加这一点将实现随机梯度下降与重启，使用余弦退火
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    
    if opt.load_weights is not None and opt.floyd is not None:
        os.mkdir('weights01')
        # 将对象SRC\TRG保存到文件中去
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    
    # 早停的一个参数
    opt.waiting_count=20
    train_model(model, opt)
    
    # if opt.floyd is False:
    #     promptNextAction(model, opt, SRC, TRG)
        
    test_model(model, opt)


def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt, SRC, TRG):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    
    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'

    while True:
        save = yesno(input('training complete, save results? [y/n] : '))
        if save == 'y':
            while True:
                if saved_once != 0:
                    res = yesno("save to same folder? [y/n] : ")
                    if res == 'y':
                        break
                dst = input('enter folder name to create for weights (no spaces) : ')
                if ' ' in dst or len(dst) < 1 or len(dst) > 30:
                    dst = input("name must not contain spaces and be between 1 and 30 characters length, enter again : ")
                else:
                    try:
                        os.mkdir(dst)
                    except:
                        res= yesno(input(dst + " already exists, use anyway? [y/n] : "))
                        if res == 'n':
                            continue
                    break
            
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
            if saved_once == 0:
                pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
                saved_once = 1
            
            print("weights and field pickles saved to " + dst)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            train_model(model, opt)
        else:
            print("exiting program...")
            break

    # for asking about further training use while true loop, and return
if __name__ == "__main__":
    main()
