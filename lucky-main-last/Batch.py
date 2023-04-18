import torch
from torch.cuda import device_of
from torchtext.legacy import data
import numpy as np
from torch.autograd import Variable


def nopeak_mask(size, opt):
    """
    decoder中使用的mask机制

    Returns:
        # [true, false, false]
        # [true, true,  false]
        # [true, true,  true ]
    """
    # np.triu : lower triangle of an array , 将第k个对角线下的元素置零  下三角
    # [1,2,3]         [0,2,3]
    # [4,5,6]     ->  [0,0,6]
    # [7,8,9]         [0,0,0]
    np_mask = np.triu(np.ones((1, size, size)),k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    # [true, false, false]
    # [true, true,  false]
    # [true, true,  true]
    np_mask = np_mask.to(opt.device)
    # if opt.device == 0:
    #   np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg, opt):
    
    src_mask = (src != opt.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size, opt).to(opt.device)
        # if trg.is_cuda:
        #     np_mask.cuda()
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask

# patch on Torchtext's batching process that makes it more efficient
# from http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks

class MyIterator(data.Iterator):
    """
    MyIterator(dataset, batch_size, sort_key=None, device=None, 
    batch_size_fn=None, train=True, repeat=False, shuffle=None, 
    sort=None, sort_within_batch=None)
    """
    def create_batches(self):
        if self.train:
            # if it is training : shuffle data
            def pool(d, random_shuffler):
                # p是batch_size * 100大小的数据（src+trg）
                # data.batch(d, self.batch_size * 100)是一堆p
                for p in data.batch(d, self.batch_size * 100):
                    # p_batch是batch_size大小的数据（src+trg）
                    # p 是100个p_batch
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "对新加进来的一条example更新此batch的信息"
    "产生动态的batchsize的函数"
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
