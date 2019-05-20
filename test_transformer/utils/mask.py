import torch

#mask顾名思义就是掩码，在我们这里的意思大概就是对某些值进行掩盖，使其不产生效果。
#padding mask在所有的scaled dot-product attention里面都需要用到，而sequence mask只有在decoder的self-attention里面用到。
def padding_mask(seq_k, seq_q):#我们要对输入序列进行对齐！具体来说，就是给在较短的序列后面填充0
    len_q = seq_q.size(1)

    pad_mask = seq_k.eq(0)
    # TODO
    pad_mask = pad_mask.unsqueeze(0).expend(-1, len_q, -1)
    return pad_mask


"""sequence mask是为了使得decoder不能看见未来的信息。也就是对于一个序列，在time_step为t的时刻
我们的解码输出应该只能依赖于t时刻之前的输出，而不能依赖t之后的输出。因此我们需要想一个办法，把t之后的信息给隐藏起来。

那么具体怎么做呢？也很简单：产生一个上三角矩阵，上三角的值全为1，下三角的值权威0，对角线也是0。
把这个矩阵作用在每一个序列上，就可以达到我们的目的啦。
"""
def sequence_mask(seq):
    batch_szie, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expend(batch_szie, -1, -1)
    return mask

"""值得注意的是，本来mask只需要二维的矩阵即可，但是考虑到我们的输入序列都是批量的，
所以我们要把原本二维的矩阵扩张成3维的张量。上面的代码可以看出，我们已经进行了处理。"""
