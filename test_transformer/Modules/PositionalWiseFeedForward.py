import torch.nn as nn
import torch.nn.functional as F


class PositionalWiseFeedForward(nn.Module): #"这就是一个全连接网络，包含两个线性变换和一个非线性函数（实际上就是ReLU）。公式如下"
    def __init__(self, mode_dims, ffn_dim, dropout_rate):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(mode_dims, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, mode_dims, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(mode_dims)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w1(output)
        output = F.relu(output)
        output = self.w2(output)
        output = self.dropout(output.transpose(1, 2))
        output = self.layer_norm(x + output)
        return output
