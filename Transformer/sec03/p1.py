import math
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 1 掩蔽softmax  实际运行看效果即可
print(d2l.masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
print(d2l.masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

# 2 加性注意力 query和key的特征数量可以不一样
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        #key_size和query_size指的是特征大小而不是key或query的数目
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = d2l.masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)
# 以queries为例,(batch_size, queries_num, queries_feature_num)
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2)) # (2,1,20) (2,10,2)
# values的小批量，两个值矩阵是相同的
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1) # (2,10,4)
valid_lens = torch.tensor([2, 6])
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
result = attention(queries, keys, values, valid_lens)  #权值矩阵为(batch_size, query_num, key_num)
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')
plt.show()

# 3 缩放点积注意力 query和key的特征数量要求一致  没有参数用于学习
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        # 假设输入每个元素均为0均值和1方差
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = d2l.masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
queries = torch.normal(0, 1, (2, 1, 2))  #key为(2,10,2)
attention = DotProductAttention(dropout=0.5)
attention.eval()
result1 = attention(queries, keys, values, valid_lens)
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')
plt.show()
