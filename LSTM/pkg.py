import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import collections
import matplotlib.pyplot as plt
import random
import math
import re

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    # with open(d2l.download('time_machine'), 'r') as f:
    #     lines = f.readlines()
    with open(r'data/timemachine.txt', 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines] #除A-Z和a-z字母外的字符都替换成空格
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

class Vocab:  #@save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk) #找不到就返回unk的索引0
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab #原始语料集(转为索引，且平展) 索引对应表

def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y) #调用函数返回值，不间断函数内部的执行过程

def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens=-1):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

def grad_clipping(net, theta):
    '''梯度剪裁（幅值）'''
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta/norm

def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1,1))
    for y in prefix[1:]: #用已知的信息做预测
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2) # 训练损失之和, 词元数量
    for X,Y in train_iter: #使用顺序采样就是整个epoch是一长串序列，如果使用的随机采样，则整个每个batch是一串序列，但是之间不是
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device='cuda')
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_() #将state从梯度图中分离出来，避免梯度计算  state是依靠net网络自己的函数更新的
            else:
                for s in state:
                    s.detach_() #detach_是直接修改原张量，而detach是创建一个新的与计算图分离的张量
        y = Y.T.reshape(-1) #将num_step放在前面
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean() #loss特有的广播机制
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0]/metric[1]), metric[1] / timer.stop() #困惑度与运行时间

#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    # plt.ion()
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
            plt.draw()
            plt.pause(0.1)
            # plt.show()
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    # plt.ioff()

class RNNModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_layers=1, bidirectional=False, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = nn.RNN(vocab_size, num_hiddens, num_layers=num_layers, bidirectional=bidirectional)
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens*2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1,Y.shape[-1]))) #num_steps*batch_size, num_hiddens
        return output, state

    def begin_state(self, device, batch_size=1):
        #nn.RNN以张量为隐藏状态
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions*self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        # nn.LSTM以元组为隐藏状态
        else:
            return(torch.zeros((self.num_directions*self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
                   torch.zeros((self.num_directions*self.rnn.num_layers, batch_size, self.num_hiddens), device=device))

class GRUModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_layers=1, bidirectional=False, **kwargs):
        super(GRUModel, self).__init__(**kwargs)
        self.rnn = nn.GRU(vocab_size, num_hiddens, num_layers=num_layers, bidirectional=bidirectional)  #GRU层的pytorch定义方式本质上与RNN一致 这里rnn对只是一个层
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens*2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1,Y.shape[-1]))) #num_steps*batch_size, num_hiddens
        return output, state

    def begin_state(self, device, batch_size=1):
        #nn.RNN以张量为隐藏状态
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions*self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        # nn.LSTM以元组为隐藏状态
        else:
            return(torch.zeros((self.num_directions*self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
                   torch.zeros((self.num_directions*self.rnn.num_layers, batch_size, self.num_hiddens), device=device))

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens,  num_layers=1, bidirectional=False, **kwargs):
        super(LSTMModel, self).__init__(**kwargs)
        self.rnn = nn.LSTM(vocab_size, num_hiddens, num_layers=num_layers, bidirectional=bidirectional)  #LSTM层的pytorch定义方式本质上与RNN一致 这里rnn对只是一个层
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens*2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1,Y.shape[-1]))) #num_steps*batch_size, num_hiddens
        return output, state

    def begin_state(self, device, batch_size=1):
        #nn.RNN以张量为隐藏状态
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions*self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        # nn.LSTM以元组为隐藏状态
        else:
            return(torch.zeros((self.num_directions*self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
                   torch.zeros((self.num_directions*self.rnn.num_layers, batch_size, self.num_hiddens), device=device))