# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from s3 import load_data_time_machine
import math
print('\n\n\n')





#RNN
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size #因为要经过one-hot独热编码
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    '''初始化初始权值，返回一个元组'''
    return (torch.zeros((batch_size, num_hiddens), device=device),)

def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,) #row：batchsizexnum_step, col:vocab

class RNNModelScratch():
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state): #内部会进行one-hot编码
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)   #num_step x batchsize x vocabsize
        return self.forward_fn(X, state, self.params) #y, state

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

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



# num_epochs, lr = 500, 1
# batch_size, num_steps = 32, 35
# num_hiddens = 512
# train_iter, vocab = load_data_time_machine(batch_size, num_steps)
# net = RNNModelScratch(len(vocab), num_hiddens, 'cuda', get_params, init_rnn_state, rnn)
# train_ch8(net, train_iter, vocab, lr, num_epochs, 'cuda')
# print()
# net = RNNModelScratch(len(vocab), num_hiddens, 'cuda', get_params, init_rnn_state, rnn)
# train_ch8(net, train_iter, vocab, lr, num_epochs, 'cuda',use_random_iter=True)

