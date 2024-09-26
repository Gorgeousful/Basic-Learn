#rnn简洁实现
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from s3 import load_data_time_machine
from s4 import train_ch8, predict_ch8

num_epochs, lr = 500, 1
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=False) #是否随机采样是在数据集上决定的
vocab_size = len(vocab)
num_hiddens = 256

class RNNModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens,  **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = nn.RNN(vocab_size, num_hiddens, num_layers=1)
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)

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


#测试
# state = torch.zeros((1, batch_size, num_hiddens)) #隐藏层数、批量大小、隐藏单元数
# X = torch.rand(size=(num_steps, batch_size, len(vocab)))
# Y, state_new = rnn_layer(X, state) #Y为(num_steps, batch_size, num_hiddens)  Y为每个时间步下的隐藏层状态h

device = d2l.try_gpu()
net = RNNModel(vocab_size, num_hiddens)
net = net.to(device)

print(predict_ch8('time traveller', 10, net, vocab, device))
train_ch8(net, train_iter, vocab, lr, num_epochs, device)




