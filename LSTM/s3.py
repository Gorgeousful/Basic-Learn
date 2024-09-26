#用双向的做填词还可以，但是做预测时，虽然困惑度低，但是效果不好。
from pkg import *

num_epochs, lr = 500, 2
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=False) #是否随机采样是在数据集上决定的
vocab_size = len(vocab)
num_hiddens = 256
num_layers = 2
device = 'cuda'

# model = LSTMModel(vocab_size, num_hiddens, num_layers)
# model = model.to(device)
# train_ch8(model, train_iter, vocab, lr, num_epochs, device)

model = RNNModel(vocab_size, num_hiddens, num_layers, bidirectional=True)
model = model.to(device)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)