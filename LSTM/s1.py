from pkg import *

num_epochs, lr = 500, 1
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=False) #是否随机采样是在数据集上决定的
vocab_size = len(vocab)
num_hiddens = 256
device = 'cuda'

model = LSTMModel(vocab_size, num_hiddens)
model = model.to(device)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)