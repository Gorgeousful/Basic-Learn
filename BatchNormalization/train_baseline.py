#lossfunction的损失函数与optimizer优化器的使用及深度学习框架
#GPU上训练 数据（输入，标注）， 网络模型， 损失函数
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
from func import *
import numpy as np
#out = (in-ksize+2*pading)/stride + 1
#NCHW
writer = SummaryWriter('logs')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

def ModelTest(model, test_loader, loss_fcn):
    model.eval() #设置为评估模式
    loss_epoch, acc = 0, 0
    with torch.no_grad(): #关闭梯度计算，节省内存
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), colour='GREEN', desc='Test    ')
        for batch_idx, (data, target) in pbar:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = loss_fcn(output, target)
            loss_epoch += loss.item()/len(test_loader)
            # 计算预测正确的数量并累加到总数中 argmax(axis=1)沿着行看的最大值对应的索引
            acc += ((output.argmax(axis=1) == target).type(torch.float32).sum().item())/len(test_loader.dataset)
            pbar.set_postfix(loss=loss_epoch, acc=acc)
        pbar.close()
    return loss_epoch, acc


model = LeNetBaseLine().cuda()
print(model)
summary(model, (1, 28, 28))

loss_fcn = nn.CrossEntropyLoss().cuda() #对于CrossEntropyLoss input:(N,C) Target(N)   交叉熵计算公式为 sum(-y*np.log(f(xj)))  但这里f(x)经过softmax处理过 即为sum(-y*np.log(exp(f(x))/sum(exp(f(xj))))
# optimizer = optim.SGD(model.parameters(), lr=1e-2) #随机梯度下降
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr) #最好用

epoch_num = 10
acc_before = 0
loss_trains = []
loss_tests = []
accs = []
for epoch in range(epoch_num):
    model.train()
    loss_train = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch-{epoch:02d}')
    for batch_idx, (data, target) in pbar:
        optimizer.zero_grad() #梯度清零 必不可少
        data, target = data.cuda(), target.cuda()
        output = model(data) #64,10
        loss = loss_fcn(output, target) #一个批次样本的平均损失
        loss.backward()  # 反向传播，更新model中各参数的梯度grad
        optimizer.step() #优化器根据梯度优化参数
        loss_train += loss.item() / len(train_loader)
        pbar.set_postfix(loss=loss_train)
    loss_test, acc = ModelTest(model,test_loader, loss_fcn)
    loss_trains.append(loss_train)
    loss_tests.append(loss_test)
    accs.append(acc)

    writer.add_scalars('Metrics',
                       {'TraLosB': loss_train, 'TesLosB': loss_test, 'TesAccB': acc},
                       epoch)
    pbar.close()
    best_index = np.argmax(np.array(accs))
    if acc >= accs[best_index]:
        torch.save(model.state_dict(), './model/LeNetBaseline.pth')

print(f"Epoch={best_index:02d}, Acc={accs[best_index]:.4f}, LossTra={loss_trains[best_index]:.4f}, LossTes={loss_tests[best_index]:.4f}.")
writer.close()
