import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
import argparse
import os
import psutil
import time

from torchvision import datasets, transforms

process = psutil.Process(os.getpid())

#Based on the basic CNN implementation from https://github.com/pytorch/examples/blob/master/mnist/main.py

class ThreeLayerCNN(nn.Module):
    def __init__(self):
        super(ThreeLayerCNN, self).__init__()
        self.c1 = nn.Conv2d(1, 50, 3, 1)
        self.c2 = nn.Conv2d(50, 50, 3, 1)
        self.c3 = nn.Conv2d(50, 50, 3, 1)

        self.linear1 = nn.Linear(50, 500)
        self.linear2 = nn.Linear(500, 10)

        self.decide1 = nn.Linear(50 * 13 * 13, 3)
        self.decide2 = nn.Linear(50 * 5 * 5, 2)

        self.c1_c3 = nn.Linear(13,13)
        self.c1_l1 = nn.Linear(1,500)
        self.c2_l1 = nn.Linear(5,500)

    def forward(self, x):

        l1_start=time.perf_counter()

        x = F.relu(self.c1(x))
        x = F.max_pool2d(x, 2, 2)

        l1_end = time.perf_counter()

        # compute an intermediary vector to decide
        # options: c2, c3, output

        dl1_start = time.perf_counter()

        iview = x.view(-1, 50*13*13)
        iview = self.decide1(iview)
        decisions1 = torch.max(iview, dim=1)[1] # gets which of the three paths  to take

        c2_idx = (decisions1 == 0).nonzero().numpy().flatten().tolist()
        c3_idx = (decisions1 == 1).nonzero().numpy().flatten().tolist()
        l11_idx = (decisions1 == 2).nonzero().numpy().flatten().tolist()

        c2 = x[c2_idx]  # picking out ones to be convolved at c2 - must keep track so we can permute labels later
        c3 = x[c3_idx]  # picking out ones to be convolved at c3
        l11 = x[l11_idx]  # picking out ones to be run through the linear layer

        #removing extra dim
        if (c2.size(0) > 0):
            c2 = c2.squeeze(dim=1)
        if (c3.size(0) > 0):
            c3 = c3.squeeze(dim=1)
        if (l11.size(0) > 0):
            l11 = l11.squeeze(dim=1)
            l11 = F.max_pool2d(l11, 7, 7)

        dl1_end = time.perf_counter()

        c33 = self.c1_c3(c3)
        l11 = self.c1_l1(l11)

        l2_start = time.perf_counter()
        c2 = F.relu(self.c2(c2))
        c2 = F.max_pool2d(c2, 2, 2)

        l2_end = time.perf_counter()

        dl2_start = time.perf_counter()

        #split for c3 and l1
        iview = c2.view(-1, 50*5*5)
        iview = self.decide2(iview)
        decisions2 = torch.max(iview, dim=1)[1]

        c32_idx = (decisions2 == 0).nonzero().numpy().flatten().tolist()
        l12_idx = (decisions2 == 1).nonzero().numpy().flatten().tolist()

        real_c32_idx = [c2_idx[i] for i in c32_idx] #c2_idx[c32_idx] #where the real indices have been shuffled to
        real_l12_idx = [c2_idx[i] for i in l12_idx] #c2_idx[l12_idx]

        c32 = c2[c32_idx]  # picking out ones to be convolved at c2
        l12 = c2[l12_idx]

        if (c32.size(0) > 0):
            c32 = c32.squeeze(dim=1)
        if (l12.size(0) > 0):
            l12 = l12.squeeze(dim=1)
            l12 = self.c2_l1(l12)
            l12 = F.max_pool2d(l12, 3, 3)

        dl2_end = time.perf_counter()

        ## combine the c32 with the pooling currently in the print statement below

        l3_start = time.perf_counter()

        c3_input = torch.cat((c32,c33))

        real_c3_idx = real_c32_idx + c3_idx

        # compute an intermediary vector to decide
        # options: c3, output

        c3_output = F.relu(self.c3(c3_input))

        c3_output = F.max_pool2d(c3_output, 2, 2)

        order = l11_idx + real_l12_idx + real_c3_idx

        holder = [l11, l12, c3_output]
        holder = [x for x in holder if x.nelement() != 0]
        c3_output = torch.cat(holder)

        c3_output = c3_output.view(-1, 50)



        ## merge l11 and l12 with the output of c3 to go into the linear layer

        x = F.relu(self.linear1(c3_output))

        x = self.linear2(x)

        l3_end = time.perf_counter()
        return (F.log_softmax(x, dim=1), order,
                (l11_idx, c3_idx, real_l12_idx),
                (l1_end-l1_start, dl1_end-dl1_start, l2_end-l2_start, dl2_end-dl2_start, l3_end-l3_start))

def train(args, mod, dev, train_loader, optimizer, epoch):
    mod.train()
    losses = []
    l11_samples = []
    c3_samples = []
    l12_samples = []

    l1_times = []
    l2_times = []
    d1_times = []
    d2_times = []
    ot_times = []

    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(dev), target.to(dev)
        optimizer.zero_grad()
        output, order, swapped, tims= mod(data)
        loss = F.nll_loss(output, target[order])
        a,b,c = swapped
        l1t, d1t, l2t, d2t, ot = tims
        l11_samples = np.concatenate((l11_samples, target[a]), axis=None)
        c3_samples = np.concatenate((c3_samples, target[b]), axis=None)
        l12_samples = np.concatenate((l12_samples, target[c]), axis=None)

        l1_times = np.concatenate((l1_times,l1t), axis=None)
        l2_times = np.concatenate((l2_times,l2t), axis=None)
        d1_times = np.concatenate((d1_times,d1t), axis=None)
        d2_times = np.concatenate((d2_times,d2t), axis=None)
        ot_times = np.concatenate((ot_times,ot), axis=None)

        loss.backward()
        optimizer.step()
        losses.append([epoch*len(train_loader.dataset) + idx, loss.item()])
        if idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_loader.dataset),
                100. * idx / len(train_loader), loss.item()))
    return (losses, (l11_samples, c3_samples, l12_samples), (l1_times, l2_times, d1_times, d2_times, ot_times))

def test(args, mod, device, test_loader):
    mod.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target  = data.to(device), target.to(device)
            output, order, swapped, tims = mod(data)
            target = target[order]
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{}  ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)
    ))
    return correct/len(test_loader.dataset), test_loss

def main():
    parser = argparse.ArgumentParser(description='Basic CNN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='batch size for testing')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='default learning rate 0.01')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='saving curr model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers':1, 'pin_memory':True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(datasets.MNIST(
        '../data', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
    ), batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(
        '../data', train=False,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
    ), batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = ThreeLayerCNN().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    losses = []
    accuracies = []
    test_losses = []
    l11_samples = []
    c3_samples = []
    l12_samples = []

    big_l1_times = []
    big_l2_times = []
    big_d1_times = []
    big_d2_times = []
    big_ot_times = []

    for epoch in range(1, args.epochs+1):
        (loss, (l11_res, c3_res, l12_res), (l1_times, l2_times, d1_times, d2_times, ot_times)) = train(args, model, device, train_loader, optimizer, epoch)
        losses.append(loss)
        accuracy, test_loss = test(args, model, device, test_loader)
        accuracies.append(accuracy)
        test_losses.append(test_loss)



        if len(big_l1_times) == 0:
            big_l1_times = np.stack((np.repeat(epoch, len(l1_times)), l1_times))
        else:
            big_l1_times = np.concatenate((big_l1_times, np.stack((np.repeat(epoch, len(l1_times)), l1_times))), axis=1)

        if len(big_l2_times) == 0:
            big_l2_times = np.stack((np.repeat(epoch, len(l2_times)), l2_times))
        else:
            big_l2_times = np.concatenate((big_l2_times, np.stack((np.repeat(epoch, len(l2_times)), l2_times))), axis=1)

        if len(big_d1_times) == 0:
            big_d1_times = np.stack((np.repeat(epoch, len(d1_times)), d1_times))
        else:
            big_d1_times = np.concatenate((big_d1_times, np.stack((np.repeat(epoch, len(d1_times)), d1_times))), axis=1)

        if len(big_d2_times) == 0:
            big_d2_times = np.stack((np.repeat(epoch, len(d2_times)), d2_times))
        else:
            big_d2_times = np.concatenate((big_d2_times, np.stack((np.repeat(epoch, len(d2_times)), d2_times))), axis=1)

        if len(big_ot_times) == 0:
            big_ot_times = np.stack((np.repeat(epoch, len(ot_times)), ot_times))
        else:
            big_ot_times = np.concatenate((big_ot_times, np.stack((np.repeat(epoch, len(ot_times)), ot_times))), axis=1)




        if len(l11_samples) == 0:
            l11_samples = np.stack((np.repeat(epoch, len(l11_res)), l11_res))
        else:
            l11_samples = np.concatenate((l11_samples, np.stack((np.repeat(epoch, len(l11_res)), l11_res))), axis=1)

        if len(c3_samples) == 0:
            c3_samples = np.stack((np.repeat(epoch, len(c3_res)), c3_res))
        else:
            c3_samples = np.concatenate((c3_samples, np.stack((np.repeat(epoch, len(c3_res)), c3_res))), axis=1)

        if len(l12_samples) == 0:
            l12_samples = np.stack((np.repeat(epoch, len(l12_res)), l12_res))
        else:
            l12_samples = np.concatenate((l12_samples, np.stack((np.repeat(epoch, len(l12_res)), l12_res))), axis=1)

    np.savetxt('time_interlayer_l1.csv', big_l1_times, delimiter=',')
    np.savetxt('time_interlayer_l2.csv', big_l2_times, delimiter=',')
    np.savetxt('time_interlayer_d1.csv', big_d1_times, delimiter=',')
    np.savetxt('time_interlayer_d2.csv', big_d2_times, delimiter=',')
    np.savetxt('time_interlayer_ot.csv', big_ot_times, delimiter=',')

    np.savetxt('interlayer_l11.csv', l11_samples, delimiter=',')
    np.savetxt('interlayer_c3.csv', c3_samples, delimiter=',')
    np.savetxt('interlayer_l12.csv', l12_samples, delimiter=',')

    if (args.save_model):
        torch.save(model.state_dict(), 'interlayer_mnist_cnn.pt')

    with open('train_losses.pkl', 'wb') as handle:
        pickle.dump(losses, handle)
    with open('model_accuracy.pkl', 'wb') as handle:
        pickle.dump(accuracies, handle)
    with open('test_losses.pkl', 'wb') as handle:
        pickle.dump(test_losses, handle)

    print(process.memory_info().rss)




if __name__ == '__main__':
    main()

