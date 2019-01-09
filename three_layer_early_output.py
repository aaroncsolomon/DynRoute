import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
import argparse

from torchvision import datasets, transforms

#Based on the basic CNN implementation from https://github.com/pytorch/examples/blob/master/mnist/main.py

class ThreeLayerCNN(nn.Module):
    def __init__(self):
        super(ThreeLayerCNN, self).__init__()
        self.c1 = nn.Conv2d(1, 50, 3, 1)
        self.c2 = nn.Conv2d(20, 50, 3, 1)
        self.c3 = nn.Conv2d(50, 50, 3, 1)

        self.linear1 = nn.Linear(50, 500)
        self.linear2 = nn.Linear(500, 10)

        self.decide1 = nn.Linear(20*13*13, 3)



    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.max_pool2d(x, 2, 2)

        # compute an intermediary vector to decide
        # options: c2, c3, output

        iview = x.view(-1, 20*13*13)
        iview = self.decide1(iview)
        decisions1 = torch.max(iview, dim=1)[1] # gets which of the three paths  to take

        c2 = x[[(decisions1 == 0).nonzero()]]  # picking out ones to be convolved at c2
        c3 = x[[(decisions1 == 1).nonzero()]]  # picking out ones to be convolved at c3
        l1 = x[[(decisions1 == 2).nonzero()]]  # picking out ones to be run throuhg the linear layer

        #removing extra dim
        if (c2.size(0) > 0):
            c2 = c2.squeeze(dim=1)
        if (c3.size(0) > 0):
            c3 = c3.squeeze(dim=1)
        if (l1.size(0) > 0):
            l1 = l1.squeeze(dim=1)

        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2, 2)

        print(F.max_pool2d(c3, 3, 3, padding=1).size())

        # compute an intermediary vector to decide
        # options: c3, output

        x = F.relu(self.c3(x))

        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 50)


        x = F.relu(self.linear1(x))

        x = self.linear2(x)
        return F.log_softmax(x, dim=1)

def train(args, mod, dev, train_loader, optimizer, epoch):
    mod.train()
    losses = []
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(dev), target.to(dev)
        optimizer.zero_grad()
        output = mod(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append([epoch*len(train_loader.dataset) + idx, loss.item()])
        if idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_loader.dataset),
                100. * idx / len(train_loader), loss.item()))
    return losses

def test(args, mod, device, test_loader):
    mod.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target  = data.to(device), target.to(device)
            output = mod(data)
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
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
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

    for epoch in range(1, args.epochs+1):
        loss = train(args, model, device, train_loader, optimizer, epoch)
        losses.append(loss)
        accuracy, test_loss = test(args, model, device, test_loader)
        accuracies.append(accuracy)
        test_losses.append(test_loss)

    if (args.save_model):
        torch.save(model.state_dict(), 'basic_mnist_cnn.pt')

    with open('train_losses.pkl', 'wb') as handle:
        pickle.dump(losses, handle)
    with open('model_accuracy.pkl', 'wb') as handle:
        pickle.dump(accuracies, handle)
    with open('test_losses.pkl', 'wb') as handle:
        pickle.dump(test_losses, handle)

if __name__ == '__main__':
    main()
