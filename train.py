import os

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from model import ResNet50
from datasets import BinaryDatasets, MultiwayDatasets, MultiwaySubDatasets, SampledTestSets
from engine import train_one_epoch, test

def train(positive_class, negative_classes, num_epochs=200, resume=False):
    writer = SummaryWriter()

    # print(f"Training {positive_class} classifier...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    directory = f"checkpoints/4way/{positive_class}"
    # directory = "checkpoints/4way"
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Data
    print('==> Preparing data...')
    trainloader, testloader = BinaryDatasets([positive_class], negative_classes)

    # Create model, criterion and optimizer
    print(f'==> Building model...')
    net = ResNet50(2)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-3) # usual 5e-4

    # load from checkpoint
    if resume:
        print('==> Resuming from checkpoint...')
        checkpoint = torch.load(f'{directory}/last_ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optim'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = torch.load(f'{directory}/best_ckpt.pth')['acc']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45, 60, 70, 75, 90, 110],
                                                         gamma=0.1, verbose=True,
                                                         last_epoch=checkpoint['epoch'])

    # create history file
    else:
        best_acc = 0
        start_epoch = 0
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 125, 150],
                                                         gamma=0.1, verbose=True)
        with open(f"{directory}/history.txt", "w") as f:
            f.write("epoch,train_loss,train_acc,test_loss,test_acc\n")

    # loop through number of epochs
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_one_epoch(net, optimizer, criterion, 
                                                trainloader, epoch, device)
        test_loss, test_acc = test(net, criterion, testloader, device)

        # save losses and accuracies into txt file
        with open(f"{directory}/history.txt", "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{test_loss:.4f},{test_acc:.4f}\n")

        # save checkpoint
        state = {
            'net': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
            'optim': optimizer.state_dict()
        }
        torch.save(state, f'{directory}/last_ckpt.pth')
        if test_acc > best_acc:
            torch.save(state, f'{directory}/best_ckpt.pth')
            best_acc = test_acc

        scheduler.step()

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)

    writer.close()


if __name__ == '__main__':
    train('airplane', ['automobile', 'ship', 'truck'], 80)
    # train('automobile', ['airplane', 'ship', 'truck'], 80)
    # train('ship', ['automobile', 'airplane', 'truck'], 80)
    # train('truck', ['automobile', 'ship', 'airplane'], 80)