import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os
import numpy as np
import argparse
from Model.Resnet import ResNet18
from Optimizer.AdaVAM import AdaVAM


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fun(worker_id, seed):
    np.random.seed(seed + worker_id)

def worker_init_fn(worker_id):
    """ DataLoader """
    seed = 1  # keep main seed consist
    worker_init_fun(worker_id, seed)

def build_dataset():
    """build dataset for CIFAR100"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data_CIFAR100', train=True, download=True,
                                             transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data_CIFAR100', train=False, download=True,
                                            transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    return train_loader, test_loader

def select_model(model_name, num_classes=100):
    if model_name == 'ResNet18':
        return ResNet18(num_classes)
    else:
        raise ValueError(f"false: {model_name}")

def select_optimizer(optimizer_name, net):
    if optimizer_name == 'SGD':
        return torch.optim.SGD(net.parameters(), lr=1e-1, momentum=0, dampening=0, weight_decay=5e-4)
    elif optimizer_name == 'Adam':
        return torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    elif optimizer_name == 'AdamW':
        return torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    elif optimizer_name == 'AdaVAM':
        return AdaVAM(net.parameters(), lr=1e-3, weight_decay=5e-4)
    else:
        raise ValueError(f"false: {optimizer_name}")

def train(net, device, data_loader, optimizer, criterion):
    net.train()
    train_loss, correct, total = 0, 0, 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f'Train accuracy: {accuracy:.3f}%')
    return train_loss / len(data_loader), accuracy

def test(net, device, data_loader, criterion):
    net.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / total
    avg_test_loss = test_loss / len(data_loader)
    print(f'Test accuracy: {accuracy:.3f}')
    return accuracy, avg_test_loss

def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-100 with various models and optimizers")
    parser.add_argument('--model', default='ResNet18', type=str, help='Model: ResNet18')
    parser.add_argument('--optimizer', default='AdaVAM', type=str, help='optimizer: SGD, Adam, AdamW, AdaVAM')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    args = parser.parse_args()

    set_seed(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = build_dataset()
    net = select_model(args.model, num_classes=100).to(device)
    optimizer = select_optimizer(args.optimizer, net)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    for epoch in range(args.epochs):
        print(f"------Epoch {epoch}------")
        start_time = time.time()
        train(net, device, train_loader, optimizer, criterion)
        test_accuracy, _ = test(net, device, test_loader, criterion)
        scheduler.step()
        elapsed_time = (time.time() - start_time) / 60
        print(f"Epoch {epoch} finished in {elapsed_time:.2f} minutes, Test Acc: {test_accuracy:.2f}%")


if __name__ == '__main__':
    main()
