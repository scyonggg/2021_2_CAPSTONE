import torch
import torch.functional as F
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import my_resnet

training_data = datasets.CIFAR10(root="./data", train=True, transform=ToTensor(), download=True)
test_data = datasets.CIFAR10(root="./data", train=False, transform=ToTensor(), download=False)

batch_size = 16

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [Batch, Ch, H, W] :", X.shape)
    print("Shape of y :", y.shape, y.dtype)
    break

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = my_resnet.ResNet50().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train(dataloader, model, criterion, optimizer):
    size = len(dataloader.dataset)
    pbar = tqdm(dataloader)
    for batch, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        pbar.set_description(f"Loss : {loss:>7f}")

        if batch % 1000 == 0:
            print('\n')

        # if batch % 625 == 0:
        #     loss, current = loss.item(), batch*len(X)
        #     print(f"\nLoss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 100

for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n---------------------")
    train(train_dataloader, model, criterion, optimizer)
    test(test_dataloader, model, criterion)
    scheduler.step()
    print("Current learning rate : ", scheduler.get_last_lr())

torch.save(model.state_dict(), "resnet50_SGD_cifar10.pth")
