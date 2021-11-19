import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from tqdm import tqdm

# dataset
# dataloader
# hyperparameter
# NeuralNet 정의
# Model 선언
# Loss function, optimizer
# training 및 test 코드



    # 모든 TorchVision Dataset은 샘플과 정답을 각각 변경하기 위한 transform과 target_transform 두 인자를 포함한다.

# Training dataset download
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
# Test dataset download
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

    # Dataset을 DataLoader의 인자로 전달한다.
    # Dataset을 iterable로 감싸고, batch, sampling, shuffle 및 multiprocessing data loading 을 지원.

batch_size = 64

# Create DataLoader
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N(batch), C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Making Model
'''
    - 모델은 nn.Module을 상속받는 class를 생성하여 정의함.
    - __init__ 함수에서 layer들을 정의
    - forward 함수에서 데이터를 어떻게 전달할지 지정함.
'''

device = "cuda" if torch.cuda.is_available() else 'cpu'
print("using {} device".format(device))


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss : {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in tqdm(range(epochs)):
    print(f"Epoch {t+1}\n---------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted : "{predicted}", Actual : "{actual}"')