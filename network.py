import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt

import time
import random

class F_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(tuple_length, w),
            nn.LeakyReLU(),
        )
        self.layers = []

        for _ in range(d):
            self.layers.append(nn.Linear(w,w))
            self.layers.append(nn.LeakyReLU())

        self.hidden = nn.Sequential(*self.layers)

        self.decoder = nn.Sequential(
            nn.Linear(w, tuple_length),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.hidden(x)
        x = self.decoder(x)
        return x

def generate_data(len1, len2):
    data = []
    for _ in range(len2):
        list = [random.uniform(-1,1) for _ in range(len1)]
        data.append(list)
    dataset = torch.tensor(data)
    return dataset


#TODO: prepare and load dataset as soon as I have it
batch_size = 256

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = generate_data(200, 100)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)#creates iterable of dataset, shunks it in batch_sizes and shuffles the data




w = 200 #width dimension of neural net
d = 5 #depth of hidden layer
tuple_length = 3 + len(z)

model = F_1.to(device)
criterion = nn.MSELoss()


learning_rate = 1e-3
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

num_epochs = 50
do = nn.Dropout()  # comment out for standard AE
print("Starting training...")
torch.cuda.empty_cache()
total = 0
for epoch in range(num_epochs):
    start = time.clock()
    for data in dataloader:
        data= data.to(device)
        data.requires_grad_()

        # ===================forward=====================
        output = model(data)
        loss = criterion(output, data.data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    end = time.clock() - start
    total += end
    print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.item():.4f}, wall time:{end:.4f}, total time:{total:.4f}')