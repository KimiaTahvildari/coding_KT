import torch
import torch.nn as nn   
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import CustomDataset

model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1))

Class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1))

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        return x


#training
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # Reset the optimizer
        optimizer.zero_grad()
        # Make predictions
        outputs = model(inputs)
        # Calculate the loss
        loss = loss_fn(outputs, targets)
        # Backpropagate the loss
        loss.backward()
        # Update the model parameters
        optimizer.step()


# Make predictions
model.eval()
with torch.no_grad():




