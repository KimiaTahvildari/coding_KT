import torch
import torch.nn as nn   
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
#from custom_dataset import CustomDataset
import torch.nn as nn
'''model = nn.Sequential(
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
with torch.no_grad():'''




# modeling/model_pytorch.py



class AFCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc(self.net(x).squeeze(-1))
