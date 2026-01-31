import torch 
import torch.nn as nn
import torch.optim as optim

model = sequential(nn.Linear(10, 50),
                   nn.ReLU(),
                   nn.Linear(50, 1))


#transforms 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
#Datasets
dataset = CustomDataset(data_path='data.csv',train=True, transform=transform)
#DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#training loop 
#0.reset the optimizer
optimizer.zero_grad()
#1.make predictions
outputs = model(inputs)
#2.calculate the loss
loss = loss_fn(outputs, targets)
#3.backpropagate the loss
loss.backward()
#4.update the model parameters
optimizer.step()

#Inference
with torch.no_grad():