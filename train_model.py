import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from matplotlib import pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import BatchNorm1d


print('imports done')
train_dataset = torch.load('saved_data/train_dataset.pt')
val_dataset = torch.load('saved_data/val_dataset.pt')
X_test_torch = torch.load('saved_data/X_test.pt')
y_test_torch = torch.load('saved_data/y_test.pt')

train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=512)

train_losses = []
val_losses = []

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(283, 512)
        self.bn1 = BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = BatchNorm1d(256)
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = BatchNorm1d(128)
        self.layer4 = nn.Linear(128, 64)
        self.bn4 = BatchNorm1d(64)
        self.layer5 = nn.Linear(64, 32)
        self.bn5 = BatchNorm1d(32)
        self.output_layer = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = torch.relu(self.bn2(self.layer2(x)))
        x = torch.relu(self.bn3(self.layer3(x)))
        x = torch.relu(self.bn4(self.layer4(x)))
        x = torch.relu(self.bn5(self.layer5(x)))
        x = self.output_layer(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = NeuralNetwork().to(device)

# Loss and optimizer
criterion = nn.MSELoss() 
optimizer = Adam(model.parameters(), lr=0.001)
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

num_epoch = 100

# Training loop with validation and accuracy
for epoch in range(num_epoch):
    # if device.type == 'cuda':
    #     torch.cuda.reset_peak_memory_stats(device)
    model.train()
    total_train_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimizations
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)


    # Validation phase
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    scheduler.step()

    if device.type == 'cpu':
        # max_memory = torch.cuda.max_memory_allocated(device)
        max_memory = 0 
        print(f"Epoch [{epoch + 1}/{num_epoch}], Max GPU Memory Used: {max_memory / (1024**2):.2f} MB, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        # print(f"Epoch [{epoch + 1}/50], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")=
        torch.save(model.state_dict(), "./saved_data/weights.pth")

model.eval()
with torch.no_grad():
    X_test_torch, y_test_torch = X_test_torch.to(device), y_test_torch.to(device)
    y_pred = model(X_test_torch)
    test_loss = criterion(y_pred, y_test_torch)
    print(f"Test Loss: {test_loss:.4f}")

    sample_indices = [0, 1, 2, 3, 4]  
    sample_inputs = X_test_torch[sample_indices]
    sample_true_values = y_test_torch[sample_indices]
    sample_predictions = model(sample_inputs)

    for i, (true, pred) in enumerate(zip(sample_true_values, sample_predictions)):
        print(f"Sample {i}: True Value = {true.tolist()}, Prediction = {pred.tolist()}")
print(f'train_loss={train_losses}')
print(f'val_loss={val_losses}')
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()