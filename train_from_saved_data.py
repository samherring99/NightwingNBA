import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from matplotlib import pyplot as plt
import numpy as np

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
        self.layer1 = nn.Linear(283, 512)  # Assuming all samples have the same number of features
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = self.output_layer(x)
        return x

model = NeuralNetwork()

# Loss and optimizer
criterion = nn.MSELoss() 
optimizer = Adam(model.parameters(), lr=0.00075)

# Training loop with validation and accuracy
for epoch in range(100):
    model.train()
    total_train_loss = 0

    for X_batch, y_batch in train_loader:
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
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)


    print(f"Epoch [{epoch + 1}/50], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

torch.save(model.state_dict(), "./saved_data/weights.pth")

model.eval()
with torch.no_grad():
    y_pred = model(X_test_torch)
    test_loss = criterion(y_pred, y_test_torch)
    print(f"Test Loss: {test_loss:.4f}")

    sample_indices = [0, 1, 2, 3, 4]  
    sample_inputs = X_test_torch[sample_indices]
    sample_true_values = y_test_torch[sample_indices]
    sample_predictions = model(sample_inputs)

    for i, (true, pred) in enumerate(zip(sample_true_values, sample_predictions)):
        print(f"Sample {i}: True Value = {true.tolist()}, Prediction = {pred.tolist()}")

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()