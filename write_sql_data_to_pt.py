from utils.misc_utils import get_all_teams
from utils.processing_utils import get_opponent_id, get_previous_game_id, get_team_previous_game_stats, get_team_for_player, get_player_data, get_team_roster, get_games_for_team

import sqlite3
import requests
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence


conn = sqlite3.connect('nba_datastore.db',
                             detect_types=sqlite3.PARSE_DECLTYPES |
                             sqlite3.PARSE_COLNAMES)
cursor = conn.cursor()

def get_context_for_game(game_id, player_id):

    player_team_id = get_team_for_player(player_id)

    player_previous_game_id = get_previous_game_id(game_id, player_team_id)

    player_previous_game_stats = get_player_data(player_previous_game_id, player_id)

    if player_previous_game_stats:

        team_previous_game_id = get_previous_game_id(game_id, team_previous_game_id)

        team_previous_game_stats = get_team_previous_game_stats(team_previous_game_id, player_team_id)

        opposing_team = get_opponent_id(game_id, player_team_id)

        opposing_team_previous_game_id = get_previous_game_id(game_id, opposing_team)

        opponent_previous_game_stats = get_team_previous_game_stats(opposing_team_previous_game_id, opposing_team)

        return [player_previous_game_stats, team_previous_game_stats[2:], opponent_previous_game_stats[2:]]

    else:
        return []

def validation(game_id, player_id):

    cursor.execute(player_template.format(game_id=game_id, player_id=player_id))
    player_game_stats = cursor.fetchall()

    if player_game_stats:

        hit_points = player_game_stats[0][53]
        hit_assists = player_game_stats[0][43]
        hit_rebounds = player_game_stats[0][15]

        return [hit_points, hit_assists, hit_rebounds]

def plot_data(array):
    x_vals = []
    y_vals = []

    i = 1
    for feature in array:
        x_vals.append(i)
        y_vals.append(feature)

        i += 1

    plt.plot(x_vals, y_vals)
    plt.show()


# Top level method
def generate_data(team_list, cursor):
    X = [] 
    y = []
    for team in team_list:
        games_list = get_games_for_team(team)

        print("TEAM ID: " + str(team))

        for game_list in games_list:
            for i in range(len(game_list)):
                game_id_num = str(game_list[i]['$ref']).split("?")[0].split("/")[-1]

                players = get_team_roster(team)

                for player_id in players:                 
                    data = get_context_for_game(game_id_num, player_id[0])
                    if data:
                        scores = validation(game_id_num, player_id[0])

                        training_example = [] 

                        for piece in data:
                            for number in piece:
                                training_example.append(number) # flatten np.flatten()

                        if training_example and scores:
                            X.append(training_example)      # X:  N x 283
                            y.append(scores)                # y:  N x 3 
    return X, y 


# TODO Clean up and organize the below code into a method utils file

team_list = get_all_teams()
X, y = generate_data(team_list, cursor)
#print(X)

# print(len(X))
# print(len(y))
# print(len(X[0]))
# print(len(y[0]))

df = pd.DataFrame(X)

print(df.shape)

# Normalize
#features = X 
#feature_means = np.mean(features, axis=1)[..., None]
#feature_stds = np.std(features, axis=1)[..., None]
#standardized_features = (features - feature_means) / feature_stds

# feature_means = np.mean(features, axis=0) ## I think this one is right
# feature_stds = np.std(features, axis=0)
# standardized_features = (features - feature_means) / feature_stds

#print(standardized_features)

#for example in standardized_features:
    # target is y[i]

#plot_data(standardized_features[0])


### Linear/Logistic Regression ###


# model = OneVsRestClassifier(LogisticRegression()).fit(X_train, y_train)

# lr = LogisticRegression().fit(X_train, y_train) # 'ovr' (one-vs-rest) strategy, fits a separate classifier for each class.

# y_pred_train = lr.predict(X_train)
# y_pred_test = lr.predict(X_test)

# train_accuracy = accuracy_score(y_train, y_pred_train)
# test_accuracy = accuracy_score(y_test, y_pred_test)

# print(f'Training Accuracy: {train_accuracy:.2f}')
# print(f'Test Accuracy: {test_accuracy:.2f}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

#print(len(X_train))
#print(len(X_train[0]))

### Random Forest ###

# rfR = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=0)
# rfR.fit(X_train, y_train)

# rfR_train_pred = rfR.predict(X_train)
# rfR_test_pred = rfR.predict(X_test)

# # Evaluation Metrics
# print("MSE RF - Train:", round(mean_squared_error(y_train, rfR_train_pred), 2))
# print("MSE RF - Test:", round(mean_squared_error(y_test, rfR_test_pred), 2))
# print("-" * 20)
# print("MAE RF - Train:", round(mean_absolute_error(y_train, rfR_train_pred), 2))
# print("MAE RF - Test:", round(mean_absolute_error(y_test, rfR_test_pred), 2))

# # Sample Predictions
# samples = X_test[:5]
# true_values = y_test[:5]
# predictions = rfR.predict(samples)

# for i, (true, pred) in enumerate(zip(true_values, predictions)):
#     print(f"Sample {i}: True Value = {true}, Prediction = {pred}")
    
    
### Neural network ###

# Define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(283, 128)  # Assuming all samples have the same number of features
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output_layer(x)
        return x

# Create the model
model = NeuralNetwork()

# Loss and optimizer
criterion = nn.MSELoss() 
optimizer = Adam(model.parameters())

X_train_padded = pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in X_train], batch_first=True)
y_train_padded = pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in y_train], batch_first=True)
X_test_padded = pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in X_test], batch_first=True)
y_test_padded = pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in y_test], batch_first=True)


 #Convert data to PyTorch tensors
X_train_torch = torch.tensor(X_train_padded, dtype=torch.float32)
y_train_torch = torch.tensor(y_train_padded, dtype=torch.float32)
X_test_torch = torch.tensor(X_test_padded, dtype=torch.float32)
y_test_torch = torch.tensor(y_test_padded, dtype=torch.float32)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_torch, y_train_torch, test_size=0.2, random_state=0)

train_dataset = TensorDataset(X_train_split, y_train_split)

val_dataset = TensorDataset(X_val_split, y_val_split)

# Save the datasets - THIS IS ALL WE REALLY CARE ABOUT HERE
torch.save(train_dataset, 'train_dataset.pt')
torch.save(val_dataset, 'val_dataset.pt')
torch.save(X_test_torch, 'X_test.pt')
torch.save(y_test_torch, 'y_test.pt')

'''

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16)

train_losses = []
val_losses = []

# Training loop with validation and accuracy
for epoch in range(50):
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

'''
conn.close()
