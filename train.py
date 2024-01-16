from utils.misc_utils import get_all_teams

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

roster_template = '''
SELECT player_id FROM player_stats WHERE team_id = {team_id};
'''

player_template = '''
SELECT 
blocks,
defensive_rebounds,
steals,
defensive_rebounds_per_game,
blocks_per_game,
steals_per_game,
defensive_rebounds_per_48,
blocks_per_48,
steals_per_48,
largest_lead,
disqualifications,
flagrant_fouls,
fouls,
ejections,
technical_fouls,
rebounds,
minutes,
minutes_per_game,
rating,
plus_minus,
rebounds_per_game,
fouls_per_game,
flagrant_fouls_per_game,
technical_fouls_per_game,
ejections_per_game,
disqualifications_per_game,
assist_to_turnover_ratio,
steal_to_foul_ratio,
block_to_foul_ratio,
team_rebounds_per_game,
total_technical_fouls,
steal_to_turnover_ratio,
rebounds_per_48_minutes,
fouls_per_48_minutes,
flagrant_fouls_per_48_minutes,
technical_fouls_per_48_minutes,
ejections_per_48_minutes,
disqualifications_per_48_minutes,
r40,
games_played,
games_started,
double_double,
triple_double,
assists,
field_goals,
field_goals_attempted,
field_goals_made,
field_goal_percentage,
free_throws,
free_throw_percentage,
free_throws_attempted,
free_throws_made,
offensive_rebounds,
points,
turnovers,
s_3_point_field_goal_percentage,
s_3_point_field_goals_attempted,
s_3_point_field_goals_made,
total_turnovers,
points_in_the_paint,
brick_index,
average_field_goals_made,
average_field_goals_attempted,
average_3_point_field_goals_made,
average_3_point_field_goals_attempted,
average_free_throws_made,
average_free_throws_attempted,
points_per_game,
offensive_rebounds_per_game,
assists_per_game,
turnovers_per_game,
offensive_rebound_percentage,
estimated_possessions,
estimated_possessions_per_game,
points_per_estimated_possession,
team_turnovers_per_game,
total_turnovers_per_game,
s_2_point_field_goals_made,
s_2_point_field_goals_attempted,
s_2_point_field_goals_made_per_game,
s_2_point_field_goals_attempted_per_game,
s_2_point_field_goal_percentage,
shooting_efficiency,
scoring_efficiency,
fieldgoals_made_per_48,
fieldgoals_attempted_per_48,
s_3_point_fieldgoals_made_per_48,
s_3_point_fieldgoals_attempted_per_48,
freethrows_made_per_48,
freethrows_attempted_per_48,
points_scored_per_48,
offensive_rebounds_per_48,
assists_per_48,
p40,
a40
 FROM player_stats WHERE game_id = {game_id} AND player_id = {player_id};
'''

matchup_template = '''
select t.team_id from player_stats p JOIN team_stats t on p.game_id = t.game_id WHERE p.game_id = {game_id} group by t.team_id;
'''

game_template = '''
SELECT game_name from player_stats where game_id = {game_id}
'''

team_template = '''
SELECT * FROM team_stats WHERE game_id = {game_id} and team_id = {team_id}
'''

get_player_team = '''
select team_id from player_stats where player_id = {player_id} group by team_id
'''

get_team_previous_game = '''
select game_id from player_stats where game_date <= (select game_date from player_stats where game_id = {game_id}) and team_id = {team_id} group by game_date order by game_date;
'''

def get_opponent_id(game_id, team_id):
    cursor.execute(matchup_template.format(game_id=game_id, team_id=team_id))
    game_team_ids = cursor.fetchall()

    opposing_team = 0
    for team in game_team_ids:
        if team[0] != team_id:
            opposing_team = team[0]

    return opposing_team

def get_previous_game_id(game_id, team_id):
    cursor.execute(get_team_previous_game.format(game_id=game_id, team_id=team_id))
    previous_opponent_game = cursor.fetchall()

    if previous_opponent_game:
        opposing_team_previous_game_id = 0

        for i in range(len(previous_opponent_game)):
            if str(previous_opponent_game[i][0]) == str(game_id) and i != 0:
                opposing_team_previous_game_id = previous_opponent_game[i-1][0]
                return opposing_team_previous_game_id
            elif i == 0:
                continue
            else:
                continue
    else:
        return 0

    return opposing_team_previous_game_id

def get_team_previous_game_stats(game_id, team_id):
    cursor.execute(team_template.format(game_id=game_id, team_id=team_id))
    team_previous_game_stats = cursor.fetchall()

    if team_previous_game_stats:
        return team_previous_game_stats[0]
    else:
        return []

def get_team_for_player(player_id):
    cursor.execute(get_player_team.format(player_id=player_id))
    player_team_id = cursor.fetchall()

    if player_team_id:
        return player_team_id[0][0]
    else:
        return []

def get_player_data(game_id, player_id):
    cursor.execute(player_template.format(game_id=game_id, player_id=player_id))
    stats = cursor.fetchall()

    if stats:
        return stats[0]
    else:
        return []

def get_team_roster(team_id):
    cursor.execute(roster_template.format(team_id=team_id))
    players = cursor.fetchall()

    return players

def get_games_for_team(team_id):
    team_games_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024/teams/{team_id}/events?lang=en&region=us".format(team_id=team_id)
    team_games_response = requests.get(team_games_endpoint, params={"page": 1})
    team_game_data = team_games_response.json()
    games = [team_game_data['items']]
    page_count = team_game_data['pageCount']

    for i in range(page_count-1):
        team_games_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024/teams/{team_id}/events?lang=en&region=us".format(team_id=team_id)
        team_games_response = requests.get(team_games_endpoint, params={"page": i+2})
        team_game_data = team_games_response.json()
        games.append(team_game_data['items'])

    return games

def get_context_for_game(game_id, player_id):

    player_team_id = get_team_for_player(player_id)

    team_previous_game_id = get_previous_game_id(game_id, player_team_id)

    player_previous_game_stats = get_player_data(team_previous_game_id, player_id)

    if player_previous_game_stats:

        guess_points = player_previous_game_stats[53]
        guess_assists = player_previous_game_stats[43]
        guess_rebounds = player_previous_game_stats[15]

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

       #points_score = 0.0 if hit_points <= guess[0] else 1.0
        #assists_score = 0.0 if hit_assists <= guess[1] else 1.0
        #rebounds_score = 0.0 if hit_rebounds <= guess[2] else 1.0

        # return [points_score, assists_score, rebounds_score] # classification
        return [hit_points, hit_assists, hit_rebounds] # regression 

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

                        #print(data[0])
                        #print(scores)
                        training_example = [] 

                        for piece in data:
                            for number in piece:
                                training_example.append(number) # flatten np.flatten()

                        #print(training_example)

                        if training_example and scores:
                            X.append(training_example)      # X:  N x 283
                            y.append(scores)                # y:  N x 3 

                        #print(X)
                        #print(y)

    return X, y 

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

# Save the datasets
torch.save(train_dataset, 'train_dataset.pt')
torch.save(val_dataset, 'val_dataset.pt')
torch.save(X_test_torch, 'X_test.pt')
torch.save(y_test_torch, 'y_test.pt')

# Load the datasets
# train_dataset = torch.load('train_dataset.pt')
# val_dataset = torch.load('val_dataset.pt')
# X_test_torch = torch.load('X_test.pt')
# y_test_torch = torch.load('y_test.pt')

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
# Plot the loss
# Evaluate predictions



conn.close()












# new features like home vs away game, averages, time series, etc

# df['feature'] = y 

# Visualize before normalizing 

#plt.figure(figsize=(15, 15))  

#num_cols = len(df.columns)   
#n_rows = 80                    
#n_cols = 4  


#sns.histplot(df, bins=286)

#for i, col in enumerate(df.columns):
#    plt.subplot(n_rows, n_cols, i+1)  
#    sns.histplot(df[col], bins=30)      # histogram for each column / feature
#    plt.title(col)                      



#correlation_data = df.corr() # not sure if this will work 
#plt.figure(figsize=(12, 10))
#sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'shrink': .82})
#plt.title('Feature Correlation')
#plt.xticks(rotation=45, ha='right')
#plt.yticks(rotation=0)
#plt.tight_layout()
#plt.show()


# Normalize

# Visualize after normalizing
#print(y)
