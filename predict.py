import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from utils.player_stat_utils import get_game_stats_by_player, get_player_stats
from utils.team_stat_utils import get_team_stats
from utils.misc_utils import get_all_teams
from utils.processing_utils import get_team_previous_game_stats, get_player_data, get_game_date, check_if_game_before_today, get_game_today_for_player, get_last_game_for_team, get_last_game_for_player, get_opponent_id, get_next_opponent_id

from matplotlib import pyplot as plt
import numpy as np
import sqlite3
import requests
from datetime import datetime, timedelta

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import BatchNorm1d

conn = sqlite3.connect('nba_datastore.db',
                             detect_types=sqlite3.PARSE_DECLTYPES |
                             sqlite3.PARSE_COLNAMES)
cursor = conn.cursor()

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

model = NeuralNetwork()

model.load_state_dict(torch.load("./saved_data/weights.pth"))

seen_players = []

model.eval()
with torch.no_grad():

    for team in get_all_teams():

        print(team)

        game_today = get_game_today_for_player([team])

        if game_today:

            cursor.execute("SELECT player_id FROM player_stats WHERE team_id = {team_id};".format(team_id=team))
            roster = cursor.fetchall()

            for player in roster:
                if player not in seen_players:
                    seen_players.append(player)
                    cursor.execute("select player_name from player_stats where player_id = {player_id} and team_id = {team_id};".format(player_id=player[0], team_id=team))
                    player_name = cursor.fetchall()[0]

                    last_game_id = get_last_game_for_player(cursor, player)

                    data = []

                    player_previous_game_stats = get_player_data(last_game_id[0], player[0], cursor)

                    for value in list(player_previous_game_stats):
                        data.append(value)

                    last_team_game = get_last_game_for_team(cursor, team)

                    team_previous_game_stats = get_team_previous_game_stats(last_team_game[0], team, cursor)

                    for value in team_previous_game_stats[2:]:
                        data.append(value)

                    opponent_id = get_next_opponent_id(game_today, team)

                    opponent_previous_game_id = get_last_game_for_team(cursor, opponent_id)

                    if opponent_previous_game_id:

                        opponent_previous_game_stats = get_team_previous_game_stats(opponent_previous_game_id[0], opponent_id, cursor)

                        if player_previous_game_stats and len(team_previous_game_stats) > 1 and len(opponent_previous_game_stats) > 1:

                            for value in opponent_previous_game_stats[2:]:
                                data.append(float(value))

                            padded_data = pad_sequence([torch.tensor(data, dtype=torch.float32)], batch_first=True)

                            inputs = torch.tensor(padded_data, dtype=torch.float32)

                            prediction = model(inputs)

                            list_preds = prediction.tolist()

                            # TODO sort by time

                            cursor.execute("SELECT game_name, team_name from player_stats where game_id = {game_id} and team_id = {team_id} group by game_name, team_name".format(game_id=game_today, team_id=team))
                            game_name = cursor.fetchall()

                            print(game_name)

                            print(player_name[0])

                            print("Prediction: " + str(list_preds[0][0]) + " points, " + str(list_preds[0][1]) + " assists, " + str(list_preds[0][2]) + " rebounds" )    