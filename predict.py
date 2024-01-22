import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from utils.player_stat_utils import get_game_stats_by_player, get_player_stats
from utils.team_stat_utils import get_team_stats
from utils.misc_utils import get_all_teams

from matplotlib import pyplot as plt
import numpy as np
import sqlite3
import requests
from datetime import datetime, timedelta

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import BatchNorm1d

#from train import player_template

conn = sqlite3.connect('nba_datastore.db',
                             detect_types=sqlite3.PARSE_DECLTYPES |
                             sqlite3.PARSE_COLNAMES)
cursor = conn.cursor()

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

def get_team_previous_game_stats(game_id, team_id, cursor):
    cursor.execute("SELECT * FROM team_stats WHERE game_id = {game_id} and team_id = {team_id}".format(game_id=game_id, team_id=team_id))
    team_previous_game_stats = cursor.fetchall()

    if team_previous_game_stats:
        return team_previous_game_stats[0]
    else:
        return []

def get_player_data(game_id, player_id, cursor):
    cursor.execute(player_template.format(game_id=game_id, player_id=player_id))
    stats = cursor.fetchall()

    if stats:
        return stats[0]
    else:
        return []

def get_game_date(game_id):
    response = requests.get("http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}?lang=en&region=us".format(game_id=game_id))
    data = response.json()

    game_dict = {'game_id' : game_id, 'date' : data['date'], 'game_name' : data['name']}

    #print(data['name'])

    return game_dict['date']

def get_game_today_for_player(team_id):
    #print("select team_id from player_stats where player_id = {player_id} group by team_id".format(player_id=player_id[0]))

    team_games_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024/teams/{team_id}/events?lang=en&region=us".format(team_id=team_id[0])
    team_games_response = requests.get(team_games_endpoint, params={"page": 1})
    team_game_data = team_games_response.json()

    # TODO methodize the below code

    page_count = team_game_data['pageCount']
    for game in team_game_data['items']:
        id_num = str(game['$ref']).split("?")[0].split("/")[-1]

        game_date = get_game_date(id_num)

        if (datetime.strptime(game_date, '%Y-%m-%dT%H:%MZ') - timedelta(days=1)).strftime('%Y-%m-%d') == datetime.today().strftime('%Y-%m-%d'):
            return id_num

    for i in range(page_count-1):
        team_games_response = requests.get(team_games_endpoint, params={"page": int(i+2)})
        team_game_data = team_games_response.json()
        for i in team_game_data['items']:
            id_num = str(i['$ref']).split("?")[0].split("/")[-1]
            game_date = get_game_date(id_num)

            if str((datetime.strptime(game_date, '%Y-%m-%dT%H:%MZ') - timedelta(days=1)).strftime('%Y-%m-%d')) == str(datetime.today().strftime('%Y-%m-%d')):
                print(game_date)
                return id_num

def get_last_game_for_team(cursor, team_id):
    cursor.execute("select game_id from player_stats where team_id = {team_id} group by game_date order by game_date;".format(team_id=team_id))
    last_game = cursor.fetchall()[-1]

    if last_game:
        return last_game

def get_last_game_for_player(cursor, player_id):
    cursor.execute("select game_id from player_stats where player_id = {player_id} group by game_date order by game_date;".format(player_id=player_id[0]))
    last_game = cursor.fetchall()[-1]

    print(get_game_date(last_game[0]))

    if last_game:
        return last_game

def get_opponent_id(game_id, team_id):
    teams_stats_endpoint = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}/competitions/{game_id}/competitors/".format(game_id=game_id)
    teams_stats_response = requests.get(teams_stats_endpoint)
    if teams_stats_response:
        teams_stats_data = teams_stats_response.json()

    game_teams_stats_dict = {'game_id' : game_id}

    competitors = []

    for competitor in teams_stats_data['items']:
        if competitor['id'] != str(team_id):
            competitors.append(competitor['id'])

    return competitors[0]

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

                    print(player_name[0])

                    last_game_id = get_last_game_for_player(cursor, player)

                    data = []

                    player_previous_game_stats = get_player_data(last_game_id[0], player[0], cursor)

                    for value in list(player_previous_game_stats):
                        data.append(value)

                    team_previous_game_stats = get_team_previous_game_stats(last_game_id[0], team, cursor)

                    for value in team_previous_game_stats[2:]:
                        data.append(value)

                    opponent_id = get_opponent_id(game_today, team)

                    opponent_previous_game_id = get_last_game_for_team(cursor, opponent_id)

                    opponent_previous_game_stats = get_team_previous_game_stats(opponent_previous_game_id[0], opponent_id, cursor)

                    if player_previous_game_stats and len(team_previous_game_stats) > 1 and len(opponent_previous_game_stats) > 1:

                        for value in opponent_previous_game_stats[2:]:
                            data.append(float(value))

                        padded_data = pad_sequence([torch.tensor(data, dtype=torch.float32)], batch_first=True)

                        inputs = torch.tensor(padded_data, dtype=torch.float32)

                        #inputs = torch.tensor([data], dtype=torch.float32)
                        prediction = model(inputs)

                        list_preds = prediction.tolist()

                        print("Prediction: " + str(list_preds[0][0]) + " points, " + str(list_preds[0][1]) + " assists, " + str(list_preds[0][2]) + " rebounds" )    