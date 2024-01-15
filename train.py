from utils.misc_utils import get_all_teams
from utils.player_stat_utils import get_all_player_stats_for_team

import sqlite3
import requests

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
select game_id from player_stats where game_date <= (select game_date from player_stats where game_id = {game_id}) and team_id = {team_id} group by game_date order by game_date limit 5;
'''

def validation(game_id, player_id, guess):
    cursor.execute(player_template.format(game_id=game_id, player_id=player_id))
    player_game_stats = cursor.fetchall()

    if player_game_stats:

        hit_points = player_game_stats[0][53]
        hit_assists = player_game_stats[0][43]
        hit_rebounds = player_game_stats[0][15]

        points_score = 0.0 if hit_points <= guess[0] else 1.0
        assists_score = 0.0 if hit_assists <= guess[1] else 1.0
        rebounds_score = 0.0 if hit_rebounds <= guess[2] else 1.0

        return [points_score, assists_score, rebounds_score]

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

    opposing_team_previous_game_id = game_id

    for i in range(len(previous_opponent_game)):
        if str(previous_opponent_game[i][0]) == str(game_id) and i != 0:
            opposing_team_previous_game_id = previous_opponent_game[i-1][0]
            return opposing_team_previous_game_id
        elif i == 0:
            continue
        else:
            continue

    return opposing_team_previous_game_id

def get_team_previous_game_stats(game_id, team_id):
    cursor.execute(team_template.format(game_id=game_id, team_id=team_id))
    team_previous_game_stats = cursor.fetchall()

    return team_previous_game_stats[0]

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

    guess_points = 10.0
    guess_assists = 10.0
    guess_rebounds = 10.0

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

        guess = [guess_points, guess_assists, guess_rebounds]

        scores = validation(game_id, player_id, guess)

        return [[player_previous_game_stats, team_previous_game_stats, opponent_previous_game_stats], scores]

    else:
        return []


def generate_data(team_list, cursor):
    for team in team_list:
        games_list = get_games_for_team(team)

        for game_list in games_list:
            for i in range(len(game_list)):
                game_id_num = str(game_list[i]['$ref']).split("?")[0].split("/")[-1]

                players = get_team_roster(team)

                for player_id in players:
                    data = get_context_for_game(game_id_num, player_id[0])
                    if data:
                        output = data[1]
                        scores = validation(game_id_num, player_id[0], output)
                        print(data)
                        print(scores)

team_list = get_all_teams()
generate_data(team_list, cursor)

conn.close()