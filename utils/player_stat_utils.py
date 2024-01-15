import requests
from datetime import datetime
import time

# TODO see what we can optimize out of the below

def get_player_stats(game_id, team_id, player_id):
    player_stats_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}/competitions/{game_id}/competitors/{team_id}/roster/{player_id}/statistics/0?lang=en&region=us".format(game_id=game_id, team_id=team_id, player_id=player_id)
    player_stats_response = requests.get(player_stats_endpoint)
    player_stats_data = player_stats_response.json()

    stats_dict = {'player_id' : player_id}

    if 'splits' in player_stats_data:

        for category in player_stats_data['splits']['categories']:
            for stat in category['stats']:
                stats_dict[stat['displayName']] = str(stat['value'])

        return stats_dict
    else:
        return {}

def get_all_player_stats_for_team(game_id, team_id):
    team_boxscore_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}/competitions/{game_id}/competitors/{team_id}/roster?lang=en&region=us".format(game_id=game_id, team_id=team_id)
    team_boxscore_response = requests.get(team_boxscore_endpoint)
    team_boxscore_data = team_boxscore_response.json()

    players_dict = {'team_id' : team_id}

    for player in team_boxscore_data['entries']:

        if not player['didNotPlay']:
            players_dict[player['displayName']] = get_player_stats(game_id, team_id, player['playerId'])

        time.sleep(0.1)

    return players_dict

def get_game_stats_by_player(game_id):
    response = requests.get("http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}?lang=en&region=us".format(game_id=game_id))
    data = response.json()

    game_dict = {'game_id' : game_id, 'date' : data['date'], 'game_name' : data['name']}

    if datetime.strptime(data['date'], '%Y-%m-%dT%H:%MZ') < datetime.today():

        print("Getting player stats for " + data['name'] + " on " + data['date'])

        for team in data['competitions'][0]['competitors']:
            team_names = data['name'].split(" at ")
            team_name = team_names[0].split(" ")[-1] if team['homeAway'] == 'home' else team_names[1].split(" ")[-1]

            game_dict[team_name] = get_all_player_stats_for_team(game_id, team['id'])

            time.sleep(0.1)

        return game_dict
    else:
        return {}