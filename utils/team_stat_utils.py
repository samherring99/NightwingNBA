import requests
import time
from datetime import datetime

# TODO finish building out the teams stats table

def get_team_stats(game_id, team_id):
    team_stats_endpoint = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}/competitions/{game_id}/competitors/{team_id}/statistics".format(game_id=game_id, team_id=team_id)
    team_stats_response = requests.get(team_stats_endpoint)
    team_stats_data = team_stats_response.json()

    team_dict = {'team_id' : team_id}

    if 'splits' in team_stats_data:
        for category in team_stats_data['splits']['categories']:
            for stat in category['stats']:
                team_dict[stat['displayName']] = str(stat['value'])

    return team_dict

def get_game_stats_for_teams(game_id, game_date):
    if datetime.strptime(game_date, '%Y-%m-%dT%H:%MZ') < datetime.today():
        teams_stats_endpoint = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}/competitions/{game_id}/competitors/".format(game_id=game_id)
        teams_stats_response = requests.get(teams_stats_endpoint)
        if teams_stats_response:
            teams_stats_data = teams_stats_response.json()

        game_teams_stats_dict = {'game_id' : game_id}

        for competitor in teams_stats_data['items']:
            game_teams_stats_dict[competitor['id']] = get_team_stats(game_id, competitor['id'])

            time.sleep(0.1)

        return game_teams_stats_dict
    else:
        return {}
    