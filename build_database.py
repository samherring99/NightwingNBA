from utils.misc_utils import get_all_teams
from utils.player_stat_utils import get_game_stats_by_player
from utils.sql_utils import write_players_data_to_db, write_team_data_to_db, create_database
from utils.team_stat_utils import get_game_stats_for_teams
from utils.processing_utils import process_team_data

import requests
import sqlite3
import time

conn = sqlite3.connect('nba_datastore.db',
                             detect_types=sqlite3.PARSE_DECLTYPES |
                             sqlite3.PARSE_COLNAMES)
cursor = conn.cursor()

ENDPOINT="http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024/teams/{team_id}/events?lang=en&region=us"

# Given a page number and a team ID, hit the ESPN endpoint to get the team data for that page
def get_team_data(page, team_id):
    team_games_response = requests.get(ENDPOINT.format(team_id=team_id), params={"page": page})
    team_game_data = team_games_response.json() 

    return team_game_data

# Iterate through all NBA teams, get the team data for each
def iterate_teams(team_list, cursor, connection):

    seen_games = []

    for id_num in team_list:
        team_game_data = get_team_data(1, id_num)
        page_count = team_game_data['pageCount']
        process_team_data(team_game_data, seen_games, cursor, connection)
    
        for i in range(page_count-1):
            team_game_data = get_team_data(int(i+2), id_num)
            process_team_data(team_game_data, seen_games, cursor, connection)
                    
create_database(cursor, conn)
iterate_teams(get_all_teams(), cursor, conn)

conn.commit()
conn.close()