from utils.misc_utils import get_all_teams
from utils.player_stat_utils import get_game_stats_by_player
from utils.sql_utils import write_players_data_to_db, write_team_data_to_db, create_database
from utils.team_stat_utils import get_game_stats_for_teams
from utils.processing_utils import add_games

import requests
import sqlite3
import time
import datetime

conn = sqlite3.connect('nba_datastore.db',
                             detect_types=sqlite3.PARSE_DECLTYPES |
                             sqlite3.PARSE_COLNAMES)
cursor = conn.cursor()

ENDPOINT="http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024/teams/{team_id}/events?lang=en&region=us"

# This method takes in a page number and team ID and returns the 
# list of games for that page of the team's games this season
def process_game_page(page, team_id):
    team_games_response = requests.get(ENDPOINT.format(team_id=team_id), params={"page": page})
    team_game_data = team_games_response.json()
    add_games(team_game_data, cursor, conn)
    return team_game_data

# For each team ID, process through each page of their game IDs this season
# This method calls the above process_game_page, which in turn calls add_games to add games to the database
def iterate_teams(team_list, cursor, conn):
    for id_num in team_list:
        team_game_data = process_game_page(1, id_num)
        page_count = team_game_data['pageCount']
    
        for i in range(page_count-1):
            team_game_data = process_game_page(int(i+2), id_num)
                    
create_database(cursor, conn)
iterate_teams(get_all_teams(), cursor, conn)

conn.commit()
conn.close()