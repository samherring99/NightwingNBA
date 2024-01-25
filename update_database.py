from utils.misc_utils import get_all_teams
from utils.player_stat_utils import get_game_stats_by_player
from utils.sql_utils import write_players_data_to_db, write_team_data_to_db, create_database
from utils.team_stat_utils import get_game_stats_for_teams
from utils.processing_utils import check_if_game_in_database, add_game_to_database, check_game_status, add_games

import requests
import sqlite3
import time
import datetime

conn = sqlite3.connect('nba_datastore.db',
                             detect_types=sqlite3.PARSE_DECLTYPES |
                             sqlite3.PARSE_COLNAMES)
cursor = conn.cursor()

def iterate_teams(team_list, cursor, conn):

    for id_num in team_list:
        team_games_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024/teams/{team_id}/events?lang=en&region=us".format(team_id=id_num)
        team_games_response = requests.get(team_games_endpoint, params={"page": 1})
        team_game_data = team_games_response.json()
        page_count = team_game_data['pageCount']
        add_games(team_game_data, cursor, conn)
    
        for i in range(page_count-1):
            team_games_response = requests.get(team_games_endpoint, params={"page": int(i+2)})
            team_game_data = team_games_response.json()
            add_games(team_game_data, cursor, conn)
                    
create_database(cursor, conn)
iterate_teams(get_all_teams(), cursor, conn)

conn.commit()
conn.close()