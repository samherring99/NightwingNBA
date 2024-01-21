from utils.misc_utils import get_all_teams
from utils.player_stat_utils import get_game_stats_by_player
from utils.sql_utils import write_players_data_to_db, write_team_data_to_db, create_database
from utils.team_stat_utils import get_game_stats_for_teams

import requests
import sqlite3
import time
import datetime

conn = sqlite3.connect('nba_datastore.db',
                             detect_types=sqlite3.PARSE_DECLTYPES |
                             sqlite3.PARSE_COLNAMES)
cursor = conn.cursor()

def check_if_game_in_database(game_id, cursor):
    cursor.execute("SELECT game_id FROM player_stats where game_id = {game_id}".format(game_id=game_id))
    game_id = cursor.fetchall()

    if game_id:
        return True
    else:
        return False

def add_game_to_database(idnum, stats, cursor):
    write_players_data_to_db(stats, cursor)
    print("Success adding player data for date: " + str(stats['date']))

    game_team_stats = get_game_stats_for_teams(idnum, stats['date']) 
    if game_team_stats:
        write_team_data_to_db(game_team_stats, cursor)
        print("Success adding team data!")
        return True

def check_game_status(game, cursor):
    id_num = str(game['$ref']).split("?")[0].split("/")[-1]
    if not check_if_game_in_database(id_num, cursor):
        game_players_stats = get_game_stats_by_player(id_num)

        if game_players_stats:
            add_game_to_database(id_num, game_players_stats, cursor)
        time.sleep(0.1)


def iterate_teams(team_list, cursor, conn):

    for id_num in team_list:
        team_games_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024/teams/{team_id}/events?lang=en&region=us".format(team_id=id_num)
        team_games_response = requests.get(team_games_endpoint, params={"page": 1})
        team_game_data = team_games_response.json()
        page_count = team_game_data['pageCount']
        for game in team_game_data['items']:
            if check_game_status(game, cursor):
                conn.commit()
    
        for i in range(page_count-1):
            team_games_response = requests.get(team_games_endpoint, params={"page": int(i+2)})
            team_game_data = team_games_response.json()
            for i in team_game_data['items']:
                if check_game_status(i, cursor):
                    conn.commit()
                    
create_database(cursor, conn)
iterate_teams(get_all_teams(), cursor, conn)

conn.commit()
conn.close()