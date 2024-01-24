from utils.misc_utils import get_all_teams
from utils.player_stat_utils import get_game_stats_by_player
from utils.sql_utils import write_players_data_to_db, write_team_data_to_db, create_database
from utils.team_stat_utils import get_game_stats_for_teams

import requests
import sqlite3
import time

conn = sqlite3.connect('nba_datastore.db',
                             detect_types=sqlite3.PARSE_DECLTYPES |
                             sqlite3.PARSE_COLNAMES)
cursor = conn.cursor()

# Player or team

def write_player_and_team_data(player_data, team_data, game_id, cursor):
    write_players_data_to_db(player_data, cursor)
    print("Success adding player data!")

    game_team_stats = get_game_stats_for_teams(game_id, player_data['date']) 

    if game_team_stats:
        write_team_data_to_db(team_data, cursor)
        print("Success adding team data!")

# Top level method for database building

def process_game(game_id, data, cursor, conn):
    id_num = str(game['$ref']).split("?")[0].split("/")[-1]
    if id_num not in seen_games:
        game_players_stats = get_game_stats_by_player(id_num)

        if game_players_stats:
            write_player_and_team_data(game_players_stats, data, id_num, cursor)

        conn.commit()
        time.sleep(0.1)
        return id_num

# TODO This could use some methodizing

def iterate_teams(team_list, cursor, connection):
    seen_games = []

    for id_num in team_list:
        team_games_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024/teams/{team_id}/events?lang=en&region=us".format(team_id=id_num)
        team_games_response = requests.get(team_games_endpoint, params={"page": 1})
        team_game_data = team_games_response.json()
        page_count = team_game_data['pageCount']
        for game in team_game_data['items']:
            game_id = process_game(id_num, team_game_data, cursor, connection)
            seen_games.append(game_id)
    
        for i in range(page_count-1):
            team_games_response = requests.get(team_games_endpoint, params={"page": int(i+2)})
            team_game_data = team_games_response.json()
            for i in team_game_data['items']:
                game_id = process_game(id_num, team_game_data, cursor, connection)
                seen_games.append(game_id)
                    
create_database(cursor, conn)
iterate_teams(get_all_teams(), cursor, conn)

conn.commit()
conn.close()