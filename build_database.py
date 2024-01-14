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

def iterate_teams(team_list, cursor, connection):
    seen_games = []

    # TODO - make the below into smaller methods

    for id_num in team_list:
        team_games_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024/teams/{team_id}/events?lang=en&region=us".format(team_id=id_num)
        team_games_response = requests.get(team_games_endpoint, params={"page": 1})
        team_game_data = team_games_response.json()
        page_count = team_game_data['pageCount']
        for game in team_game_data['items']:
            id_num = str(game['$ref']).split("?")[0].split("/")[-1]
            if id_num not in seen_games:
                game_players_stats = get_game_stats_by_player(id_num)

                if game_players_stats:
                    seen_games.append(id_num)
                    write_players_data_to_db(game_players_stats, cursor)
                    print("Success adding player data!")

                    game_team_stats = get_game_stats_for_teams(id_num, game_players_stats['date']) 

                    if game_team_stats:
                        write_team_data_to_db(game_team_stats, cursor)
                        print("Success adding team data!")

                connection.commit()
                time.sleep(0.1)
    
        for i in range(page_count-1):
            team_games_response = requests.get(team_games_endpoint, params={"page": int(i+2)})
            team_game_data = team_games_response.json()
            for i in team_game_data['items']:
                id_num = str(i['$ref']).split("?")[0].split("/")[-1]
                if id_num not in seen_games:
                    game_players_stats = get_game_stats_by_player(id_num)

                    if game_players_stats:
                        seen_games.append(id_num)
                        write_players_data_to_db(game_players_stats, cursor)
                        print("Success adding player data!")

                        game_team_stats = get_game_stats_for_teams(id_num, game_players_stats['date']) 

                        if game_team_stats:
                            write_team_data_to_db(game_team_stats, cursor)
                            print("Success adding team data!")

                    connection.commit()        
                    time.sleep(0.1)
                    
create_database(cursor, conn)
iterate_teams(get_all_teams(), cursor, conn)

conn.commit()
conn.close()