def write_player_and_team_data(player_data, team_data, game_id, cursor):
    write_players_data_to_db(player_data, cursor)
    print("Success adding player data!")

    game_team_stats = get_game_stats_for_teams(game_id, player_data['date']) 

    if game_team_stats:
        write_team_data_to_db(team_data, cursor)
        print("Success adding team data!")

# Top level method for database building

def process_game(game, seen_games, cursor, conn):
    id_num = str(game['$ref']).split("?")[0].split("/")[-1]
    if id_num not in seen_games:
        game_players_stats = get_game_stats_by_player(id_num)

        if game_players_stats:
            write_player_and_team_data(game_players_stats, data, id_num, cursor)

        seen_games.append(id_num)
        conn.commit()
        time.sleep(0.1)
        return id_numx

def process_team_data(data, seen_games, cursor, conn):
    for game in data['items']:
        process_game(game, seen_games, cursor, conn)


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
    if id_num and not check_if_game_in_database(id_num, cursor):
        game_players_stats = get_game_stats_by_player(id_num)

        if game_players_stats:
            return add_game_to_database(id_num, game_players_stats, cursor)
        time.sleep(0.1)

def add_games(data, cursor, connection):
    for game in data['items']:
        if check_game_status(game, cursor):
            connection.commit()