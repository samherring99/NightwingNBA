from templates import roster_template, player_template, matchup_template, game_template, team_template, get_player_team, get_team_previous_game


def write_player_and_team_data(player_data, team_data, game_id, cursor):
    write_players_data_to_db(player_data, cursor)
    print("Success adding player data!")

    game_team_stats = get_game_stats_for_teams(game_id, player_data['date']) 

    if game_team_stats:
        write_team_data_to_db(team_data, cursor)
        print("Success adding team data!")

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

    opposing_team_previous_game_id = 0

    if previous_opponent_game:

        for i in range(len(previous_opponent_game)):
            if str(previous_opponent_game[i][0]) == str(game_id) and i != 0:
                opposing_team_previous_game_id = previous_opponent_game[i-1][0]
                return opposing_team_previous_game_id
    return opposing_team_previous_game_id

def get_team_previous_game_stats(game_id, team_id):
    cursor.execute(team_template.format(game_id=game_id, team_id=team_id))
    team_previous_game_stats = cursor.fetchall()

    if team_previous_game_stats:
        return team_previous_game_stats[0]
    else:
        return []

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

    if players:
        return players
    else:
        return []

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