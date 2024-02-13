from utils.templates import *
import requests
import sqlite3
from datetime import datetime, timedelta
from utils.player_stat_utils import get_game_stats_by_player
from utils.sql_utils import write_players_data_to_db, write_team_data_to_db
from utils.team_stat_utils import get_game_stats_for_teams
import time

############################## BUILD DATABASE ##############################

# Given all player's data and team data in JSON format, and a game ID, write it to the database
def write_player_and_team_data(player_data, game_id, cursor):
    write_players_data_to_db(player_data, cursor)
    print("Success adding player data!")

    game_team_stats = get_game_stats_for_teams(game_id, player_data['date']) 

    if game_team_stats:
        write_team_data_to_db(game_team_stats, cursor)
        print("Success adding team data!")

# Given game data in JSON format and a list of seen games, process the game and write it to the database
def process_game(game, seen_games, cursor, conn):
    id_num = str(game['$ref']).split("?")[0].split("/")[-1]
    if id_num not in seen_games:
        game_players_stats = get_game_stats_by_player(id_num)

        if game_players_stats:
            write_player_and_team_data(game_players_stats, id_num, cursor)

        seen_games.append(id_num)
        conn.commit()
        time.sleep(0.1)
        return id_num

# Given team data in JSON format and a list of seen games, 
# process each game for the team and write it to the database
def process_team_data(data, seen_games, cursor, conn):
    for game in data['items']:
        process_game(game, seen_games, cursor, conn)

############################## UPDATE DATABASE ##############################

# Check if the given game ID exists in the database
def check_if_game_in_database(game_id, cursor):
    cursor.execute("SELECT game_id FROM player_stats where game_id = {game_id}".format(game_id=game_id))
    game_id = cursor.fetchall()

    if game_id:
        return True
    else:
        return False

# Given a game ID and player stats in JSON format, write player and team data to the database
def add_game_to_database(game_id, stats, cursor):
    write_players_data_to_db(stats, cursor)
    print("Success adding player data for date: " + str(stats['date']))

    game_team_stats = get_game_stats_for_teams(game_id, stats['date']) 
    if game_team_stats:
        write_team_data_to_db(game_team_stats, cursor)
        print("Success adding team data!")
        return True

# Given game data in JSON, check if the game is not in the database and needs to be added
def check_game_status(game, cursor):
    id_num = str(game['$ref']).split("?")[0].split("/")[-1]
    if id_num and not check_if_game_in_database(id_num, cursor):
        game_players_stats = get_game_stats_by_player(id_num)

        if game_players_stats:
            return add_game_to_database(id_num, game_players_stats, cursor)
        time.sleep(0.1)

# Iterate through the games in a list of team's games and add each one if they need to be added.
def add_games(data, cursor, connection):
    for game in data['items']:
        if check_game_status(game, cursor):
            connection.commit()

############################## WRITE DATA TO PYTORCH ##############################

# Given a game ID and a team ID, get the opponent's team ID
def get_opponent_id(game_id, team_id, cursor):
    cursor.execute(matchup_template.format(game_id=game_id, team_id=team_id))
    game_team_ids = cursor.fetchall()

    opposing_team = 0
    for team in game_team_ids:
        if team[0] != team_id:
            opposing_team = team[0]

    return opposing_team

# Given a game ID and a team ID, get the previous game ID for that team from the database
def get_previous_game_id(game_id, team_id, cursor):
    cursor.execute(get_team_previous_game.format(game_id=game_id, team_id=team_id))
    previous_opponent_game = cursor.fetchall()

    opposing_team_previous_game_id = 0

    if previous_opponent_game:

        for i in range(len(previous_opponent_game)):
            if str(previous_opponent_game[i][0]) == str(game_id) and i != 0:
                opposing_team_previous_game_id = previous_opponent_game[i-1][0]
                return opposing_team_previous_game_id
    return opposing_team_previous_game_id

# Get the team ID for a given player ID
def get_team_for_player(player_id, cursor):
    cursor.execute(get_player_team.format(player_id=player_id))
    player_team_id = cursor.fetchall()

    if player_team_id:
        return player_team_id[0][0]
    else:
        return []

# Given a game ID and a player ID, get the player stat data 
def get_player_data(game_id, player_id, cursor):
    cursor.execute(player_template.format(game_id=game_id, player_id=player_id))
    stats = cursor.fetchall()

    if stats:
        return stats[0]
    else:
        return []

# Get the roster for a given team ID, returns a list of player IDs
def get_team_roster(team_id, cursor):
    cursor.execute(roster_template.format(team_id=team_id))
    players = cursor.fetchall()

    if players:
        return players
    else:
        return []

# Get a list of all game IDs for a given team ID
# This method is needed because we haven't added future games to the database
# Can change this in the future but would need to change how we update the games etc.
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

############################## PREDICT ##############################

# Given a game ID and a team ID get the team's stats from the game 
def get_team_previous_game_stats(game_id, team_id, cursor):
    cursor.execute("SELECT * FROM team_stats WHERE game_id = {game_id} and team_id = {team_id}".format(game_id=game_id, team_id=team_id))
    team_previous_game_stats = cursor.fetchall()

    if team_previous_game_stats:
        return team_previous_game_stats[0]
    else:
        return []

# Given a game ID get the game date
def get_game_date(game_id):
    response = requests.get("http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}?lang=en&region=us".format(game_id=game_id))
    data = response.json()

    game_dict = {'game_id' : game_id, 'date' : data['date'], 'game_name' : data['name']}

    time.sleep(0.1)

    return game_dict

# Given a game date check if it is before today
def check_if_game_before_today(game_date):
    return (datetime.strptime(game_date, '%Y-%m-%dT%H:%MZ') - timedelta(days=1)).strftime('%Y-%m-%d') == datetime.today().strftime('%Y-%m-%d')

# Given a list of games find the game that is happening today, return the ID if so, return 0 if no games are happening today
def find_game_today_in_team_data(data):
    for game in data['items']:
        id_num = str(game['$ref']).split("?")[0].split("/")[-1]

        game_data = get_game_date(id_num)
        game_date = game_data['date']

        if check_if_game_before_today(game_date):
            print(game_data['game_name']) # Add game name to dict
            return id_num
    return 0

# Given a team ID, get the game happening for that team today
def get_game_today_for_player(team_id):

    team_games_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024/teams/{team_id}/events?lang=en&region=us".format(team_id=team_id[0])
    team_games_response = requests.get(team_games_endpoint, params={"page": 1})
    team_game_data = team_games_response.json()

    page_count = team_game_data['pageCount']

    id_num = find_game_today_in_team_data(team_game_data)

    if id_num != 0:
        return id_num

    for i in range(page_count-1):
        team_games_response = requests.get(team_games_endpoint, params={"page": int(i+2)})
        team_game_data = team_games_response.json()

        id_num = find_game_today_in_team_data(team_game_data)

        if id_num != 0:
            return id_num

# Given a team ID, get the last game ID the team has played
def get_last_game_for_team(cursor, team_id):
    cursor.execute("select game_id from player_stats where team_id = {team_id} group by game_date order by game_date;".format(team_id=team_id))
    last_games = cursor.fetchall()

    if last_games:
        return last_games[-1]

# Given a player ID, get the last game ID the player has played
def get_last_game_for_player(cursor, player_id):
    cursor.execute("select game_id from player_stats where player_id = {player_id} group by game_date order by game_date;".format(player_id=player_id[0]))
    last_game = cursor.fetchall()[-1]

    if last_game:
        return last_game

# Given a game ID and a team ID, get the next game that hasn't happened yet for the team
# Can change this in the future if we change the way we check game eligibility
def get_next_opponent_id(game_id, team_id):
    teams_stats_endpoint = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}/competitions/{game_id}/competitors/".format(game_id=game_id)
    teams_stats_response = requests.get(teams_stats_endpoint)
    if teams_stats_response:
        teams_stats_data = teams_stats_response.json()

    game_teams_stats_dict = {'game_id' : game_id}

    competitors = []

    for competitor in teams_stats_data['items']:
        if competitor['id'] != str(team_id):
            competitors.append(competitor['id'])

    return competitors[0]

# https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/