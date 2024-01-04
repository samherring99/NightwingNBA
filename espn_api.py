import requests
import sqlite3
from datetime import datetime

conn = sqlite3.connect('nba_stats.db',
                             detect_types=sqlite3.PARSE_DECLTYPES |
                             sqlite3.PARSE_COLNAMES)
cursor = conn.cursor()

# This method returns the data from a given URL using the requests library
def get_request(url):
    response = requests.get(url)
    data = response.json()
    return data

# This method finds the location of the requested statistic name in a given 
# dictionary of NBA statistics for a given player. 
# It returns the value of the requested statistic
def get_player_average(stats, stat_name):
    avg_value = 0

    # Find the location of the statistic value given the name of the stat
    avg_value_obj = next((item for item in stats if item["name"] == stat_name), None)

    # Return the display value if it exists
    if avg_value_obj is not None:
        avg_value = avg_value_obj['displayValue']

    return avg_value


# This is the top level method that handles all operations to get ESPN statsitics for NBA players
def create_nba_database():

    # Get the list of NBA teams from ESPN 
    nba_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
    teams_data = get_request(nba_url)

    # Iterate over the teams in the list
    for team in teams_data["sports"][0]["leagues"][0]["teams"]:
        print(team["team"]["id"])
        print(team["team"]["displayName"])

        # Get the roster of the team we are currently on when iterating through teams
        team_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{id}/roster".format(id=team["team"]["id"])
        players_roster = get_request(team_url)

        # Iterate over the players in the roster
        print("Getting player stats")
        for athlete in players_roster["athletes"]:
            print(athlete["id"])
            print(athlete["fullName"])

            # Get the stats for the player we are on when iterating over the players in the team we are currently looking at
            player_url = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/athletes/{id}/statistics".format(id=athlete["id"])
            response = requests.get(player_url)
            player = response.json()

            # Storage variables
            avg_points = 0
            avg_rebounds = 0
            avg_assists = 0

            # If we get data, iterate over the stats of the player
            if response.status_code != 404:
                for stattype in player["splits"]["categories"]: #General, Defensive, Offensive
                    for stat in stattype["stats"]:
                        if stat["name"] in ["avgPoints", "avgAssists", "avgRebounds"]:
                            print(stat["displayName"] + " " + str(stat["value"]))

                    # Update the storage variables
                    avg_points = get_player_average(stattype["stats"], "avgPoints")
                    avg_rebounds = get_player_average(stattype["stats"], "avgRebounds")
                    avg_assists = get_player_average(stattype["stats"], "avgAssists")

                # Create an entry for the player
                player_data = (athlete["fullName"], str(team["team"]["displayName"]), "Team", str(avg_points), str(avg_rebounds), str(avg_assists), '0', '0', '0', '0', '0', '0', '0', '0', '0', datetime.today().strftime('%Y-%m-%d'))

                print(player_data)

                # Add the player data into the database
                cursor.execute('''
                    INSERT INTO nba_statistics (player_name, player_team, next_matchup, avg_points, avg_rebounds, avg_assists, predicted_points, ppg_over, ppg_under, predicted_rebounds, prg_over, prg_under, predicted_assists, pag_over, pag_under, game_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', player_data)

create_nba_database()
conn.commit()
conn.close()