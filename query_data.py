import sqlite3
import random
from datetime import datetime, timedelta
from itertools import groupby

# This file generates the string for SQL queries and executes them based on the following steps:
# - Generate 10 SQL queries of parlays from the NBA statistics and odds database 
#   created with espn_api.py (nba_stats.db) with the following requirements:
#   - For ~8 of these queries, search for negative odds picks (liklier to happen)
#   - For the remaining ~2 queries, search for positive odds between 0 and 120 (less likely)
#   - Query randomly for points, assists, or rebounds where the average value is higher than the sportsbook's predicted value
# - Using the returned entries that meet the above criteria, group them randomly into groups of 5
# - Filter each group down to only valid bets for the sportsbook
# - Return the list of remaining parlays sorted by length (number of legs)

# This method creates the 'query string' used to extract data from the NBA statsistics database
def create_query():
    score_type = random.choice(["rebounds", "assists", "points"])
    identifier=score_type[0]
    mark = "p{id}g".format(id=identifier)
    
    outcomes = ["< 0", "> 0 AND {mark}_over < 120".format(mark=mark)]
    outcome = random.choices(outcomes, cum_weights=[0.8, 0.2], k=1)

    query_string = """
        SELECT player_name, player_team, avg_{type}, predicted_{type}, {mark}_over, game_date, next_matchup FROM nba_statistics
        WHERE game_date LIKE "%{game_date}%" AND avg_{type} > predicted_{type} AND {mark}_over {tail};
    """.format(type=score_type, mark=mark, tail=outcome[0], game_date=str((datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')) + " 00:00:00")

    return [query_string, score_type]

# Iterate 10 times, creating a query each time that randomly picks valid
# entires that meet the criteria of being < 0 odds (likely) 80% of the time or between 
# 0-120 (less likely, more money) and between querying for points, 
# rebounds, and assists that meet the criteria
def generate_bets(cursor):
    picks = []
    for i in range(10):

        # Generate a query and execute on the database
        query_pair = create_query()
        cursor.execute(query_pair[0])
        result = cursor.fetchall()

        for pick in result:
            bet_string = query_pair[1] + "-over"
            bet = tuple(filter(lambda x: x != pick[-2], pick)) # Don't include the game date in making picks
            bet = bet + tuple([bet_string])
            picks.append(bet)

    return picks


# This method merges picks from two teams that are playing in the same matchup, given a list of games.
# It will then return the merged list of bets for the given game
def merge_matchup_teams(matchup, games):
    bets = []
    for i in range(2):
        if matchup[i] in games:
            for pick in games[matchup[i]]:
                bets.append(pick)    

    return bets

# This method breaks out the list of bets into parlays by game, where each parlay
# has only picks from players that are playing each other
def break_out_bets(bets):
    matchups = {}
    parlays = {}

    for bet in bets:
        if str(bet[1]).split(" ")[-1] not in matchups:
            matchups[str(bet[1]).split(" ")[-1]] = [bet]
        else:
            current = matchups[str(bet[1]).split(" ")[-1]]
            current.append(bet)
            matchups[str(bet[1]).split(" ")[-1]] = current

    for team in matchups:
        matchup = sorted([team.split(" ")[-1], matchups[team][0][5].split(" ")[-1]])
        parlays[str(matchup[0] + "-" + matchup[1])] = merge_matchup_teams(matchup, matchups)

    return parlays

# Print method for the parlays to abstract most of the logic out of the below main method
def print_parlays(parlays):
    log_string = ""
    for key in parlays:
        print("GAME -> " + key.split("-")[0] + " vs " + key.split("-")[1])
        for bet in parlays[key]:
            bet_components = bet[6].split("-")
            output = bet[0] + " " + bet_components[0] + " " + bet_components[1] + " " + str(bet[3]) + " at " + str(bet[4])
            print(output)
            log_string = log_string + output + "\n"
        print("\n")

    file_name = "picks/picks_{month}_{day}.txt".format(month='{:02d}'.format(datetime.now().month), day='{:02d}'.format(datetime.now().day))
    logfile = open(file_name, 'w')
    logfile.write(log_string)
    logfile.close()

# Top level execution method used to call all methods above in order.
def execute():

    # Connect to the NBA stats database
    conn = sqlite3.connect('nba_stats.db')
    cursor = conn.cursor()

    bets = generate_bets(cursor)

    # Calculate total picks generated and the number of parlays to create (picks / avg 5 legs)
    print("Total picks: " + str(len(bets)))
    num_bets = int(len(bets) / 5)
    print("# of parlays to create: " + str(num_bets) + "\n")

    # Remove duplicates
    bets = list(set(bets))
    parlays = break_out_bets(bets)

    print_parlays(parlays)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

execute()