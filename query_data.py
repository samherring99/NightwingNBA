import sqlite3
import random
from datetime import datetime

# This file generates the string for SQL queries and executes them based on the following steps:
# - Generate 10 SQL queries of parlays from the NBA statistics and odds database 
#   created with espn_api.py (nba_stats.db) with the following requirements:
#   - For ~8 of these queries, search for negative odds picks (liklier to happen)
#   - For the remaining ~2 queries, search for positive odds between 0 and 120 (less likely)
#   - Query randomly for points, assists, or rebounds where the average value is higher than the sportsbook's predicted value
# - Using the returned entries that meet the above criteria, group them randomly into groups of 5
# - Filter each group down to only valid bets for the sportsbook
# - Return the list of remaining parlays sorted by length (number of legs)

# Connect to the NBA stats database
conn = sqlite3.connect('nba_stats.db')
cursor = conn.cursor()

# Array to store picks
total_picks = []

# This method creates the 'query string' used to extract data from the NBA statsistics database
def create_query():
    score_type = random.choice(["rebounds", "assists", "points"])
    identifier=score_type[0]
    mark = "p{id}g".format(id=identifier)

    
    outcomes = ["< 0", "> 0 AND {mark}_over < 120".format(mark=mark)]
    outcome = random.choices(outcomes,cum_weights=[0.8, 0.2], k=1)

    query_string = """
    
        SELECT player_name, avg_{type}, predicted_{type}, {mark}_over, game_date FROM nba_statistics
        WHERE game_date = {today} avg_{type} > predicted_{type} AND {mark}_over {tail};

    """.format(type=score_type, mark=mark, tail=outcome, today=str(datetime.date.today()))

    return [query_string, score_type]

# Iterate 10 times, creating a query each time that randomly picks valid
# entires that meet the criteria of being < 0 odds (likely) 80% of the time or between 
# 0-120 (less likely, more money) and between querying for points, 
# rebounds, and assists that meet the criteria
for i in range(10):

    query_pair = create_query()

    cursor.execute(query_pair[0])
    result = cursor.fetchall()

    for pick in result:
        bet_string = query_pair[1] + "-over"
        pick = pick + tuple([bet_string])
        total_picks.append(pick)

# Calculate total picks generated and the number of parlays to create (picks / 5 legs)
print("Total picks: " + str(len(total_picks)))
num_bets = int(len(total_picks) / 5)
print("# of parlays to create: " + str(num_bets) + "\n")

# Remove duplicates and shuffle the list
total_picks = list(set(total_picks))
random.shuffle(total_picks)

# This method breaks out the list of bets into distinct groups of 5
def break_out_bets():
    for i in range(0, len(total_picks), 5):  
        yield total_picks[i:i + 5]

# Make a list of 5 leg parlays using the above method
parlays = list(break_out_bets())

# Empty array to store the list of parlays after filtering
parlay_list = []

# Iterate over the list of 5 leg parlays created above
for parlay in parlays:
    # Create an empty array to store the legs of the parlay
    legs = []
    # Iterate over each leg in the current 5 leg parlay
    for leg in list(parlay):
        # If 
        if leg[2] % 5 == 0 or leg[2] in [4, 6, 8, 10]:
            legs.append(str(leg[0]) + " " + str(leg[4]).split("-")[0] + " " + str(leg[4]).split("-")[1] + " " + str(leg[2]) + " " + str(leg[3]))
    parlay_list.append(legs)

count = 1

# This method is used as a sort key for the list of parlays base don how many legs they have that are valid bets
def parlength(parlay):
    return len(parlay)

# Sort the remaining parlays by length
parlay_list.sort(key=parlength)

# Iterate over the list of parlays
for parlay in parlay_list:
    # If we have an actual parlay in the current parlay (2+ legs)
    if len(parlay) >= 2:
        print("PARLAY #{count} --------------------------------------------------".format(count=count))
        # Print each leg
        for p in parlay:
            print(p)
        count += 1
        print("\n")
print("--------------------------------------------------------")

# Commit the changes and close the connection
conn.commit()
conn.close()