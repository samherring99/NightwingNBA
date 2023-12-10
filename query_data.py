import sqlite3
import random

conn = sqlite3.connect('nba_stats.db')
cursor = conn.cursor()

types = ["rebounds", "assists", "points"]
bet = "over"
symbol = ["<", ">"]

total_picks = []

for i in range(10):
    risk = random.random()

    query_string = ""
    mark = "ppg"
    symbol_choice = "<"
    tail = "0"

    score_type = random.choice(types)
    bet_type = "over"

    if score_type == 'points':
        mark = "ppg"
    elif score_type == 'rebounds':
        mark = "prg"
    else:
        mark="pag"

    if risk < 0.2:
        # Greater than 0 less than 120
        symbol_choice = symbol[1]
        tail = "0 AND {mark}_{bet} < 120".format(mark=mark, bet=bet_type)
        
    else:
        symbol_choice = symbol[0]
        tail = "0"

    query_string = """
    
        SELECT player_name, avg_{type}, predicted_{type}, {mark}_{bet} FROM nba_statistics
        WHERE avg_{type} > predicted_{type} AND {mark}_{bet} {symbol} {tail};

    """.format(type=score_type, mark=mark, bet=bet_type, symbol=symbol_choice, tail=tail)

    #print(query_string)


# Create a table to store the statistics
    #print(score_type + " " + bet_type)
    cursor.execute(query_string)
    result = cursor.fetchall()
    #result.append(score_type + "-" + bet_type)
    #print(result)
    for pick in result:
        bet_string = score_type + "-" + bet_type
        pick = pick + tuple([bet_string])
        total_picks.append(pick)

def myFunc(e):
  return e[3]

total_picks.sort(key=myFunc, reverse=True)

#print(total_picks)
print("Total picks: " + str(len(total_picks)))
num_bets = int(len(total_picks) / 5)
print("# of parlays to create: " + str(num_bets))

total_picks = list(set(total_picks))

random.shuffle(total_picks)

def break_out_bets():
    for i in range(0, len(total_picks), 5):  
        yield total_picks[i:i + 5]

parlays = list(break_out_bets())

#print(parlays)

print("\n")

parlay_list = []

def check_valid(bet):
    if str(bet[4]).split("-")[0] == "points":
        if int(bet[2]) % 5 == 0:
            return True
    elif str(bet[4]).split("-")[0] == "rebounds":
        if int(bet[2]) in [6, 8, 10]:
            return True
    elif str(bet[4]).split("-")[0] == "assists":
        if int(bet[2]) in [4, 6, 8 ,10]:
            return True
    else:
        return False
    
    return False

for parlay in parlays:
    p = []
    for bet in list(parlay):
        if bet[2] > 6:
            p.append(str(bet[0]) + " " + str(bet[4]).split("-")[0] + " " + str(bet[4]).split("-")[1] + " " + str(bet[2]) + " " + str(bet[3]))
    parlay_list.append(p)

count = 1

def parlength(parlay):
    return len(parlay)

parlay_list.sort(key=parlength)

for parlay in parlay_list:
    if len(parlay) >= 2:
        print("PARLAY #{count} --------------------------------------------------".format(count=count))
        for p in parlay:
            print(p)
        count += 1
        print("\n")
print("--------------------------------------------------------")

    #for bet in parlay:
    #    if bet[3] > 0:
    #        print(parlay)

# Commit the changes and close the connection
conn.commit()
conn.close()