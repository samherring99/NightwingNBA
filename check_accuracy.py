import requests
import time

# TODO clean up the below code

def get_all_teams():

    game_id_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/teams/"
    game_page_response = requests.get(game_id_endpoint)
    game_page = game_page_response.json()

    id_list = []

    page_count = game_page['pageCount']
    for i in range(len(game_page['items'])):
        id_num = str(game_page['items'][i]['$ref']).split("?")[0].split("/")[-1]
        id_list.append(int(id_num))

    for i in range(page_count-1):
        game_page_response = requests.get(game_id_endpoint, params={"page": int(i+2)})
        game_page = game_page_response.json()
        for i in range(len(game_page['items'])):
            id_num = str(game_page['items'][i]['$ref']).split("?")[0].split("/")[-1]
            id_list.append(int(id_num))

    return id_list

    
#print(id_list)

# Given 1/09/24 - Scottie Barnes - assists over 4.5 - check if accurate
# Do by PARLAY

# Maybe just game ID, then do both teams?? - CLEAN ALL THIS UP

def get_team_stats(game_id, team_id):
    team_stats_endpoint = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}/competitions/{game_id}/competitors/{team_id}/statistics".format(game_id=game_id, team_id=team_id)
    team_stats_response = requests.get(team_stats_endpoint)
    team_stats_data = team_stats_response.json()

    print("TEAM: " + str(team_id))

    for category in team_stats_data['splits']['categories']:
        for stat in category['stats']:
            print(stat['displayName'] + " " + str(stat['value']))

def get_game_stats(game_id):
    teams_stats_endpoint = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}/competitions/{game_id}/competitors/".format(game_id=game_id) # {team_id}/statistics
    teams_stats_response = requests.get(teams_stats_endpoint)
    teams_stats_data = teams_stats_response.json()

    #print(teams_stats_data)

    for competitor in teams_stats_data['items']:
        #print(competitor['id'])
        get_team_stats(game_id, competitor['id'])

    # Do more here

def get_player_stats(game_id, team_id, player_id):
    player_stats_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}/competitions/{game_id}/competitors/{team_id}/roster/{player_id}/statistics/0?lang=en&region=us".format(game_id=game_id, team_id=team_id, player_id=player_id)
    player_stats_response = requests.get(player_stats_endpoint)
    player_stats_data = player_stats_response.json()

    for category in player_stats_data['splits']['categories']:
        for stat in category['stats']: # Maybe only do the stat we are looking for
            print(stat['displayName'] + " " + str(stat['value']))

def get_team_scores(game_id, team_id):
    team_boxscore_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}/competitions/{game_id}/competitors/{team_id}/roster?lang=en&region=us".format(game_id=game_id, team_id=team_id)
    team_boxscore_response = requests.get(team_boxscore_endpoint)
    team_boxscore_data = team_boxscore_response.json()

    for player in team_boxscore_data['entries'][:1]:
        print(player['playerId'])
        print(player['displayName']) # Maybe only do the player we are looking for
        if not player['didNotPlay']:
            get_player_stats(game_id, team_id, player['playerId'])

def get_game_data(game_id):
    # Using each ID, we want to get the games for each team ID using the below URL
    response = requests.get("http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}?lang=en&region=us".format(game_id=game_id))
    data = response.json()
    print(data['name'])
    print(data['date']) # Maybe only find the date we are looking for
    for team in data['competitions'][0]['competitors'][:1]:
        print(team['id'])
        print(team['homeAway']) # Maybe only do the team we are looking for
        get_team_scores(game_id, team['id'])

# First we want to read the picks in to a dict in this file
#id_list = get_all_teams()

def iterate_teams(team_list):

    for id_num in team_list[:1]:
        url = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024/teams/{team_id}/events?lang=en&region=us".format(team_id=id_num)

        response = requests.get(url, params={"page": 1})
        data = response.json()
        page_count = data['pageCount']
        print(page_count)
        for game in data['items'][:1]:
            # If game is not the one we are looking for

            id_num = str(game['$ref']).split("?")[0].split("/")[-1]
            print(id_num)
            get_game_data(id_num)
            time.sleep(2)

get_game_stats(401585087)

        # else get team and player scores

    #for i in range(page_count - 1):
    #    response = requests.get(url, params={"page": i+2})
    #    data = response.json()
    #    id_num = str(game['$ref']).split("?")[0].split("/")[-1]
    #    print(id_num)
    #    get_game_data(id_num)
    #    time.sleep(2)

#test = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/401585087/competitions/401585087/competitors/4/roster/4395651/statistics/0?lang=en&region=us"
'''

player_stats = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/401591883/competitions/401591883/competitors/1/roster/4277905/statistics/0?"
page_index = 1
response = requests.get(url, params={"page": page_index}) # url, params={"page": page_index}
data = response.json()

# TODO page counts iteration with the above, return more data than just date

# Iterate over the games for each team here, and get the ID number of each game
# Use the method call to return game data

page_count = data['pageCount']
for item in data['items'][:1]:
    id_num = str(item['$ref']).split("?")[0].split("/")[-1]
    print(id_num)
    get_game_data(id_num)

    #Sleep for rate limiting

# Then validation pieces

# Do we want to put everything into a database and then query it? Not sure best way to organize all this info 

'''