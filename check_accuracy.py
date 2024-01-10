import requests

game_id_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/teams/"
game_page_response = requests.get(game_id_endpoint)
game_page = game_page_response.json()
# url, params={"page": page_index}
print(game_page['pageCount'])
print(game_page['items'][0]['$ref'])
print("ID: " + str(game_page['items'][0]['$ref']).split("?")[0][-1])

# TODO page counts iteration

# Using each ID, we want to get the games for each team ID using the below URL

# First we want to read the picks in to a dict in this file
url = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024/teams/20/events?lang=en&region=us"

# Make this get game data instead
def get_game_date(game_id):
    response = requests.get("http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}?lang=en&region=us".format(game_id=game_id))
    data = response.json()
    print(data['date'])

#test = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/401585087/competitions/401585087/competitors/4/roster/4395651/statistics/0?lang=en&region=us"

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
    get_game_date(id_num)

    #Sleep for rate limiting

# Then validation pieces
