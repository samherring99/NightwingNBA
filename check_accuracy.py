import requests

# First we want to read the picks in to a dict in this file
url = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024/teams/20/events?lang=en&region=us"

# Make this get game data instead
def get_game_date(game_id):
    response = requests.get("http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}?lang=en&region=us".format(game_id=game_id))
    data = response.json()
    print(data['date'])

test = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/401585087/competitions/401585087/competitors/4/roster/4395651/statistics/0?lang=en&region=us"

page_index = 1
response = requests.get(url, params={"page": page_index}) # url, params={"page": page_index}
data = response.json()

page_count = data['pageCount']
for item in data['items'][:1]:
    id_num = str(item['$ref']).split("?")[0].split("/")[-1]
    print(id_num)
    get_game_date(id_num)

    #Sleep for rate limiting
