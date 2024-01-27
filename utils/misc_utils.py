import requests

ENDPOINT="http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/teams/"

# This method wraps the team IDs from the endpoint in an array and returns it
def add_team_to_list(data):
    temp_list = []
    for i in range(len(data['items'])):
        id_num = str(data['items'][i]['$ref']).split("?")[0].split("/")[-1]
        temp_list.append(int(id_num))

    return temp_list

# This method can be called from any of the main files and returns a list of all team IDs
def get_all_teams():

    game_page_response = requests.get(ENDPOINT)
    game_page = game_page_response.json()

    id_list = []

    page_count = game_page['pageCount']

    id_list += add_team_to_list(game_page)

    for i in range(page_count-1):
        game_page_response = requests.get(ENDPOINT, params={"page": int(i+2)})
        game_page = game_page_response.json()
        id_list += add_team_to_list(game_page)

    return id_list