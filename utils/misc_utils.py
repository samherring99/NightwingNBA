import requests

def get_all_teams():

    game_id_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/teams/"
    game_page_response = requests.get(game_id_endpoint)
    game_page = game_page_response.json()

    id_list = []

    # TODO optimize the below code

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