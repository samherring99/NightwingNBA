import requests
import sqlite3

conn = sqlite3.connect('nba_stats.db')
cursor = conn.cursor()

# Comment all of the below to explain each loop, add print statements explainings whats going on

def create_nba_database(): 
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
    response = requests.get(url)
    data = response.json()

    for team in data["sports"][0]["leagues"][0]["teams"]:
        print(team["team"]["id"])
        print(team["team"]["displayName"])

        url2 = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{id}/roster".format(id=team["team"]["id"])
        response2 = requests.get(url2)
        data2 = response2.json()

        print("Getting player stats")

        for athlete in data2["athletes"]:
            print(athlete["id"])
            print(athlete["fullName"])

            url3 = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/athletes/{id}/statistics".format(id=athlete["id"])
            response3 = requests.get(url3)
            data3 = response3.json()

            avg_points = 0
            avg_rebounds = 0
            avg_assists = 0

            if response3.status_code != 404:
                for stattype in data3["splits"]["categories"]: #General, Defensive, OffensIve
                    for stat in stattype["stats"]:
                        #print(stat["name"])
                        if stat["name"] in ["avgPoints", "avgAssists", "avgRebounds"]:
                            print(stat["displayName"] + " " + str(stat["value"]))

                    avg_points_obj = next((item for item in stattype["stats"] if item["name"] == "avgPoints"), None)
                    avg_rebounds_obj = next((item for item in stattype["stats"] if item["name"] == "avgRebounds"), None)
                    avg_assists_obj = next((item for item in stattype["stats"] if item["name"] == "avgAssists"), None)

                    if avg_assists_obj is not None:
                        avg_points = avg_points_obj["displayValue"]
                    
                    if avg_rebounds_obj is not None:
                        avg_rebounds = avg_rebounds_obj["displayValue"]

                    if avg_assists_obj is not None:
                        avg_assists = avg_assists_obj["displayValue"]

                player_data = (athlete["fullName"], str(avg_points), str(avg_rebounds), str(avg_assists), '0', '0', '0', '0', '0', '0', '0', '0', '0')

                print(player_data)

                cursor.execute('''
                    INSERT INTO nba_statistics (player_name, avg_points, avg_rebounds, avg_assists, predicted_points, ppg_over, ppg_under, predicted_rebounds, prg_over, prg_under, predicted_assists, pag_over, pag_under)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', player_data)

create_nba_database()
conn.commit()
conn.close()