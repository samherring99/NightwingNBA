import requests
import sqlite3

API_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

SPORT = 'basketball_nba' # use the sport_key from the /sports endpoint below, or use 'upcoming' to see the next 8 games across all sports

REGIONS = 'us' # uk | us | eu | au. Multiple can be specified if comma delimited

MARKETS = 'h2h,spreads' # h2h | spreads | totals. Multiple can be specified if comma delimited

ODDS_FORMAT = 'american' # decimal | american

DATE_FORMAT = 'iso' # iso | unix

odds_response = requests.get(f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds', params={
    'api_key': API_KEY,
    'regions': REGIONS,
    'markets': MARKETS,
    'oddsFormat': ODDS_FORMAT,
    'dateFormat': DATE_FORMAT,
    'bookmakers': 'fanduel'
})

if odds_response.status_code != 200:
    print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

else:
    odds_json = odds_response.json()
    #print('Number of events:', len(odds_json))
    #print(odds_json)

    #print(odds_json)

    conn = sqlite3.connect('./autobet/nba_stats.db')
    cursor = conn.cursor()

    for game in odds_json:
        #print(game['id'])

        odds_response2 = requests.get("https://api.the-odds-api.com/v4/sports/{sport}/events/{eventId}/odds?".format(sport=SPORT, eventId=game['id']), params={
            'api_key': API_KEY,
            'regions': REGIONS,
            'markets': 'player_points,player_rebounds,player_assists',
            'oddsFormat': 'american',
            'dateFormat': DATE_FORMAT,
            'bookmakers': 'draftkings,fanduel'
        })

        odds_2_json = odds_response2.json()

        print(odds_2_json)

        if odds_2_json['bookmakers'] != []:

            # All of the above is fine to leave

            for prop in odds_2_json['bookmakers'][0]['markets']: # Might want to test this
                print(prop['key']) # Edit this
                for outcome in prop['outcomes']:
                    print(outcome) # Edit these print statements

                    # Here is where we need to optimize!!!

                    if prop['key'] == 'player_points':

                        # All of the below should be a method - set_odds - pass key in

                        ppg_over = 0
                        ppg_under = 0

                        if outcome['name'] == 'Over':

                            # All of the below should be another method - pass key and name

                            ppg_over = outcome['price']

                            ppg_data = (outcome['point'], ppg_over)

                            cursor.execute('''
                                UPDATE nba_statistics SET predicted_points = ?, ppg_over = ? WHERE player_name = ?
                                ''', (*ppg_data, outcome['description'])) # String replace column name - switch

                        if outcome['name'] == 'Under':
                            ppg_under = outcome['price']

                            ppg_data = (outcome['point'], ppg_under)

                            cursor.execute('''
                                UPDATE nba_statistics SET predicted_points = ?, ppg_under = ? WHERE player_name = ?
                                ''', (*ppg_data, outcome['description']))

                    elif prop['key'] == 'player_rebounds':

                        prg_over = 0
                        prg_under = 0

                        if outcome['name'] == 'Over':
                            prg_over = outcome['price']

                            prg_data = (outcome['point'], prg_over)

                            cursor.execute('''
                                UPDATE nba_statistics SET predicted_rebounds = ?, prg_over = ? WHERE player_name = ?
                                ''', (*prg_data, outcome['description']))

                        if outcome['name'] == 'Under':
                            prg_under = outcome['price']

                            prg_data = (outcome['point'], prg_under)

                            cursor.execute('''
                                UPDATE nba_statistics SET predicted_rebounds = ?, prg_under = ? WHERE player_name = ?
                                ''', (*prg_data, outcome['description']))

                    else:

                        pag_over = 0
                        pag_under = 0

                        if outcome['name'] == 'Over':
                            pag_over = outcome['price']

                            pag_data = (outcome['point'], pag_over)

                            cursor.execute('''
                                UPDATE nba_statistics SET predicted_assists = ?, pag_over = ? WHERE player_name = ?
                                ''', (*pag_data, outcome['description']))

                        if outcome['name'] == 'Under':
                            pag_under = outcome['price']

                            pag_data = (outcome['point'], pag_under)

                            cursor.execute('''
                                UPDATE nba_statistics SET predicted_assists = ?, pag_under = ? WHERE player_name = ?
                                ''', (*pag_data, outcome['description']))
                    
    conn.commit()
    conn.close()
            #for market in game['bookmakers'][0]['markets']:
            #    print(market['outcomes'])
            #    print(market)

        # Check the usage quota

print('Remaining requests', odds_response.headers['x-requests-remaining'])
print('Used requests', odds_response.headers['x-requests-used'])
