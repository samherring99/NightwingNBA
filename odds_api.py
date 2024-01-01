import requests
import sqlite3
from datetime import datetime

# Insert your Odds API key here
API_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

# API constants
SPORT = 'basketball_nba'
REGIONS = 'us'
MARKETS = 'h2h,spreads'
ODDS_FORMAT = 'american'
DATE_FORMAT = 'iso'

# GET request using the above parameters to get NBA odds for the next ~1 week
# TODO: Make this just for the day. Right now it doesnt make sense to run this daily
odds_response = requests.get(f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds', params={
    'api_key': API_KEY,
    'regions': REGIONS,
    'markets': MARKETS,
    'oddsFormat': ODDS_FORMAT,
    'dateFormat': DATE_FORMAT,
    'bookmakers': 'fanduel'
})

def get_player_team(player, matchup, sql):
    sql.execute('''SELECT player_team FROM nba_statistics WHERE player_name = "{player}"'''.format(player=player))
    team = sql.fetchall()
    name = str(team[0][0]).split(" ")[-1]

    if name != matchup[0].split(" ")[-1]:
        return matchup[0]
    else:
        return matchup[1]

# This method inserts the Odds information into the previously created nba_stats.db database
def insert_odds(score_type, outcome, sql, date, opposing_team):
    value = outcome['price']
    data = (outcome['point'], value, date, opposing_team)
    odds_type=outcome['name'].lower()
    identifier=score_type[0]
    sql.execute('''
            UPDATE nba_statistics SET predicted_{score} = ?, p{id}g_{odds} = ?, game_date = ?, next_matchup = ? WHERE player_name = ?
            '''.format(score=score_type, id=identifier, odds=odds_type), (*data, outcome['description']))


# Main execution of this file

if odds_response.status_code == 200:

    print("Success! Inserting odds into the database...")

    # Convert the games API call response to JSON
    odds_json = odds_response.json()

    # Connect to the NBA stats database
    conn = sqlite3.connect('nba_stats.db',
                             detect_types=sqlite3.PARSE_DECLTYPES |
                             sqlite3.PARSE_COLNAMES)
    cursor = conn.cursor()

    # Iterate over the games returned from the Odds API call
    for game in odds_json:

        # Make a call to the API to get the odds for a specific game
        game_odds_response = requests.get("https://api.the-odds-api.com/v4/sports/{sport}/events/{eventId}/odds?".format(sport=SPORT, eventId=game['id']), params={
            'api_key': API_KEY,
            'regions': REGIONS,
            'markets': 'player_points,player_rebounds,player_assists',
            'oddsFormat': 'american',
            'dateFormat': DATE_FORMAT,
            'bookmakers': 'draftkings,fanduel'
        })

        # Convert the response to JSON
        odds_game_json = game_odds_response.json()

        game_time = datetime.strptime(str(odds_game_json['commence_time'][:10]), '%Y-%m-%d')

        # If we have entries from bookmakers
        if odds_game_json['bookmakers'] != []:

            # For 'player prop' in the returned odds JSON for a given game
            for prop in odds_game_json['bookmakers'][0]['markets']: # TODO: need to make this confirmed fanduel, can be DK as of now
                for outcome in prop['outcomes']:
                    insert_odds(prop['key'][7:], outcome, cursor, game_time, get_player_team(outcome['description'], [game['home_team'], game['away_team']], cursor))
                    
    conn.commit()
    conn.close()

else:
    print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

print("Finished!")
print('Remaining requests', odds_response.headers['x-requests-remaining'])
print('Used requests', odds_response.headers['x-requests-used'])
