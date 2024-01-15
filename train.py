from utils.misc_utils import get_all_teams
from utils.player_stat_utils import get_all_player_stats_for_team

import sqlite3
import requests

conn = sqlite3.connect('nba_datastore.db',
                             detect_types=sqlite3.PARSE_DECLTYPES |
                             sqlite3.PARSE_COLNAMES)
cursor = conn.cursor()

roster_template = '''
SELECT player_id FROM player_stats WHERE team_id = {team_id};
'''

player_template = '''
SELECT 
blocks,
defensive_rebounds,
steals,
defensive_rebounds_per_game,
blocks_per_game,
steals_per_game,
defensive_rebounds_per_48,
blocks_per_48,
steals_per_48,
largest_lead,
disqualifications,
flagrant_fouls,
fouls,
ejections,
technical_fouls,
rebounds,
minutes,
minutes_per_game,
rating,
plus_minus,
rebounds_per_game,
fouls_per_game,
flagrant_fouls_per_game,
technical_fouls_per_game,
ejections_per_game,
disqualifications_per_game,
assist_to_turnover_ratio,
steal_to_foul_ratio,
block_to_foul_ratio,
team_rebounds_per_game,
total_technical_fouls,
steal_to_turnover_ratio,
rebounds_per_48_minutes,
fouls_per_48_minutes,
flagrant_fouls_per_48_minutes,
technical_fouls_per_48_minutes,
ejections_per_48_minutes,
disqualifications_per_48_minutes,
r40,
games_played,
games_started,
double_double,
triple_double,
assists,
field_goals,
field_goals_attempted,
field_goals_made,
field_goal_percentage,
free_throws,
free_throw_percentage,
free_throws_attempted,
free_throws_made,
offensive_rebounds,
points,
turnovers,
s_3_point_field_goal_percentage,
s_3_point_field_goals_attempted,
s_3_point_field_goals_made,
total_turnovers,
points_in_the_paint,
brick_index,
average_field_goals_made,
average_field_goals_attempted,
average_3_point_field_goals_made,
average_3_point_field_goals_attempted,
average_free_throws_made,
average_free_throws_attempted,
points_per_game,
offensive_rebounds_per_game,
assists_per_game,
turnovers_per_game,
offensive_rebound_percentage,
estimated_possessions,
estimated_possessions_per_game,
points_per_estimated_possession,
team_turnovers_per_game,
total_turnovers_per_game,
s_2_point_field_goals_made,
s_2_point_field_goals_attempted,
s_2_point_field_goals_made_per_game,
s_2_point_field_goals_attempted_per_game,
s_2_point_field_goal_percentage,
shooting_efficiency,
scoring_efficiency,
fieldgoals_made_per_48,
fieldgoals_attempted_per_48,
s_3_point_fieldgoals_made_per_48,
s_3_point_fieldgoals_attempted_per_48,
freethrows_made_per_48,
freethrows_attempted_per_48,
points_scored_per_48,
offensive_rebounds_per_48,
assists_per_48,
p40,
a40
 FROM player_stats WHERE game_id = {game_id} AND player_id = {player_id};
'''

matchup_template = '''
select t.team_id from player_stats p JOIN team_stats t on p.game_id = t.game_id WHERE p.game_id = {game_id} group by t.team_id;
'''

game_template = '''
SELECT game_name from player_stats where game_id = {game_id}
'''

team_template = '''
SELECT * FROM team_stats WHERE game_id = {game_id} and team_id = {team_id};
'''

get_player_team = '''
select team_id from player_stats where game_id = {game_id} and player_id = {player_id}
'''

get_team_previous_game = '''
select game_id from player_stats where game_date <= (select game_date from player_stats where game_id = {game_id}) and team_id = {team_id} group by game_date order by game_date limit 5;
'''

def iterate_teams(team_list, cursor):
    for team in team_list:
        team_games_endpoint = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2024/teams/{team_id}/events?lang=en&region=us".format(team_id=team)
        team_games_response = requests.get(team_games_endpoint, params={"page": 1})
        team_game_data = team_games_response.json()
        page_count = team_game_data['pageCount']

        #print(team_game_data)
        #print(team)

        for i in range(len(team_game_data['items'])-1):
            id_num = str(team_game_data['items'][i]['$ref']).split("?")[0].split("/")[-1]
            #print(id_num)
            
            cursor.execute(roster_template.format(team_id=team))
            players = cursor.fetchall()
            #print(list(players))

            for player_id in players:

                #print(player_id[0])
                
                #print(id_num)
                #print(player_id[0])

                #print(player_template.format(game_id=id_num, player_id=player_id[0]))

                cursor.execute(player_template.format(game_id=id_num, player_id=player_id[0]))
                result = cursor.fetchall()

                #print(result)

                if result:

                    print("#######################################################################")

                    next_id_num = str(team_game_data['items'][i+1]['$ref']).split("?")[0].split("/")[-1]
                    print("For game with ID " + str(next_id_num) + " and player with ID " + str(player_id[0]))

                    guess_points = result[0][53]
                    guess_assists = result[0][43]
                    guess_rebounds = result[0][15]

                    #print(str(guess_points) + " " + str(guess_assists) + " " + str(guess_rebounds))

                    previous_player_stats = result

                    cursor.execute(get_player_team.format(game_id=id_num, player_id=player_id[0]))
                    player_team_id = cursor.fetchall()

                    print("Previous Player Stats: " + str(previous_player_stats[0]))
                    print("\n")

                    cursor.execute(team_template.format(game_id=id_num, team_id=player_team_id[0][0]))
                    result = cursor.fetchall()

                    #TODO Remove team IDs and iterate
                    print("Previous Game Stats: " + str(result[0][2:]))
                    print("\n")

                    cursor.execute(matchup_template.format(game_id=next_id_num, team_id=player_team_id[0]))
                    game_team_ids = cursor.fetchall()

                    opposing_team = 0
                    for game in game_team_ids:
                        if game[0] != player_team_id[0]:
                            opposing_team = game[0]

                    cursor.execute(get_team_previous_game.format(game_id=next_id_num, team_id=opposing_team))
                    previous_opponent_game = cursor.fetchall()

                    opposing_team_previous_game_id = 0

                    for i in range(len(previous_opponent_game)):
                        #print(previous_opponent_game[i][0])
                        if str(previous_opponent_game[i][0]) == str(next_id_num) and i != 0:
                            opposing_team_previous_game_id = previous_opponent_game[i-1][0]
                        elif i == 0:
                            continue
                        else:
                            continue

                    cursor.execute(team_template.format(game_id=opposing_team_previous_game_id, team_id=opposing_team))
                    previous_opponent_matchup_stats = cursor.fetchall()
                    print("Opponent's Previous Game Stats: " + str(previous_opponent_matchup_stats[0][2:]))
                    print("\n")

                    ## 287 input values + 3 for guesses

                    print("Validation: ")

                    cursor.execute(player_template.format(game_id=next_id_num, player_id=player_id[0]))
                    result = cursor.fetchall()

                    if result:

                        hit_points = result[0][53]
                        hit_assists = result[0][43]
                        hit_rebounds = result[0][15]

                        #print(str(hit_points) + " " + str(hit_assists) + " " + str(hit_rebounds))

                        print("Points: 0.0" if hit_points <= guess_points else "Points: 1.0")
                        print("Assists: 0.0" if hit_assists <= guess_assists else "Assists: 1.0")
                        print("Rebounds: 0.0" if hit_rebounds <= guess_rebounds else "Rebounds: 1.0")
                        print("\n")

team_list = get_all_teams()
#print(team_list)
iterate_teams(team_list, cursor)

conn.close()
'''
        for i in range(page_count-1):
            team_games_response = requests.get(team_games_endpoint, params={"page": int(i+2)})
            team_game_data = team_games_response.json()
            for i in team_game_data['items'][1:]:
                id_num = str(i['$ref']).split("?")[0].split("/")[-1]
                '''