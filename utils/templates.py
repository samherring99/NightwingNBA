# Template to get the list of players in 
roster_template = '''
SELECT player_id FROM player_stats WHERE team_id = {team_id} GROUP BY player_id;
'''

# Player stats template minus the DATE object to avoid string casting issues
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

# Get team IDs in a matchup from the database given the game ID
matchup_template = '''
select t.team_id from player_stats p JOIN team_stats t on p.game_id = t.game_id WHERE p.game_id = {game_id} group by t.team_id;
'''

# Get the name of the game for the matchup given the game ID
game_template = '''
SELECT game_name from player_stats where game_id = {game_id}
'''

# Get the team stats given a game ID and a team ID
team_template = '''
SELECT * FROM team_stats WHERE game_id = {game_id} and team_id = {team_id}
'''

# Get the team ID for a player given their player ID
get_player_team = '''
select team_id from player_stats where player_id = {player_id} group by team_id
'''

# Get the list of game IDs sorted by date for a given team ID, where the game date is before the date
# of a given game ID
get_team_previous_game = '''
select game_id from player_stats where game_date <= (select game_date from player_stats where game_id = {game_id}) and team_id = {team_id} group by game_date order by game_date;
'''