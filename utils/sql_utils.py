import sqlite3

def create_database(cursor, conn):

    # Create a table to store the statistics - TODO move this string into a new file?
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS player_stats (
            game_name TEXT,
            game_id INTEGER,
            game_date DATE,
            team_name TEXT,
            team_id INTEGER,
            player_name TEXT,
            player_id INTEGER,
            blocks FLOAT,
            defensive_rebounds FLOAT,
            steals FLOAT,
            defensive_rebounds_per_game FLOAT,
            blocks_per_game FLOAT,
            steals_per_game FLOAT,
            defensive_rebounds_per_48 FLOAT,
            blocks_per_48 FLOAT,
            steals_per_48 FLOAT,
            largest_lead FLOAT,
            disqualifications FLOAT,
            flagrant_fouls FLOAT,
            fouls FLOAT,
            ejections FLOAT,
            technical_fouls FLOAT,
            rebounds FLOAT,
            minutes FLOAT,
            minutes_per_game FLOAT,
            rating FLOAT,
            plus_minus FLOAT,
            rebounds_per_game FLOAT,
            fouls_per_game FLOAT,
            flagrant_fouls_per_game FLOAT,
            technical_fouls_per_game FLOAT,
            ejections_per_game FLOAT,
            disqualifications_per_game FLOAT,
            assist_to_turnover_ratio FLOAT,
            steal_to_foul_ratio FLOAT,
            block_to_foul_ratio FLOAT,
            team_rebounds_per_game FLOAT,
            total_technical_fouls FLOAT,
            steal_to_turnover_ratio FLOAT,
            rebounds_per_48_minutes FLOAT,
            fouls_per_48_minutes FLOAT,
            flagrant_fouls_per_48_minutes FLOAT,
            technical_fouls_per_48_minutes FLOAT,
            ejections_per_48_minutes FLOAT,
            disqualifications_per_48_minutes FLOAT,
            r40 FLOAT,
            games_played FLOAT,
            games_started FLOAT,
            double_double FLOAT,
            triple_double FLOAT,
            assists FLOAT,
            field_goals FLOAT,
            field_goals_attempted FLOAT,
            field_goals_made FLOAT,
            field_goal_percentage FLOAT,
            free_throws FLOAT,
            free_throw_percentage FLOAT,
            free_throws_attempted FLOAT,
            free_throws_made FLOAT,
            offensive_rebounds FLOAT,
            points FLOAT,
            turnovers FLOAT,
            s_3_point_field_goal_percentage FLOAT,
            s_3_point_field_goals_attempted FLOAT,
            s_3_point_field_goals_made FLOAT,
            total_turnovers FLOAT,
            points_in_the_paint FLOAT,
            brick_index FLOAT,
            average_field_goals_made FLOAT,
            average_field_goals_attempted FLOAT,
            average_3_point_field_goals_made FLOAT,
            average_3_point_field_goals_attempted FLOAT,
            average_free_throws_made FLOAT,
            average_free_throws_attempted FLOAT,
            points_per_game FLOAT,
            offensive_rebounds_per_game FLOAT,
            assists_per_game FLOAT,
            turnovers_per_game FLOAT,
            offensive_rebound_percentage FLOAT,
            estimated_possessions FLOAT,
            estimated_possessions_per_game FLOAT,
            points_per_estimated_possession FLOAT,
            team_turnovers_per_game FLOAT,
            total_turnovers_per_game FLOAT,
            s_2_point_field_goals_made FLOAT,
            s_2_point_field_goals_attempted FLOAT,
            s_2_point_field_goals_made_per_game FLOAT,
            s_2_point_field_goals_attempted_per_game FLOAT,
            s_2_point_field_goal_percentage FLOAT,
            shooting_efficiency FLOAT,
            scoring_efficiency FLOAT,
            fieldgoals_made_per_48 FLOAT,
            fieldgoals_attempted_per_48 FLOAT,
            s_3_point_fieldgoals_made_per_48 FLOAT,
            s_3_point_fieldgoals_attempted_per_48 FLOAT,
            freethrows_made_per_48 FLOAT,
            freethrows_attempted_per_48 FLOAT,
            points_scored_per_48 FLOAT,
            offensive_rebounds_per_48 FLOAT,
            assists_per_48 FLOAT,
            p40 FLOAT,
            a40 FLOAT
        )
    ''')
    # Same here
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_stats (
            game_id INTEGER,
            team_id INTEGER,
            blocks FLOAT,
            defensive_rebounds FLOAT,
            steals FLOAT,
            points_off_turnovers FLOAT,
            defensive_rebounds_per_game FLOAT,
            blocks_per_game FLOAT,
            steals_per_game FLOAT,
            defensive_rebounds_per_48 FLOAT,
            blocks_per_48 FLOAT,
            steals_per_48 FLOAT,
            largest_lead FLOAT,
            disqualifications FLOAT,
            flagrant_fouls FLOAT,
            fouls FLOAT,
            ejections FLOAT,
            technical_fouls FLOAT,
            rebounds FLOAT,
            value_over_replacement_player FLOAT,
            minutes_per_game FLOAT,
            rating FLOAT,
            rebounds_per_game FLOAT,
            fouls_per_game FLOAT,
            flagrant_fouls_per_game FLOAT,
            technical_fouls_per_game FLOAT,
            ejections_per_game FLOAT,
            disqualifications_per_game FLOAT,
            assist_to_turnover_ratio FLOAT,
            steal_to_foul_ratio FLOAT,
            block_to_foul_ratio FLOAT,
            team_rebounds_per_game FLOAT,
            total_technical_fouls FLOAT,
            steal_to_turnover_ratio FLOAT,
            rebounds_per_48_minutes FLOAT,
            fouls_per_48_minutes FLOAT,
            flagrant_fouls_per_48_minutes FLOAT,
            technical_fouls_per_48_minutes FLOAT,
            ejections_per_48_minutes FLOAT,
            disqualifications_per_48_minutes FLOAT,
            games_played FLOAT,
            games_started FLOAT,
            double_double FLOAT,
            triple_double FLOAT,
            assists FLOAT,
            field_goals FLOAT,
            field_goals_attempted FLOAT,
            field_goals_made FLOAT,
            field_goal_percentage FLOAT,
            free_throws FLOAT,
            free_throw_percentage FLOAT,
            free_throws_attempted FLOAT,
            free_throws_made FLOAT,
            offensive_rebounds FLOAT,
            points FLOAT,
            turnovers FLOAT,
            s_3_point_field_goal_percentage FLOAT,
            s_3_point_field_goals_attempted FLOAT,
            s_3_point_field_goals_made FLOAT,
            team_turnovers FLOAT,
            total_turnovers FLOAT,
            points_in_the_paint FLOAT,
            brick_index FLOAT,
            fast_break_points FLOAT,
            average_field_goals_made FLOAT,
            average_field_goals_attempted FLOAT,
            average_3_point_field_goals_made FLOAT,
            average_3_point_field_goals_attempted FLOAT,
            average_free_throws_made FLOAT,
            average_free_throws_attempted FLOAT,
            points_per_game FLOAT,
            offensive_rebounds_per_game FLOAT,
            assists_per_game FLOAT,
            turnovers_per_game FLOAT,
            offensive_rebound_percentage FLOAT,
            estimated_possessions FLOAT,
            estimated_possessions_per_game FLOAT,
            points_per_estimated_possession FLOAT,
            team_turnovers_per_game FLOAT,
            total_turnovers_per_game FLOAT,
            s_2_point_field_goals_made FLOAT,
            s_2_point_field_goals_attempted FLOAT,
            s_2_point_field_goals_made_per_game FLOAT,
            s_2_point_field_goals_attempted_per_game FLOAT,
            s_2_point_field_goal_percentage FLOAT,
            shooting_efficiency FLOAT,
            scoring_efficiency FLOAT,
            fieldgoals_made_per_48 FLOAT,
            fieldgoals_attempted_per_48 FLOAT,
            s_3_point_fieldgoals_made_per_48 FLOAT,
            s_3_point_fieldgoals_attempted_per_48 FLOAT,
            freethrows_made_per_48 FLOAT,
            freethrows_attempted_per_48 FLOAT,
            points_scored_per_48 FLOAT,
            offensive_rebounds_per_48 FLOAT,
            assists_per_48 FLOAT
        )
    ''')

    # Commit the changes
    conn.commit()

# Given a database entry, write it to the player stats database if it fits the format using the cursor
def write_entry_to_players_db(entry, cursor):
    row = []
    for key in entry.keys():
        row.append(str(entry[key]))

    if len(row) == 102:
        cursor.execute('''
            INSERT INTO player_stats (game_name, game_id, game_date, team_name, team_id, player_name, player_id, blocks, defensive_rebounds, steals, defensive_rebounds_per_game, blocks_per_game, steals_per_game, defensive_rebounds_per_48, blocks_per_48, steals_per_48, largest_lead, disqualifications, flagrant_fouls, fouls, ejections, technical_fouls, rebounds, minutes, minutes_per_game, rating, plus_minus, rebounds_per_game, fouls_per_game, flagrant_fouls_per_game, technical_fouls_per_game, ejections_per_game, disqualifications_per_game, assist_to_turnover_ratio, steal_to_foul_ratio, block_to_foul_ratio, team_rebounds_per_game, total_technical_fouls, steal_to_turnover_ratio, rebounds_per_48_minutes, fouls_per_48_minutes, flagrant_fouls_per_48_minutes, technical_fouls_per_48_minutes, ejections_per_48_minutes, disqualifications_per_48_minutes, r40, games_played, games_started, double_double, triple_double, assists, field_goals, field_goals_attempted, field_goals_made, field_goal_percentage, free_throws, free_throw_percentage, free_throws_attempted, free_throws_made, offensive_rebounds, points, turnovers, s_3_point_field_goal_percentage, s_3_point_field_goals_attempted, s_3_point_field_goals_made, total_turnovers, points_in_the_paint, brick_index, average_field_goals_made, average_field_goals_attempted, average_3_point_field_goals_made, average_3_point_field_goals_attempted, average_free_throws_made, average_free_throws_attempted, points_per_game, offensive_rebounds_per_game, assists_per_game, turnovers_per_game, offensive_rebound_percentage, estimated_possessions, estimated_possessions_per_game, points_per_estimated_possession, team_turnovers_per_game, total_turnovers_per_game, s_2_point_field_goals_made, s_2_point_field_goals_attempted, s_2_point_field_goals_made_per_game, s_2_point_field_goals_attempted_per_game, s_2_point_field_goal_percentage, shooting_efficiency, scoring_efficiency, fieldgoals_made_per_48, fieldgoals_attempted_per_48, s_3_point_fieldgoals_made_per_48, s_3_point_fieldgoals_attempted_per_48, freethrows_made_per_48, freethrows_attempted_per_48, points_scored_per_48, offensive_rebounds_per_48, assists_per_48, p40, a40)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', tuple(row))
    else:
        print("Skipping this game")

# Given a database entry, write it to the team stats database if it fits the format using the cursor
def write_entry_to_team_db(entry, cursor):
    row = []
    for key in entry.keys():
        row.append(str(entry[key]))

    if len(row) == 96:
        cursor.execute('''
            INSERT INTO team_stats (game_id, team_id, blocks, defensive_rebounds, steals, points_off_turnovers, defensive_rebounds_per_game, blocks_per_game, steals_per_game, defensive_rebounds_per_48, blocks_per_48, steals_per_48, largest_lead, disqualifications, flagrant_fouls, fouls, ejections, technical_fouls, rebounds, value_over_replacement_player, minutes_per_game, rating, rebounds_per_game, fouls_per_game, flagrant_fouls_per_game, technical_fouls_per_game, ejections_per_game, disqualifications_per_game, assist_to_turnover_ratio, steal_to_foul_ratio, block_to_foul_ratio, team_rebounds_per_game, total_technical_fouls, steal_to_turnover_ratio, rebounds_per_48_minutes, fouls_per_48_minutes, flagrant_fouls_per_48_minutes, technical_fouls_per_48_minutes, ejections_per_48_minutes, disqualifications_per_48_minutes, games_played, games_started, double_double, triple_double, assists, field_goals, field_goals_attempted, field_goals_made, field_goal_percentage, free_throws, free_throw_percentage, free_throws_attempted, free_throws_made, offensive_rebounds, points, turnovers, s_3_point_field_goal_percentage, s_3_point_field_goals_attempted, s_3_point_field_goals_made, team_turnovers, total_turnovers, points_in_the_paint, brick_index, fast_break_points, average_field_goals_made, average_field_goals_attempted, average_3_point_field_goals_made, average_3_point_field_goals_attempted, average_free_throws_made, average_free_throws_attempted, points_per_game, offensive_rebounds_per_game, assists_per_game, turnovers_per_game, offensive_rebound_percentage, estimated_possessions, estimated_possessions_per_game, points_per_estimated_possession, team_turnovers_per_game, total_turnovers_per_game, s_2_point_field_goals_made, s_2_point_field_goals_attempted, s_2_point_field_goals_made_per_game, s_2_point_field_goals_attempted_per_game, s_2_point_field_goal_percentage, shooting_efficiency, scoring_efficiency, fieldgoals_made_per_48, fieldgoals_attempted_per_48, s_3_point_fieldgoals_made_per_48, s_3_point_fieldgoals_attempted_per_48, freethrows_made_per_48, freethrows_attempted_per_48, points_scored_per_48, offensive_rebounds_per_48, assists_per_48)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', tuple(row))
    else:
        print("Skipping this game")

# TODO maybe split the below code into smaller methods

# Top level method to write player stats data to the player stats database
# Player stats is in the form of the JSON response from the player data API endpoint
# Returns the game date
def write_players_data_to_db(player_stats, cursor):
    game_name = player_stats['game_name']
    game_id = player_stats['game_id']
    game_date = player_stats['date']

    team_names = game_name.split(" at ")

    for team in team_names:
        team_name = team.split(" ")[-1]
        if team_name in player_stats:
            team_player_stats = player_stats[team_name]
            if team_player_stats:
                team_id = team_player_stats['team_id']
                for key in team_player_stats.keys():
                    if key != 'team_id' and 'player_id' in team_player_stats[key]:
                        player_name = key
                        player_id = team_player_stats[key]['player_id']

                        entry = {'game_name' : game_name, 'game_id' : game_id, 'game_date' : game_date, 'team_name' : team_name, 'team_id' : team_id, 'player_name' : player_name, 'player_id' : player_id}

                        for stat in team_player_stats[key]:
                            entry['player_id'] = player_id
                            if stat != 'player_id':
                                table_stat_pieces = stat.split(" ")
                                table_stat = ""
                                for piece in table_stat_pieces:
                                    table_stat = table_stat + piece.lower() + "_"
                                table_stat = table_stat[0:len(table_stat)-1].replace('-','_')
                                table_stat = table_stat.replace('/','')

                                if table_stat[0].isnumeric():
                                    table_stat = "s_" + table_stat

                                entry[table_stat] = team_player_stats[key][stat]
                        write_entry_to_players_db(entry, cursor)

    return game_date

# Top level method to write team stats data to the team stats database
# team_stats is in the form of the JSON response from the team data API endpoint
def write_team_data_to_db(team_stats, cursor):
    game_id = team_stats['game_id']

    for key in team_stats.keys():
        if key != 'game_id':
            team_id = key

            print("Writing game stats for game " + str(game_id) + " and team " + str(team_id))

            entry = {'game_id' : game_id, 'team_id' : team_id}

            for stat in team_stats[team_id].keys():
                if stat != 'team_id':
                    table_stat_pieces = stat.split(" ")
                    table_stat = ""
                    for piece in table_stat_pieces:
                        table_stat = table_stat + piece.lower() + "_"
                    table_stat = table_stat[0:len(table_stat)-1].replace('-','_')
                    table_stat = table_stat.replace('/','')

                    if table_stat[0].isnumeric():
                        table_stat = "s_" + table_stat

                    entry[table_stat] = team_stats[team_id][stat]
            write_entry_to_team_db(entry, cursor)