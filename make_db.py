import sqlite3

conn = sqlite3.connect('nba_stats.db',
                             detect_types=sqlite3.PARSE_DECLTYPES |
                             sqlite3.PARSE_COLNAMES)
cursor = conn.cursor()

# Create a table to store the statistics
cursor.execute('''
        CREATE TABLE IF NOT EXISTS nba_statistics (
            player_name TEXT,
            player_team TEXT,
            next_matchup TEXT,
            avg_points INTEGER,
            avg_rebounds INTEGER,
            avg_assists INTEGER,
            predicted_points INTEGER,
            ppg_over INTEGER,
            ppg_under INTEGER,
            predicted_rebounds INTEGER,
            prg_over INTEGER,
            prg_under INTEGER,
            predicted_assists INTEGER,
            pag_over INTEGER,
            pag_under INTEGER,
            game_date DATE
        )
''')

# Commit the changes and close the connection
conn.commit()
conn.close()