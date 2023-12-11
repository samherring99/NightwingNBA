### NightwingBets ###

This repository contains code for the NightwingBets NBA player prop parlay generation tool. This tool currently uses NO machine learning, just simple statistics rules to generate player prop parlays using data from ESPN and popular sportsbooks thorough the Odds API (https://the-odds-api.com/).

The steps this took takes:
- make_db.py - Creates the empty `nba_stats.db` file to store data.
- espn_api.py - No API key needed, hits the public ESPN endpoint for NBA teams, rosters, and player data.
- odds_api.py - Needs Odds API key (see above), pulls Fanduel and DraftKings player prop data (odds, predicted value) for all NBA players.
- query_data.py - Generates SQL queries to execute on the NBA stats database to extract 'picks' (single bets on player props).
- generate_picks.sh - Top level shell script that archives historical picks, refreshes to an empty database, and runs the above Python files in the above order.

Query string example:

"""
SELECT player_name, avg_points, predicted_points, ppg_over FROM nba_statistics
WHERE avg_points > predicted_points AND ppg_over > 0;
"""

Parlay example:
"""
PARLAY #9 --------------------------------------------------
Kevin Durant rebounds over 6.5 -135
Evan Mobley points over 15.5 -120
Saddiq Bey points over 13.5 -105
Landry Shamet points over 7.5 -105
LeBron James points over 24.5 -120
"""

This repository is still very much a work in progress, more updates to come soon.