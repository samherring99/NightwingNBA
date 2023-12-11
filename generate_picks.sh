#!/bin/zsh

# Make the directory to store picks over time if it doesn't exist
mkdir picks

# Copy the old picks to this directory, if it exists
cp picks.txt "picks/picks_`date -v-1d +%m%d`.txt"

# Remove the old stats database and old picks
rm nba_stats.db
rm picks.txt

# Make a new file to store the picks
touch picks.txt

# Make the database to store statistics
python3 make_db.py

# Run the espn_api file to store ESPN statistics in the database
python3 espn_api.py

# Run the odds_api file to store odds in the database
python3 odds_api.py

# Generate picks 5 times and store in the file
for i in {1..5}
do
    python3 query_data.py >> picks.txt
done