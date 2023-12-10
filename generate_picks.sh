#!/bin/zsh

mkdir picks

cp picks.txt "picks/picks_`date -v-1d +%m%d`.txt"

rm nba_stats.db
rm picks.txt
touch picks.txt

python3 make_db.py
python3 espn_api.py
python3 read_games.py

for i in {1..5}
do
    python3 query_data.py >> picks.txt
done