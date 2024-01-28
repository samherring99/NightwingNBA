### NightwingNBA Data Analysis ###

This repository is still very much a work in progress, more updates to come soon.

Steps to build the database:
1. Install requirements if needed
2. Run only the first time: `python3 build_database.py`
3. `python3 write_sql_data_to_pt.py` - files will save in saved_data/(train,val, test).pt
4. `python3 train_model.py` - model will save in saved_data/(train,val, test).pt
5. `python3 predict.py`

To run daily predictions:
1. `python3 update_database.py` - to store recent games not yet in the database
2. `python3 predict.py`

Model retraining can be done with:
1. `python3 update_database.py`
2. `python3 write_sql_data_to_pt.py`
3. `python3 train_model.py`
