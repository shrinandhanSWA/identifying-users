# Python script that attempts to parse the OFCOM CSV file
import pandas as pd
from datetime import datetime

# Simple csv crop function
def crop(file):
    df = pd.read_csv(file)

    df_cropped = df.head(10)

    df_cropped.to_csv('ofcom_small.csv')


# Parses given OFCOM CSV file
def parse(file):

    # desired fields:
    # HashedGUID is one user (I think) - AUSIds are all different (TODO: double check this)
    # There could be an edge case where the start and end days are on different days
    fields = ['HashedGUID', 'PackageName_AppName', 'TimeInfoOnStart_WeekDay', 'TimeInfoOnStart_Timestamp',
              'TimeInfoOnEnd_Timestamp', 'AppUsageDuration_InSeconds_mean']

    df = pd.read_csv(file)

    # Step 1: Extract and rename desired columns
    df_fields = df[fields]
    df_fields = df_fields.rename(columns={'HashedGUID': 'UUID', 'PackageName_AppName': 'App',
                                          'TimeInfoOnStart_WeekDay': 'Day', 'TimeInfoOnStart_Timestamp': 'Start',
                                          'TimeInfoOnEnd_Timestamp': 'End',
                                          'AppUsageDuration_InSeconds_mean': 'Duration'})

    # Step 2: Convert UUIDs into easier to read IDs + convert start and and timestamps to Datetime objects
    uuid_map = {}
    for i, row in df_fields.iterrows():

        # UUID mapping
        row_uuid = row['UUID']

        if row_uuid not in uuid_map:
            uuid_map[row_uuid] = len(uuid_map)

        df_fields.at[i, 'UUID'] = uuid_map[row_uuid]

        # start and end date conversion - TODO - note that the +0000 is offset from GMT/UTC (need to consider this)
        # new_start = 0
        # new_end = new_start + datetime.timedelta(seconds=row['Duration'])


    print(df_fields.head())

if __name__ == '__main__':
    # crop('ofcom.csv')

    parse('ofcom_small.csv')

