# Python script that processes OFCOM data


import pandas as pd
from dateutil.parser import parse
from tqdm import tqdm
from process_data import DurationsPerApp, DurationsOverall, PickupsPerApp, PickupsOverall, FirstUseTime, AverageDurationPerApp


# Simple csv crop function
def crop(file):
    df = pd.read_csv(file)

    df_cropped = df.head(10)

    df_cropped.to_csv('ofcom_small.csv')


# Helper that processes the original OFCOM file
# Mainly renaming + sorting to get it into a usable format
def process():
    # desired fields:
    # HashedGUID represents the user
    # PackageName_AppName just means the App name e.g. Whatsapp
    # TimeInfoOnStart_Timestamp --> Time of use
    # AppUsageDuration_InSeconds_mean --> Duration of usage, in seconds
    desired_fields = ['HashedGUID', 'PackageName_AppName', 'TimeInfoOnStart_Timestamp',
                      'AppUsageDuration_InSeconds_mean']

    print('Opening file...')
    df = pd.read_csv('csv_files/ofcom.csv')

    # Step 1: Extract and rename desired columns
    df_fields = df[desired_fields]
    input = df_fields.rename(columns={'HashedGUID': 'participantnumber', 'PackageName_AppName': 'event',
                                      'TimeInfoOnStart_Timestamp': 'timestamp',
                                      'AppUsageDuration_InSeconds_mean': 'duration'})
    users_seen = {}
    users_count = 1

    # Pro-processing required for Ofcom dataset - sort the data!
    # Before that, need to convert UUIDs + timestamps to integers, to be sorted
    # iterate through all the inputs
    for row in tqdm(input.itertuples(), total=len(input.index)):
        user = row.participantnumber
        if user not in users_seen:
            users_seen[user] = users_count
            users_count += 1
        input.at[row.Index, 'participantnumber'] = users_seen[user]

        timestamp = row.timestamp
        time = parse(timestamp)
        unix_time = int(time.timestamp())
        input.at[row.Index, 'timestamp'] = unix_time

    print('sorting...')
    input = input.sort_values(['participantnumber', 'timestamp'])

    # Save the final version, to be used later
    input.to_csv('csv_files/ofcom_processed.csv')


if __name__ == '__main__':
    # crop('ofcom.csv')
    # process()

    # set up arguments, then call the function
    input_file = 'csv_files/ofcom_processed.csv'

    output_file = 'csv_files/ofcom_10_days.csv'  # needs to be a tuple when weekdays_split is True
    # output_file = ('csv_files/ofcom_10_weekdays.csv', 'csv_files/ofcom_10_weekends.csv')

    time_bins = 1440  # in minutes

    pop_apps = []

    weekdays_split = False

    z_scores = False

    day_lim = 10  # limit each person to only day_lim days of data

    user_lim = 778  # limit number of users, default: 10000 (i.e. no limit)

    selected_days = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7']

    agg = False

    # remember, caller has to clear output_file
    f = open(output_file, "w+")
    f.close()

    # f = open(output_file[0], "w+")
    # f.close()
    # f = open(output_file[1], "w+")
    # f.close()

    # Do the analysis
    DurationsPerApp(input_file, output_file, time_bins, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                  agg=agg).process_data()
    DurationsOverall(input_file, output_file, 180, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                  agg=agg).process_data()
    PickupsPerApp(input_file, output_file, time_bins, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                  agg=agg).process_data()
    PickupsOverall(input_file, output_file, 180, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                  agg=agg).process_data()
    FirstUseTime(input_file, output_file, time_bins, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                  agg=agg).process_data()
    AverageDurationPerApp(input_file, output_file, time_bins, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                  agg=agg).process_data()