import os
import pandas as pd
from tqdm import tqdm

INPUT = 'csv_files/mobile_phone_use/data'
OUTPUT = 'csv_files/mobile_phone_use/processed'


# process one user - each user has a CSV file in INPUT
def process_user(file_path):
    info_csv = pd.read_csv(file_path, low_memory=False)

    curr_app = None  # track current app, resets whenever the screen is turned off or another app is opened
    curr_app_start = None
    user_output = pd.DataFrame({})

    for row in info_csv.itertuples():

        event_type = row.sensor_id

        # only concerned with 'Screen' and 'App' events

        if event_type not in ['Screen', 'App']:
            continue

        curr_time = row.ts_raw // 1000  # timestamps are in milliseconds

        if event_type == 'Screen':
            if row.Screen_Value == 'Off':
                if curr_app is not None:
                    duration = curr_time - curr_app_start
                    user_output = user_output.append({'participantnumber': row.uuid,
                                                      'timestamp': curr_app_start,
                                                      'event': curr_app,
                                                      'duration': duration}, ignore_index=True)
                curr_app = None
                curr_app_start = None

        if event_type == 'App':
            app_name = row.App_Value

            # if user opens another app, terminate the current app instance
            if app_name != curr_app and curr_app is not None:
                # save entry
                duration = curr_time - curr_app_start
                user_output = user_output.append({'participantnumber': row.uuid,
                                                  'timestamp': curr_app_start,
                                                  'event': curr_app,
                                                  'duration': duration}, ignore_index=True)

            curr_app = app_name
            curr_app_start = curr_time

    return user_output


# function to rename raw app names to processed names
def rename_apps(input_path):
    apps_dict = {}
    new_df = pd.DataFrame({})
    ignored = []
    input_df = pd.read_csv(input_path)

    # read in apps.txt --> app name dict
    with open('csv_files/mobile_phone_use/apps.txt', 'r') as file:
        lines = [line.rstrip() for line in file]
        for line in lines:
            stuff = line.split(':')
            apps_dict[stuff[0]] = stuff[1]

    # read in ignored.txt --> apps (system apps) to ignore
    with open('csv_files/mobile_phone_use/ignored.txt', 'r') as file:
        lines = [line.rstrip() for line in file]
        for line in lines:
            ignored.append(line)

    for i, row in input_df.iterrows():
        old_app_name = row.event
        new_app_name = old_app_name

        if old_app_name in ignored:
            continue  # ignore it as it's a system app

        if old_app_name in apps_dict:
            new_app_name = apps_dict[old_app_name]

        new_df = new_df.append({'participantnumber': row.participantnumber,
                                'timestamp': row.timestamp,
                                'event': new_app_name,
                                'duration': row.duration}, ignore_index=True)

    return new_df


# function to process all users - calls process_user for each user in
def process_all_users(input_path, output_path):
    for file in tqdm(os.listdir(input_path)):
        user_df = process_user(input_path + '/' + file)
        user_df.to_csv(output_path + '/' + file, index=False)


# merge all created CSVs into one big CSV with all the users
def merge_csvs():
    f_path = 'csv_files/mobile_phone_use/processed'

    overall_df = pd.DataFrame({})

    for file in tqdm(os.listdir(f_path)):
        full_name = f_path + '/' + file
        curr_df = pd.read_csv(full_name)
        overall_df = pd.concat([overall_df, curr_df], ignore_index=False)

    overall_df.to_csv('csv_files/crawdad_raw.csv', index=False)


# do some pre-processing - CSV is already processed, only thing to do is rename the participant numbers (like ofcom.py)
def process_merged_csv(path):

    input = pd.read_csv(path)

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
        unix_time = int(timestamp) // 1000
        input.at[row.Index, 'timestamp'] = unix_time

    print('sorting...')
    input = input.sort_values(['participantnumber', 'timestamp'])

    input = input.dropna()

    # Save the final version, to be used later
    input.to_csv('csv_files/crawdad_processed.csv', index=False)


if __name__ == '__main__':
    process_all_users(INPUT, OUTPUT)
    # merge_csvs()
    # process_merged_csv('csv_files/crawdad_raw.csv')

    # OFCOM only has 18 apps
    # compute_N_pop_apps(18, 'csv_files/crawdad_pop_apps.csv')


    # set up arguments, then call the function
    # input_file = 'csv_files/crawdad_processed.csv'
    #
    # output_file = 'csv_files/crawdad_pop_apps.csv'  # needs to be a tuple when weekdays_split is True
    # # output_file = ('csv_files/ofcom_10_weekdays.csv', 'csv_files/ofcom_10_weekends.csv')
    #
    # time_bins = 1440  # in minutes
    #
    # pop_apps = []
    #
    # weekdays_split = False
    #
    # z_scores = False
    #
    # day_lim = 7  # limit each person to only day_lim days of data
    #
    # user_lim = 778  # limit number of users, default: 10000 (i.e. no limit)
    #
    # selected_days = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7']
    #
    # agg = True
    #
    # # remember, caller has to clear output_file
    # f = open(output_file, "w+")
    # f.close()
    #
    # # f = open(output_file[0], "w+")
    # # f.close()
    # # f = open(output_file[1], "w+")
    # # f.close()
    #
    # # Do the analysis
    # DurationsPerApp(input_file, output_file, time_bins, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
    #                 agg=agg).process_data()