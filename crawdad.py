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
                                                      'timestamp': row.ts_raw,
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
                                                  'timestamp': row.ts_raw,
                                                  'event': curr_app,
                                                  'duration': duration}, ignore_index=True)

            curr_app = app_name
            curr_app_start = curr_time

    return user_output


# function to rename raw app names to processed names
def rename_apps(input_df):
    apps_dict = {}
    new_df = pd.DataFrame({})
    apps_missing = set()
    ignored = []

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

        if old_app_name in ignored:
            continue  # ignore it as it's a system app

        if old_app_name not in apps_dict:
            apps_missing.add(old_app_name)
        else:
            new_app_name = apps_dict[old_app_name]

            new_df = new_df.append({'participantnumber': row.participantnumber,
                           'timestamp': row.timestamp,
                           'event': new_app_name,
                           'duration': row.duration}, ignore_index=True)

    if len(apps_missing) != 0:
        print(f'Need to add the following apps to apps.txt: {apps_missing}')

    return new_df


# function to process all users - calls process_user for each user in
def process_all_users(input_path, output_path):
    for file in tqdm(os.listdir(input_path)):
        user_df = process_user(input_path + '/' + file)
        user_df = rename_apps(user_df)
        user_df.to_csv(output_path + '/' + file, index=False)


if __name__ == '__main__':
    process_all_users(INPUT, OUTPUT)
