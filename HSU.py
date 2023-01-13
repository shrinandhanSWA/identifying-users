# Script that generates time event matrices for HSU data
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.decomposition import PCA

MINS_IN_DAY = 1440


# function that calculates z scores for each column of the dataframe
def calc_z_scores(df):

    for col_name in df:
        col = df[col_name]
        avg = col.mean()
        std = col.std()

        col = col.apply(lambda x: (x - avg) / std if std != 0 else 0)

        df[col_name] = col

    return df


# Helper to process user data TODO: compression e.g. PCA
# pca = PCA(n_components=46)  # PCA sample code
# pca.fit(data)
# new_data = pca.singular_values_
def process_user_info(data):

    # takes in a dictionary
    # returns x different dictionaries - 2 for weekends, 5 for weekdays
    # whether data is weekday or weekend is inferred in the code

    days_info = []
    curr_day = -1
    curr_day_info = {}
    days = []

    # iterate through user data
    for name, [value] in data.items():
        bin_info = name.split('-')
        app_name = bin_info[0]
        bin_day = int(bin_info[1])
        bin_number = bin_info[2]

        if bin_day != curr_day:
            if curr_day != -1:  # curr_day only -1 at the start
                days.append(f'Day{curr_day + 1}')
                days_info.append(curr_day_info)
                curr_day_info = {}
            curr_day = bin_day

        new_name = app_name + '-' + bin_number  # omit the day (its per-day now)
        curr_day_info[new_name] = value

    days_info.append(curr_day_info)
    days.append(f'Day{curr_day + 1}')

    return days_info, days


# helper to create empty bins for the given inputs
# these bins are appended onto ori_dict
# apps: list, bins: number (of bins), day: string (0, 1, etc.)
def create_bins(ori_dict, apps, bins, day):

    for app in apps:
        for time in range(bins):
            bin_name = app + '-' + str(day) + '-Bin' + str(time + 1)
            ori_dict[bin_name] = [0]

    return ori_dict


# small helper func to add on new_data to input_df
def update_df(input_df, new_data, user_number):
    # process new_data (this is the user's data for all the days)
    new_data, days = process_user_info(new_data)

    tuples = [(user_number, day) for day in days]
    index = pd.MultiIndex.from_tuples(tuples, names=["User", "Day"])

    new_df = pd.DataFrame(data=new_data, index=index)

    return pd.concat([input_df, new_df], ignore_index=False)  # Might need to change to true


if __name__ == "__main__":

    input = pd.read_csv('EveryonesAppData.csv')


    # first step: identify all the apps
    all_apps = sorted(list(set(input['event'])))  # sorting for consistency
    # removing leading spaces
    all_apps = [x.strip() for x in all_apps]

    # popular apps
    pop_apps = ['Calculator', 'Calendar', 'Camera', 'Clock', 'Contacts', 'Facebook',
                'Gallery', 'Gmail', 'Google Play Store', 'Google Search', 'Instagram',
                'Internet', 'Maps', 'Messaging', 'Messenger', 'Phone', 'Photos',
                'Settings', 'Twitter', 'WhatsApp', 'YouTube']

    all_apps = pop_apps  # popular apps only (for testing)

    # all time bins - split into weekdays and weekends
    # granularity of time bins is determined by the time_gran variable (min)

    time_gran = 1440  # e.g. a day (1440 min)
    no_bins = MINS_IN_DAY // time_gran

    # helper variables
    curr_user = None
    curr_user_weekday_dict = {}
    curr_user_weekend_dict = {}
    curr_user_curr_day = 0
    curr_user_day_count = 0
    user_count = 1

    # overall dataframe for all users (will be built up) - weekday and weekend info separated
    users_weekday_df = pd.DataFrame(data=curr_user_weekday_dict)
    users_weekend_df = pd.DataFrame(data=curr_user_weekend_dict)

    # iterate through all the inputs
    for row in input.itertuples():
        user = row.participantnumber
        timestamp = row.timestamp
        app = row.event.strip()
        duration = row.duration

        if app not in pop_apps:
            continue

        if user != curr_user:
            # moved on to next user, so save previous user's information
            if curr_user is not None:
                users_weekday_df = update_df(users_weekday_df, curr_user_weekday_dict, user_count)
                users_weekend_df = update_df(users_weekend_df, curr_user_weekend_dict, user_count)
                user_count += 1


            # set the rolling parameters for the new user
            curr_user = user
            curr_user_weekday_dict = {}
            curr_user_weekend_dict = {}
            curr_user_curr_time = datetime.fromtimestamp(int(timestamp))
            curr_user_curr_day = curr_user_curr_time.day
            curr_user_day_count = 0

            if datetime.weekday(curr_user_curr_time) in [5, 6]:
                curr_user_weekend_dict = create_bins(curr_user_weekend_dict, all_apps, no_bins, curr_user_day_count)
            else:
                curr_user_weekday_dict = create_bins(curr_user_weekday_dict, all_apps, no_bins, curr_user_day_count)

        # calculate day and hour of this event
        curr_event_time = datetime.fromtimestamp(int(timestamp))

        # check if a new day has been reached
        if curr_event_time.day != curr_user_curr_day:
            curr_user_curr_day = curr_event_time.day
            curr_user_day_count += 1
            if datetime.weekday(curr_event_time) in [5, 6]:
                curr_user_weekend_dict = create_bins(curr_user_weekend_dict, all_apps, no_bins, curr_user_day_count)
            else:
                curr_user_weekday_dict = create_bins(curr_user_weekday_dict, all_apps, no_bins, curr_user_day_count)

        curr_event_day = datetime.weekday(curr_event_time)

        # determine which time bin the current time is in
        curr_event_min = curr_event_time.hour * 60 + curr_event_time.minute
        curr_event_bin = curr_event_min // time_gran

        # create bin name
        bin_name = app + '-' + str(curr_user_day_count) + '-Bin' + str(curr_event_bin + 1)

        if curr_event_day in [5, 6]:  # weekend
            new_duration = curr_user_weekend_dict.get(bin_name, [0])[0] + duration
            curr_user_weekend_dict[bin_name] = [new_duration]
        else:  # weekday
            new_duration = curr_user_weekday_dict.get(bin_name, [0])[0] + duration
            curr_user_weekday_dict[bin_name] = [new_duration]

    # updating info of the last user
    users_weekday_df = update_df(users_weekday_df, curr_user_weekday_dict, user_count)
    users_weekend_df = update_df(users_weekend_df, curr_user_weekend_dict, user_count)


    # OPTIONAL: calculate z-scores for each column
    users_weekday_df = calc_z_scores(users_weekday_df)
    users_weekend_df = calc_z_scores(users_weekend_df)


    print('saving to CSV')

    # save both dataframes to separate CSVs
    users_weekday_df.to_csv('output_weekday.csv')
    users_weekend_df.to_csv('output_weekend.csv')
