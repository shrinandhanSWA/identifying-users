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


# Given a list of multiple of the same day, return just one final day
# For now, just takes the first one. TODO: average
def process_different_day_info(days):
    return days[0]


# Helper to process user data TODO: compress user daily data
# pca = PCA(n_components=46)  # PCA sample code
# pca.fit(data)
# new_data = pca.singular_values_
def process_user_info(data, day_lim):
    # data: {day_of_week --> [day_info_dicts]}

    days = []
    days_info = []
    day_count = 0  # just used for limit breaching check

    for day in range(7):

        if day_count == day_lim:
            break

        day_info = data[day]  # list

        if not day_info:
            continue

        final_day_info = process_different_day_info(day_info)  # one dictionary

        days_info.append(final_day_info)
        days.append(f'Day{day + 1}')

        day_count += 1


    return days_info, days


# helper to create empty bins for the given inputs
# these bins are appended onto ori_dict
# apps: list, bins: number (of bins), day: string (0, 1, etc.)
# the bin also stores the day of the week!
def create_bins(ori_dict, apps, bins, day_of_week):
    new_dict = {}

    for app in apps:
        for time in range(bins):
            bin_name = app + '-Bin' + str(time + 1)
            new_dict[bin_name] = 0

    ori_dict[day_of_week].append(new_dict)

    return ori_dict


# small helper func to add on new_data to input_df
def update_df(input_df, new_data, user_number, day_lim):
    # process new_data (this is the user's data for all the days)
    new_data, days = process_user_info(new_data, day_lim)

    tuples = [(user_number, day) for day in days]
    index = pd.MultiIndex.from_tuples(tuples, names=["Person", "Day"])

    new_df = pd.DataFrame(data=new_data, index=index)

    return pd.concat([input_df, new_df], ignore_index=False)


# function that processes HSU data with the given arguments
def process_data(time_gran=1440, pop_apps_only=True, weekdays_split=True, z_score=True, day_lim=7):
    input = pd.read_csv('csv_files/EveryonesAppData.csv')

    # first step: identify all the apps
    all_apps = sorted(list(set(input['event'])))  # sorting for consistency
    # removing leading spaces
    all_apps = [x.strip() for x in all_apps]

    # only consider the most popular apps (defined below)
    if pop_apps_only:
        # popular apps
        pop_apps = ['Calculator', 'Calendar', 'Camera', 'Clock', 'Contacts', 'Facebook',
                    'Gallery', 'Gmail', 'Google Play Store', 'Google Search', 'Instagram',
                    'Internet', 'Maps', 'Messaging', 'Messenger', 'Phone', 'Photos',
                    'Settings', 'Twitter', 'WhatsApp', 'YouTube']

        all_apps = pop_apps

    # all time bins - split into weekdays and weekends if required
    # granularity of time bins is determined by the time_gran variable (in min)

    no_bins = MINS_IN_DAY // time_gran

    # helper variables
    curr_user = None
    curr_user_weekday_dict = {i: [] for i in range(7)}  # {day_of_week --> [per_day_info_dicts]}
    curr_user_weekend_dict = {i: [] for i in range(7)}  # there will be a MAX of 3 weekends
    curr_user_curr_day = 0
    user_count = 1

    # overall dataframe for all users (will be built up) - weekday and weekend info separated
    users_weekday_df = pd.DataFrame({})
    users_weekend_df = pd.DataFrame({})

    # this variable determines which days are weekends
    # if weekdays_split is true, this results in 2 CSVs (weekday and weekend)
    # otherwise, one combined CSV is produced (with all days)
    weekend_days = [5, 6] if weekdays_split else []

    # iterate through all the inputs
    for row in input.itertuples():
        user = row.participantnumber
        timestamp = row.timestamp
        app = row.event.strip()
        duration = row.duration

        if app not in all_apps:
            continue

        if user != curr_user:
            # moved on to next user, so save previous user's information
            if curr_user is not None:
                users_weekday_df = update_df(users_weekday_df, curr_user_weekday_dict, user_count, day_lim)
                users_weekend_df = update_df(users_weekend_df, curr_user_weekend_dict, user_count, day_lim)
                user_count += 1

            # set the rolling parameters for the new user
            curr_user = user
            curr_user_weekday_dict = {i: [] for i in range(7)}
            curr_user_weekend_dict = {i: [] for i in range(7)}
            curr_user_curr_time = datetime.fromtimestamp(int(timestamp))
            curr_user_curr_day = curr_user_curr_time.day
            day_of_week = datetime.weekday(curr_user_curr_time)

            if day_of_week in weekend_days:
                curr_user_weekend_dict = create_bins(curr_user_weekend_dict, all_apps, no_bins, day_of_week)
            else:
                curr_user_weekday_dict = create_bins(curr_user_weekday_dict, all_apps, no_bins, day_of_week)

        # calculate day and hour of this event
        curr_event_time = datetime.fromtimestamp(int(timestamp))

        # check if a new day has been reachedZ
        if curr_event_time.day != curr_user_curr_day:
            curr_user_curr_day = curr_event_time.day
            day_of_week = datetime.weekday(curr_event_time)
            if day_of_week in weekend_days:
                curr_user_weekend_dict = create_bins(curr_user_weekend_dict, all_apps, no_bins, day_of_week)
            else:
                curr_user_weekday_dict = create_bins(curr_user_weekday_dict, all_apps, no_bins, day_of_week)

        curr_event_day = datetime.weekday(curr_event_time)

        # determine which time bin the current time is in
        curr_event_min = curr_event_time.hour * 60 + curr_event_time.minute
        curr_event_bin = curr_event_min // time_gran

        # create bin name
        bin_name = app + '-Bin' + str(curr_event_bin + 1)

        if curr_event_day in weekend_days:  # weekend
            curr_day_info = curr_user_weekend_dict.get(curr_event_day)[-1]
            new_duration = curr_day_info.get(bin_name, 0) + duration
            curr_user_weekend_dict[curr_event_day][-1][bin_name] = new_duration
        else:  # weekday
            curr_day_info = curr_user_weekday_dict.get(curr_event_day)[-1]
            new_duration = curr_day_info.get(bin_name, 0) + duration
            curr_user_weekday_dict[curr_event_day][-1][bin_name] = new_duration

    # updating info of the last user
    users_weekday_df = update_df(users_weekday_df, curr_user_weekday_dict, user_count, day_lim)
    users_weekend_df = update_df(users_weekend_df, curr_user_weekend_dict, user_count, day_lim)

    # OPTIONAL: calculate z-scores for each column
    if z_score:
        users_weekday_df = calc_z_scores(users_weekday_df)
        users_weekend_df = calc_z_scores(users_weekend_df)

    print('saving to CSV')

    # save both dataframes to separate CSVs
    if weekdays_split:
        users_weekday_df.to_csv('csv_files/output_weekday.csv')
        users_weekend_df.to_csv('csv_files/output_weekend.csv')
    else:
        # if no weekday-weekend split, store output in one CSV file
        users_weekday_df.to_csv('csv_files/output.csv')


if __name__ == "__main__":
    # set up arguments, then call the function
    time_bins = 1440  # in minutes
    pop_apps_only = True
    weekdays_split = True
    z_scores = False
    day_lim = 7  # limit each person to only day_lim days of data

    process_data(time_bins, pop_apps_only, weekdays_split, z_scores, day_lim)
