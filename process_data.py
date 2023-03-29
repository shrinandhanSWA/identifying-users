# Helper functions to process data
# the input file MUST contain the following 4 rows with these exact names:
# 1. participantnumber --> the UUID of the user, can be any type
# 2. timestamp --> event timestamp, in UNIX timestamp format
# 3. event --> App in use
# 4. duration --> Duration of usage, in seconds
# Note that in general, the data must be sorted on (participantnumber, timestamp) to work smoothly
# Lastly, if this is run to merge multiple datasets, remember to turn z-scores off
import math
from functools import reduce
import pandas as pd
from datetime import datetime
from statistics import mean
from tqdm import tqdm

MINS_IN_DAY = 1440
DAYS_OF_WEEK = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
WEEKDAYS = ('Day1', 'Day2', 'Day3', 'Day4', 'Day5')
WEEKENDS = ('Day6', 'Day7')


# function that calculates z scores for each column of the given dataframe
def calc_z_scores(df):
    to_be_removed = []

    for col_name in df:
        col = df[col_name]
        avg = col.mean()
        std = col.std()

        # remove column if the column is entirely 0
        if avg == 0:
            to_be_removed.append(col_name)
        else:
            col = col.apply(lambda x: (x - avg) / std if std != 0 else 0)
            df[col_name] = col

    df = df.drop(columns=to_be_removed)

    return df


# Given a list of multiple of the same day, return just one final day
# Currently taking the average across all days
def process_different_day_info(days):
    new_day_info = {}

    for day in days:
        # day is a dictionary
        for bin, duration in day.items():
            val = new_day_info.get(bin, [])
            val.append(duration)
            new_day_info[bin] = val

    # take average
    for bin, values in new_day_info.items():
        # new_day_info[bin] = mean(values)
        # new_day_info[bin] = int(values[0] > 0)
        # new_day_info[bin] = int(mean([int(value > 0) for value in values]) > 0)
        # new_day_info[bin] = values[0]

        import numpy as np
        # sig = lambda x: 1/(1 + np.exp(-x))
        # new_day_info[bin] = sig(mean(values))  # sigmoid activation

        tanh = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        new_day_info[bin] = tanh(mean(values)) if not math.isnan(tanh(mean(values))) else 0  # tanh activation


    return new_day_info


# Helper to process user data TODO: compress user daily data
# pca = PCA(n_components=46)  # PCA sample code
# pca.fit(data)
# new_data = pca.singular_values_
def process_user_info(data, day_lim, expected_days):
    # data: {day_of_week --> [day_info_dicts]}

    days = []
    days_info = []

    for day in range(7):

        day_info = data[day]  # list

        if not day_info:
            continue

        final_day_info = process_different_day_info(day_info)  # one dictionary

        days_info.append(final_day_info)
        days.append(f'Day{day + 1}')

    # enforce day_lim + determine if the data is complete
    # current idea: hard coded days to be selected (for day limit checking)
    if len(days) == 7:
        # in this case, the raw data is a combination of weekdays and weekends
        # for this scenario, the data is obviously complete, then enforce limits if necessary
        # limits are hardcoded at the moment
        all_data = True
        if day_lim < 7:
            indices = [i for i in range(7) if days[i] in weekend_selected or days[i] in weekday_selected]
        else:
            indices = [*range(7)]
    elif len(days) == 5:
        # in this case, there are 2 scenarios: the first is when we are splitting weekdays and weekends and
        # this is the weekdays case (5 days), or when we are not splitting days but the user happens to have
        # 5 days of data. These cases need to be handled separately.
        # expected_days is defined at the start of the master for loop depending on the weekdays_split var
        if expected_days == 5:
            # weekdays only, so by the if case, we have complete data, since the caller has only put weekdays data in
            # thus, the data is complete and once again, enforce limits
            all_data = True
            if day_lim < 5:
                indices = [i for i in range(5) if days[i] in weekday_selected]
            else:
                indices = [*range(5)]
        else:
            # in this case, this means there is no split. This is only OK if the day_lim < 5, since we have a chance.
            # otherwise, we discard it as it is incomplete
            if day_lim < 5:
                chosen_days = weekend_selected + weekday_selected
                indices = [i for i in range(len(days)) if days[i] in chosen_days]
                all_data = len(indices) == len(chosen_days)
            else:
                indices = []
                all_data = False

    elif len(days) == 2:
        # in this case, the raw data has been separated, and this is the weekends data
        # need to check if the weekends data is actually weekends data and that we only expect there to be 2 days
        # don't need to enforce limits here since only a limit of 1 would be an issue and that doesn't make sense!
        all_data = reduce(lambda x, y: x and y, [d in WEEKENDS for d in days]) and expected_days == 2
        indices = [*range(2)]
    elif len(days) == 0:
        # This case is only reachable in 2 case: when the weekend_split is set to false
        # and we call update_df on the weekend_dict, which is empty.
        # In the other case, if we split weekdays off, then if one of them is empty,
        # we need to handle that case as well.
        # Both cases can be handled by checking expected_days
        indices = []
        all_data = True if expected_days == 0 else False
    elif len(days) >= day_lim:
        # example when this will be hit: if day_lim is 5 and the person happens to have 6 days of data
        # remember, every person needs to have the same data, so this is only allowed
        # if all their days are in the union of weekday_selected and weekend_selected
        # this allows for some more data to be 'allowed' through as long as it meets the requirements
        if expected_days == 2:
            chosen_days = weekend_selected
        elif expected_days == 5:
            chosen_days = weekday_selected
        else:
            chosen_days = weekend_selected + weekday_selected
        indices = [i for i in range(len(days)) if days[i] in chosen_days]
        all_data = len(indices) == len(chosen_days)
    else:
        # in all other cases, it is GUARANTEED be incomplete
        indices = []
        all_data = False

    days_info = [days_info[i] for i in indices]
    days = [days[i] for i in indices]

    return days_info, days, all_data


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


# small helper func to add on new_data to input_df (does weekends and weekdays at the same time)
# input_dfs, new_data, and expected_days are tuples (weekday, weekend)
def update_df(input_dfs, new_data, user_number, day_lim, expected_days):
    input_df_weekday = input_dfs[0]
    input_df_weekend = input_dfs[1]

    new_data_weekday = new_data[0]
    new_data_weekend = new_data[1]

    expected_days_weekday = expected_days[0]
    expected_days_weekend = expected_days[1]

    # process weekdays new_data (this is the user's data for weekdays)
    new_data_weekday, days, complete = process_user_info(new_data_weekday, day_lim, expected_days_weekday)

    # If a user does not have the required amount of weekday data, ignore them
    # also decrement the count of users (since this one doesn't count now)
    if not complete:
        return input_dfs, user_number - 1

    tuples = [(user_number, day) for day in days]
    index = pd.MultiIndex.from_tuples(tuples, names=["Person", "Day"])

    new_df = pd.DataFrame(data=new_data_weekday, index=index)
    new_weekdays_df = pd.concat([input_df_weekday, new_df], ignore_index=False)

    # repeat the process for weekends new_data
    new_data_weekend, days, complete = process_user_info(new_data_weekend, day_lim, expected_days_weekend)
    if not complete:
        return input_dfs, user_number - 1

    tuples = [(user_number, day) for day in days]
    index = pd.MultiIndex.from_tuples(tuples, names=["Person", "Day"])

    new_df = pd.DataFrame(data=new_data_weekend, index=index)
    new_weekends_df = pd.concat([input_df_weekend, new_df], ignore_index=False)

    # return the new data, and since the data is complete, no need to decrement the user number
    return (new_weekdays_df, new_weekends_df), user_number


def process_data(input_file, output_file, time_gran=1440, pop_apps=(), weekdays_split=True, z_score=True, day_lim=7,
                 selected_days=WEEKDAYS + WEEKENDS, user_lim=10000):
    # Read in the days to be selected - these are global variables since the need to be accessed elsewhere
    # Also verify that this is the same as the day_limit
    assert len(selected_days) == day_lim, "the number of selected_days must match day_lim!"
    global weekend_selected, weekday_selected
    weekend_selected = [day for day in selected_days if day in WEEKENDS]
    weekday_selected = [day for day in selected_days if day in WEEKDAYS]

    input = pd.read_csv(input_file)

    # first step: identify all the apps
    all_apps = sorted(list(set(input['event'])))  # sorting for consistency
    # removing leading spaces
    all_apps = [x.strip() for x in all_apps]

    # only consider the most popular apps (these are passed into the function)
    if pop_apps:
        # popular apps
        all_apps = pop_apps

    # all time bins - split into weekdays and weekends if required
    # granularity of time bins is determined by the time_gran variable (in min)
    no_bins = MINS_IN_DAY // time_gran

    # helper variables
    curr_user = None
    curr_user_weekday_dict = {i: [] for i in range(7)}  # {day_of_week --> {day-month: per_day_info_dicts}}
    curr_user_weekend_dict = {i: [] for i in range(7)}
    curr_user_curr_day = 0
    user_count = 1
    user_limit_hit = False

    expected_days_weekday = 7 if not weekdays_split else 5
    expected_days_weekend = 0 if not weekdays_split else 2

    # overall dataframe for all users (will be built up) - weekday and weekend info separated
    users_weekday_df = pd.DataFrame({})
    users_weekend_df = pd.DataFrame({})

    # this variable determines which days are weekends
    # if weekdays_split is true, this results in 2 CSVs (weekday and weekend)
    # otherwise, one combined CSV is produced (with all days)
    weekend_days = [5, 6] if weekdays_split else []

    # iterate through all the inputs
    for row in tqdm(input.itertuples(), total=len(input.index)):
        user = row.participantnumber
        timestamp = row.timestamp
        app = row.event.strip()
        duration = row.duration

        if app not in all_apps:
            continue

        if user != curr_user:
            # moved on to next user, so save previous user's information
            if curr_user is not None:
                dfs = (users_weekday_df, users_weekend_df)
                expected_days = (expected_days_weekday, expected_days_weekend)
                user_dicts = (curr_user_weekday_dict, curr_user_weekend_dict)
                (users_weekday_df, users_weekend_df), user_count = update_df(dfs, user_dicts, user_count, day_lim,
                                                                             expected_days)
                user_count += 1
                if user_count > user_lim:
                    user_limit_hit = True
                    break

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

        # check if a new day has been reached

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

    # updating info of the last user, if the user limit has not been hit
    if not user_limit_hit:
        dfs = (users_weekday_df, users_weekend_df)
        expected_days = (expected_days_weekday, expected_days_weekend)
        user_dicts = (curr_user_weekday_dict, curr_user_weekend_dict)
        (users_weekday_df, users_weekend_df), user_count = update_df(dfs, user_dicts, user_count, day_lim,
                                                                     expected_days)
    # OPTIONAL: calculate z-scores for each column
    if z_score:
        users_weekday_df = calc_z_scores(users_weekday_df)
        users_weekend_df = calc_z_scores(users_weekend_df)

    # save both dataframes to separate CSVs
    if weekdays_split:
        # assumes output_file has 2 things - one for weekdays, and the other for weekends, in that order!
        users_weekday_df.to_csv(output_file[0])
        users_weekend_df.to_csv(output_file[1])
    else:
        # if no weekday-weekend split, store output in one CSV file
        users_weekday_df.to_csv(output_file)

# Other possible improvements
# 1. Have a limit on the number of days that are processed --> easy to do
# you can also enforce this by just ignoring all other data in process_different_day_info()
# 2. Don't tie together weekdays and weekends i.e. separate counter for both
# 3. There is an annoying corner case where an app usage event might span across 2 days
# if so, the current code only considers the bin of the starting time. This could be an issue
