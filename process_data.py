# Class to process data
# the input file MUST contain the following 4 rows with these exact names:
# 1. participantnumber --> the UUID of the user, can be any type
# 2. timestamp --> event timestamp, in UNIX timestamp format
# 3. event --> App in use
# 4. duration --> Duration of usage, in seconds
# Note that in general, the data must be sorted on (participantnumber, timestamp) to work smoothly
# Lastly, if this is run to merge multiple datasets, remember to turn z-scores off
# 2 functions must be implemented - create_bins() and get_bin_name()
# IMPORTANT: At the start, the caller must ensure output_file is empty, since information gets appended now
# do this by: f = open(filename, "w+"); f.close() --> w+ truncates CSV, clearing it
# At the end, there are some ready-made classes

import math
from functools import reduce
import pandas as pd
from datetime import datetime
from statistics import mean
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod
import os

MINS_IN_DAY = 1440
DAYS_OF_WEEK = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
WEEKDAYS = ('Day1', 'Day2', 'Day3', 'Day4', 'Day5')
WEEKENDS = ('Day6', 'Day7')
sig = lambda x: 1 / (1 + np.exp(-x))
tanh = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
relu = lambda x: max(0, x)


class ProcessData(ABC):

    # initialize parameters
    def __init__(self, input_file, output_file, time_gran=1440, pop_apps=(), weekdays_split=True, z_score=True,
                 day_lim=7,
                 selected_days=WEEKDAYS + WEEKENDS, user_lim=10000, agg=False):
        self.input_file = input_file
        self.output_file = output_file
        self.time_gran = time_gran
        self.pop_apps = pop_apps
        self.weekdays_split = weekdays_split
        self.z_score = z_score
        self.day_lim = day_lim
        self.selected_days = selected_days
        self.user_lim = user_lim
        self.agg = agg

        # check compatibility of these 2 arguments if agg is true
        if self.agg:
            assert len(selected_days) == day_lim, "the number of selected_days must match day_lim!"

        # Read in the days to be selected - these are global variables since the need to be accessed elsewhere
        # Also verify that this is the same as the day_limit
        self.weekend_selected = [day for day in selected_days if day in WEEKENDS]
        self.weekday_selected = [day for day in selected_days if day in WEEKDAYS]

    @staticmethod
    def is_non_zero_file(fpath):
        return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

    # function that calculates z scores for each column of the given dataframe
    # static to bind it to the class rather than an instance (no self.XX usages)
    @staticmethod
    def calc_z_scores(df):
        to_be_removed = []

        for col_name in df:
            col = df[col_name]
            avg = col.mean()
            std = col.std()

            max = col.max()
            min = col.min()

            # remove column if the column is entirely 0
            if avg == 0:
                to_be_removed.append(col_name)
            else:
                col = col.apply(lambda x: (x - avg) / std if std != 0 else 0)
                # col = col.apply(lambda x: (x - min) / (max - min) if std != 0 else 0)

                df[col_name] = col

        df = df.drop(columns=to_be_removed)

        return df

    # Given a list of multiple of the same day, return just one final day
    # Currently taking the average across all days
    @staticmethod
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
            new_day_info[bin] = values[0]

            # new_day_info[bin] = sig(mean(values))
            # new_day_info[bin] = sig(mean([int(value > 0) for value in values]))  # sigmoid activation (v2)

            # new_day_info[bin] = tanh(mean(values)) if not math.isnan(tanh(mean(values))) else 0
            # new_day_info[bin] = tanh(mean([int(value > 0) for value in values]))  # tanh activation (v2)

        return new_day_info

    # Helper to enforce day_lim + determine if the data is complete
    def check_lim(self, days, day_lim, expected_days):
        # current idea: hard coded days to be selected (for day limit checking)
        if len(days) == 7:
            # in this case, the raw data is a combination of weekdays and weekends
            # for this scenario, the data is obviously complete, then enforce limits if necessary
            # limits are hardcoded at the moment
            all_data = True
            if day_lim < 7:
                indices = [i for i in range(7) if days[i] in self.weekend_selected or days[i] in self.weekday_selected]
            else:
                indices = [*range(7)]
        elif len(days) == 5:
            # in this case, there are 2 scenarios: the first is when we are splitting weekdays and weekends and
            # this is the weekdays case (5 days), or when we are not splitting days but the user happens to have
            # 5 days of data. These cases need to be handled separately.
            # expected_days is defined at the start of the master for loop depending on the weekdays_split var
            if expected_days == 5:
                # weekdays only, so by the if case, we have complete data, since the caller has only put weekdays
                # data in thus, the data is complete and once again, enforce limits
                all_data = True
                if day_lim < 5:
                    indices = [i for i in range(5) if days[i] in self.weekday_selected]
                else:
                    indices = [*range(5)]
            else:
                # in this case, this means there is no split. This is only OK if the day_lim < 5, since we have a
                # chance. otherwise, we discard it as it is incomplete
                if day_lim < 5:
                    chosen_days = self.weekend_selected + self.weekday_selected
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
                chosen_days = self.weekend_selected
            elif expected_days == 5:
                chosen_days = self.weekday_selected
            else:
                chosen_days = self.weekend_selected + self.weekday_selected
            indices = [i for i in range(len(days)) if days[i] in chosen_days]
            all_data = len(indices) == len(chosen_days)
        else:
            # in all other cases, it is GUARANTEED be incomplete
            indices = []
            all_data = False

        return indices, all_data

    # Helper to aggregate user data (when the AGG flag is False)
    def aggregate_user_info(self, data, day_lim, expected_days):
        # data: {day_of_week --> [day_info_dicts]}
        days = []
        days_info = []

        days_done = 0  # needs to be <= day_lim
        day = 0
        progress = False
        day_counters = {i: 1 for i in range(7)}

        acc_lim = expected_days * 5  # 10 for weekends, 25 for weekdays

        while days_done < day_lim:

            if data[day]:  # i.e. if there is anything left
                curr_day_info = data[day][0]  # list
                del data[day][0]

                days_info.append(curr_day_info)
                days.append(f'Day{day + 1}-{day_counters[day]}')
                day_counters[day] += 1

                days_done += 1
                progress = True

            day = (day + 1) % 7  # wrap around

            # check progress
            if day == 0:
                if progress:
                    progress = False
                else:
                    break


        # if expected days is 2, then we will already have the required number of days
        if expected_days == 7:  # this is all_days, need 3 random weekends and 7 random weekdays (over time)
            # TODO
            pass
        # if expected_days == 5:  # pick 2 random weekdays per week
        #     new_days = []
        #     new_info = []
        #
        #     # step 1: split days into weeks
        #     weeks = [[] for _ in range(5)]
        #     weeks_days = [[] for _ in range(5)]
        #     for day, info in zip(days, days_info):
        #         curr_week = int(day.split('-')[-1]) - 1
        #         if curr_week >= 5:
        #             break
        #         weeks[curr_week].append(info)
        #         weeks_days[curr_week].append(day)
        #
        #     # step 2: for each week, pick 2 random days
        #     for i, week in enumerate(weeks):
        #         # pick 2 days
        #         if len(week) < 2:
        #             break
        #
        #         import random
        #         indices = random.sample(range(len(week)), 2)
        #         for index in indices:
        #             new_days.append(weeks_days[i][index])
        #             new_info.append(week[index])
        #
        #     # put it all back
        #     days = new_days
        #     days_info = new_info

        all_data = (days_done >= day_lim and len(days) == 10) or (expected_days == 0)

        return days_info, days, all_data

    # Helper to process user data (when the AGG flag is True)
    # TODO: compress user daily data
    # pca = PCA(n_components=46)  # PCA sample code
    # pca.fit(data)
    # new_data = pca.singular_values_
    def process_user_info(self, data, day_lim, expected_days):
        # data: {day_of_week --> [day_info_dicts]}

        days = []
        days_info = []

        for day in range(7):

            day_info = data[day]  # list

            if not day_info:
                continue

            final_day_info = self.process_different_day_info(day_info)  # one dictionary

            days_info.append(final_day_info)
            days.append(f'Day{day + 1}')

        indices, all_data = self.check_lim(days, day_lim, expected_days)

        days_info = [days_info[i] for i in indices]
        days = [days[i] for i in indices]

        return days_info, days, all_data

    # helper func to add on new_data to input_df (does weekends and weekdays at the same time)
    # input_dfs, new_data, and expected_days are tuples (weekday, weekend)
    def update_df(self, input_dfs, new_data, user_number, day_lim, expected_days, agg):
        input_df_weekday = input_dfs[0]
        input_df_weekend = input_dfs[1]

        new_data_weekday = new_data[0]
        new_data_weekend = new_data[1]

        expected_days_weekday = expected_days[0]
        expected_days_weekend = expected_days[1]

        # process weekdays new_data (this is the user's data for weekdays)
        if agg:
            new_data_weekday, days, complete = self.process_user_info(new_data_weekday, day_lim, expected_days_weekday)
        else:
            new_data_weekday, days, complete = self.aggregate_user_info(new_data_weekday, day_lim,
                                                                        expected_days_weekday)

        # If a user does not have the required amount of weekday data, ignore them
        # also decrement the count of users (since this one doesn't count now)
        if not complete:
            return input_dfs, user_number - 1

        tuples = [(user_number, day) for day in days]
        index = pd.MultiIndex.from_tuples(tuples, names=["Person", "Day"])

        new_df = pd.DataFrame(data=new_data_weekday, index=index)
        new_weekdays_df = pd.concat([input_df_weekday, new_df], ignore_index=False)

        # repeat the process for weekends new_data
        if agg:
            new_data_weekend, days, complete = self.process_user_info(new_data_weekend, day_lim, expected_days_weekend)
        else:
            new_data_weekend, days, complete = self.aggregate_user_info(new_data_weekend, day_lim,
                                                                        expected_days_weekend)

        if not complete:
            return input_dfs, user_number - 1

        tuples = [(user_number, day) for day in days]
        index = pd.MultiIndex.from_tuples(tuples, names=["Person", "Day"])

        new_df = pd.DataFrame(data=new_data_weekend, index=index)
        new_weekends_df = pd.concat([input_df_weekend, new_df], ignore_index=False)

        # return the new data, and since the data is complete, no need to decrement the user number
        return (new_weekdays_df, new_weekends_df), user_number

    # helper to create empty bins for the given inputs
    # needs to be implemented by child class --> implementation specific
    # same with get_bin_name + add_info
    @abstractmethod
    def create_bins(self, ori_dict, apps, bins, day_of_week):
        pass

    @abstractmethod
    def get_bin_name(self, app, bin):
        pass

    @abstractmethod
    def add_info(self, dict, day, bin_name, duration, curr_time):
        pass

    # actually doing the work
    def process_data(self):

        input = pd.read_csv(self.input_file)

        # first step: identify all the apps
        all_apps = sorted(list(set(input['event'])))  # sorting for consistency
        # removing leading spaces
        all_apps = [x.strip() for x in all_apps]

        # only consider the most popular apps (these are passed into the function)
        if self.pop_apps:
            # popular apps
            all_apps = self.pop_apps

        # all time bins - split into weekdays and weekends if required
        # granularity of time bins is determined by the time_gran variable (in min)
        no_bins = MINS_IN_DAY // self.time_gran

        # helper variables
        curr_user = None
        curr_user_weekday_dict = {i: [] for i in range(7)}  # {day_of_week --> {day-month: per_day_info_dicts}}
        curr_user_weekend_dict = {i: [] for i in range(7)}
        curr_user_curr_day = 0
        user_count = 1
        user_limit_hit = False

        expected_days_weekday = 7 if not self.weekdays_split else 5
        expected_days_weekend = 0 if not self.weekdays_split else 2

        # overall dataframe for all users (will be built up) - weekday and weekend info separated
        users_weekday_df = pd.DataFrame({})
        users_weekend_df = pd.DataFrame({})

        # this variable determines which days are weekends
        # if weekdays_split is true, this results in 2 CSVs (weekday and weekend)
        # otherwise, one combined CSV is produced (with all days)
        weekend_days = [5, 6] if self.weekdays_split else []

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
                    (users_weekday_df, users_weekend_df), user_count = self.update_df(dfs, user_dicts, user_count,
                                                                                      self.day_lim,
                                                                                      expected_days, self.agg)
                    user_count += 1
                    if user_count > self.user_lim:
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
                    curr_user_weekend_dict = self.create_bins(curr_user_weekend_dict, all_apps, no_bins, day_of_week)
                else:
                    curr_user_weekday_dict = self.create_bins(curr_user_weekday_dict, all_apps, no_bins, day_of_week)

            # calculate day and hour of this event
            curr_event_time = datetime.fromtimestamp(int(timestamp))

            # check if a new day has been reached

            if curr_event_time.day != curr_user_curr_day:
                curr_user_curr_day = curr_event_time.day
                day_of_week = datetime.weekday(curr_event_time)
                if day_of_week in weekend_days:
                    curr_user_weekend_dict = self.create_bins(curr_user_weekend_dict, all_apps, no_bins, day_of_week)
                else:
                    curr_user_weekday_dict = self.create_bins(curr_user_weekday_dict, all_apps, no_bins, day_of_week)

            curr_event_day = datetime.weekday(curr_event_time)

            # determine which time bin the current time is in
            curr_event_min = curr_event_time.hour * 60 + curr_event_time.minute
            curr_event_bin = curr_event_min // self.time_gran

            # create bin name
            bin_name = self.get_bin_name(app, curr_event_bin)

            if curr_event_day in weekend_days:  # weekend
                self.add_info(curr_user_weekend_dict, curr_event_day, bin_name, duration, curr_event_time)
            else:  # weekday
                self.add_info(curr_user_weekday_dict, curr_event_day, bin_name, duration, curr_event_time)

        # updating info of the last user, if the user limit has not been hit
        if not user_limit_hit:
            dfs = (users_weekday_df, users_weekend_df)
            expected_days = (expected_days_weekday, expected_days_weekend)
            user_dicts = (curr_user_weekday_dict, curr_user_weekend_dict)
            (users_weekday_df, users_weekend_df), user_count = self.update_df(dfs, user_dicts, user_count, self.day_lim,
                                                                              expected_days, self.agg)
        # OPTIONAL: calculate z-scores for each column
        if self.z_score:
            users_weekday_df = self.calc_z_scores(users_weekday_df)
            users_weekend_df = self.calc_z_scores(users_weekend_df)

        # save both dataframes to separate CSVs
        # if the output file(s) are empty CSVs, need to make an empty dataframe + a check for this
        if self.weekdays_split:
            # assumes output_file has 2 things - one for weekdays, and the other for weekends, in that order!
            curr_output_weekday = pd.read_csv(self.output_file[0], index_col=[0, 1]) if self.is_non_zero_file(self.output_file[0]) \
                else pd.DataFrame({})
            curr_output_weekend = pd.read_csv(self.output_file[1], index_col=[0, 1]) if self.is_non_zero_file(self.output_file[1]) \
                else pd.DataFrame({})

            new_output_weekday = pd.concat(
                [curr_output_weekday, users_weekday_df], axis=1)
            new_output_weekend = pd.concat(
                [curr_output_weekend, users_weekend_df], axis=1)

            new_output_weekday.to_csv(self.output_file[0])
            new_output_weekend.to_csv(self.output_file[1])
        else:
            # if no weekday-weekend split, store output in one CSV file
            curr_output = pd.read_csv(self.output_file, index_col=[0, 1]) if self.is_non_zero_file(
                self.output_file) else \
                pd.DataFrame({})
            new_output = pd.concat(
                [curr_output, users_weekday_df], axis=1)
            new_output.to_csv(self.output_file)


# Other possible improvements
# 1. Have a limit on the number of days that are processed --> easy to do
# you can also enforce this by just ignoring all other data in process_different_day_info()
# 2. Don't tie together weekdays and weekends i.e. separate counter for both
# 3. There is an annoying corner case where an app usage event might span across 2 days
# if so, the current code only considers the bin of the starting time. This could be an issue

# --------------------------------------------------------------------------------
# Some examples
# --------------------------------------------------------------------------------

def durations(dict, day, bin_name, duration):
    curr_day_info = dict.get(day)[-1]
    new_info = curr_day_info.get(bin_name, 0) + duration
    dict[day][-1][bin_name] = new_info


def pickups(dict, day, bin_name):
    curr_day_info = dict.get(day)[-1]
    new_info = curr_day_info.get(bin_name, 0) + 1
    dict[day][-1][bin_name] = new_info


class DurationsPerApp(ProcessData):

    def add_info(self, dict, day, bin_name, duration, curr_time):
        return durations(dict, day, bin_name, duration)

    def create_bins(self, ori_dict, apps, bins, day_of_week):
        new_dict = {}

        for app in apps:
            for time in range(bins):
                bin_name = app + '-Dur-' + str(time + 1)
                new_dict[bin_name] = 0

        ori_dict[day_of_week].append(new_dict)

        return ori_dict

    def get_bin_name(self, app, bin):
        return app + '-Dur-' + str(bin + 1)


class DurationsOverall(ProcessData):

    def add_info(self, dict, day, bin_name, duration, curr_time):
        return durations(dict, day, bin_name, duration)

    def create_bins(self, ori_dict, apps, bins, day_of_week):
        new_dict = {}
        for time in range(bins):
            bin_name = 'Overall-Dur-' + str(time + 1)
            new_dict[bin_name] = 0

        ori_dict[day_of_week].append(new_dict)

        return ori_dict

    def get_bin_name(self, app, bin):
        return 'Overall-Dur-' + str(bin + 1)


class PickupsPerApp(ProcessData):

    def create_bins(self, ori_dict, apps, bins, day_of_week):
        new_dict = {}

        for app in apps:
            for time in range(bins):
                bin_name = app + '-PiUp-' + str(time + 1)
                new_dict[bin_name] = 0

        ori_dict[day_of_week].append(new_dict)

        return ori_dict

    def get_bin_name(self, app, bin):
        return app + '-PiUp-' + str(bin + 1)

    def add_info(self, dict, day, bin_name, duration, curr_time):
        return pickups(dict, day, bin_name)


class PickupsOverall(ProcessData):

    def create_bins(self, ori_dict, apps, bins, day_of_week):
        new_dict = {}
        for time in range(bins):
            bin_name = 'Overall-PiUp' + str(time + 1)
            new_dict[bin_name] = 0

        ori_dict[day_of_week].append(new_dict)

        return ori_dict

    def get_bin_name(self, app, bin):
        return 'Overall-PiUp' + str(bin + 1)

    def add_info(self, dict, day, bin_name, duration, curr_time):
        return pickups(dict, day, bin_name)


class DurationsAppClass(ProcessData):
    app_classes = ['Social', 'Info', 'Entertainment', 'Productivity', 'Travel', 'Utilities', 'Creativity', 'Shopping']
    app_to_class = {'BBC News': 'Info', 'Facebook': 'Social', 'Gmail': 'Productivity', 'Inbox': 'Productivity',
                    'Instagram': 'Social', 'Internet': 'Utilities', 'Maps': 'Travel', 'Messenger': 'Social',
                    'Nova Launcher': 'Utilities', 'Outlook': 'Productivity', 'Photos': 'Creativity',
                    'Snapchat': 'Social', 'Spotify': 'Entertainment', 'Twitter': 'Social', 'WhatsApp': 'Social',
                    'Yahoo Mail': 'Productivity', 'YouTube': 'Entertainment', 'eBay': 'Shopping'}

    # eBay as creativity seems sus, and only BBC News is in Social and only Maps is in Travel.
    # I think these are OK though since they are distinct, the idea is grouping similar apps

    def create_bins(self, ori_dict, apps, bins, day_of_week):
        new_dict = {}

        for app in self.app_classes:
            for time in range(bins):
                bin_name = app + '-Dur-' + str(time + 1)
                new_dict[bin_name] = 0

        ori_dict[day_of_week].append(new_dict)

        return ori_dict

    def get_bin_name(self, app, bin):
        return self.app_to_class[app] + '-Dur-' + str(bin + 1)

    def add_info(self, dict, day, bin_name, duration, curr_time):
        return durations(dict, day, bin_name, duration)


class PickupsAppClass(ProcessData):
    app_classes = ['Social', 'Info', 'Entertainment', 'Productivity', 'Travel', 'Utilities', 'Creativity', 'Shopping']
    app_to_class = {'BBC News': 'Info', 'Facebook': 'Social', 'Gmail': 'Productivity', 'Inbox': 'Productivity',
                    'Instagram': 'Social', 'Internet': 'Utilities', 'Maps': 'Travel', 'Messenger': 'Social',
                    'Nova Launcher': 'Utilities', 'Outlook': 'Productivity', 'Photos': 'Creativity',
                    'Snapchat': 'Social', 'Spotify': 'Entertainment', 'Twitter': 'Social', 'WhatsApp': 'Social',
                    'Yahoo Mail': 'Productivity', 'YouTube': 'Entertainment', 'eBay': 'Shopping'}

    # eBay as creativity seems sus, and only BBC News is in Social and only Maps is in Travel.
    # I think these are OK though since they are distinct, the idea is grouping similar apps

    def create_bins(self, ori_dict, apps, bins, day_of_week):
        new_dict = {}

        for app in self.app_classes:
            for time in range(bins):
                bin_name = app + '-Dur-' + str(time + 1)
                new_dict[bin_name] = 0

        ori_dict[day_of_week].append(new_dict)

        return ori_dict

    def get_bin_name(self, app, bin):
        return self.app_to_class[app] + '-Dur-' + str(bin + 1)

    def add_info(self, dict, day, bin_name, duration, curr_time):
        return pickups(dict, day, bin_name)


class FirstUseTimeApp(ProcessData):

    def create_bins(self, ori_dict, apps, bins, day_of_week):
        new_dict = {}

        for app in apps:
            for time in range(bins):
                bin_name = app + '-FirstUse-' + str(time + 1)
                new_dict[bin_name] = 0

        ori_dict[day_of_week].append(new_dict)

        return ori_dict

    def get_bin_name(self, app, bin):
        return app + '-FirstUse-' + str(bin + 1)

    def add_info(self, dict, day, bin_name, duration, curr_time):
        curr_day_info = dict.get(day)[-1]
        old_info = curr_day_info.get(bin_name, 0)

        # get time in seconds in the current day, not the actual timestamp
        curr_seconds = curr_time.hour * 3600 + curr_time.minute * 60 + curr_time.second

        new_info = curr_seconds if old_info == 0 else old_info
        dict[day][-1][bin_name] = new_info


class FirstUseTime(ProcessData):

    def create_bins(self, ori_dict, apps, bins, day_of_week):
        new_dict = {}

        for time in range(bins):
            bin_name = 'FirstUse-' + str(time + 1)
            new_dict[bin_name] = 0

        ori_dict[day_of_week].append(new_dict)

        return ori_dict

    def get_bin_name(self, app, bin):
        return 'FirstUse-' + str(bin + 1)

    def add_info(self, dict, day, bin_name, duration, curr_time):
        curr_day_info = dict.get(day)[-1]
        old_info = curr_day_info.get(bin_name, 0)

        # get time in seconds in the current day, not the actual timestamp
        curr_seconds = curr_time.hour * 3600 + curr_time.minute * 60 + curr_time.second

        new_info = curr_seconds if old_info == 0 else old_info
        dict[day][-1][bin_name] = new_info


class AverageDurationPerApp(ProcessData):

    def __init__(self, input_file, output_file, time_gran=1440, pop_apps=(), weekdays_split=True, z_score=True,
                 day_lim=7,
                 selected_days=WEEKDAYS + WEEKENDS, user_lim=10000, agg=False):

        super().__init__(input_file, output_file, time_gran, pop_apps, weekdays_split, z_score, day_lim,
                         selected_days, user_lim, agg)
        self.counts = {}

    def create_bins(self, ori_dict, apps, bins, day_of_week):

        new_dict = {}
        self.counts = {}

        for app in apps:
            for time in range(bins):
                bin_name = app + '-avgDur-' + str(time + 1)
                new_dict[bin_name] = 0
                self.counts[bin_name] = 0

        ori_dict[day_of_week].append(new_dict)

        return ori_dict

    def get_bin_name(self, app, bin):
        return app + '-avgDur-' + str(bin + 1)

    def add_info(self, dict, day, bin_name, duration, curr_time):
        curr_day_info = dict.get(day)[-1]
        curr_avg = curr_day_info.get(bin_name, 0)
        curr_count = self.counts.get(bin_name, 0)
        new_count = curr_count + 1

        new_avg = ((curr_avg * curr_count) + duration) / new_count

        dict[day][-1][bin_name] = new_avg
        self.counts[bin_name] = new_count
