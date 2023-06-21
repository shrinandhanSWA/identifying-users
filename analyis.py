# Python file that has various statistical analyses

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from merge_datasets import compute_N_pop_apps
from process_data import DurationsPerApp, DurationsOverall, PickupsPerApp, PickupsOverall, FirstUseTimeApp, \
    FirstUseTime, AverageDurationPerApp, DurationsAppClass, Fingerprint, MeanDurationBetweenPickups, PickupsAppClass, \
    LastUseTime, LastUseTimeApp
from rf_classifier import agg_classifier as rf_classifier, weekdays_plot, agg_classifier, line_plot, all_data_classifier
from lr_classifier import classifier as lr_classifier
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np

import shap


# verifying that the ofcom dataset really has 5064 people + plotting how much info there is
def ofcom_data_analysis():
    input = pd.read_csv('csv_files/ofcom_processed.csv')

    users_to_days = {}

    # iterate through every input
    for row in input.itertuples():
        user = row.participantnumber
        timestamp = row.timestamp
        event_time = datetime.fromtimestamp(int(timestamp))
        event_day = event_time.day

        if user not in users_to_days:
            users_to_days[user] = set()

        users_to_days[user].add(event_day)

    print(f'There were {len(users_to_days)} users')

    # count number of days
    for user, days in users_to_days.items():
        day_count = len(days)
        users_to_days[user] = day_count

    days = list(users_to_days.values())

    plt.hist(days)
    plt.xlabel("Days of data")
    plt.ylabel("Number of users")
    plt.title("Distribution of days of data for each user")
    plt.show()

    users_enough_data = len(list(filter(lambda x: x > 6, days)))
    print(f'There were {users_enough_data} users with 7 or more days of data')


def hsu_app_analysis():
    input = pd.read_csv('csv_files/HSU_processed.csv', index_col=[0, 1])

    cols = list(input.columns)
    cols = [c.split('-')[0] for c in cols]
    apps_count = {c: 0 for c in cols}

    for row in input.itertuples():
        # go through row
        for i, app_usage in enumerate(list(row)[1:]):
            apps_count[cols[i]] += app_usage

    sorted_durations = sorted(apps_count.items(), key=lambda x: x[1], reverse=True)

    bad_apps = ['System UI', 'Samsung Experience Home', 'TouchWiz Home', 'Xperia Home',
                'Android System', 'x']  # To be excluded from popular apps.txt calculation]

    top_apps = [x for x, _ in sorted_durations if x not in bad_apps][:19]

    print(top_apps)


# Helper to run the analysis that determines the optimal temporal resolution for a metric
def time_gran_analysis():
    files_to_test = {}
    files_to_plot = {}
    time_bin_list = [60, 180, 360, 720, 1440]  # resolutions to test

    name = 'csv_files/ofcom_pickup_app'
    print_name = 'Pickups per App Class\n'

    for time_bins in time_bin_list:
        # analysis that investigates different activations/thresholds
        # stores results in results.txt, will all be plotted later

        time_bin = time_bins // 60

        input_file = 'csv_files/ofcom_processed.csv'
        output_file = f'{name}_{time_bin}h.csv'
        # clear output, as required
        f = open(output_file, "w+")
        f.close()

        weekdays_split = False
        z_scores = False
        agg = True
        day_lim = 7  # limit each person to only day_lim days of data
        user_lim = 778  # limit number of users, default: 10000 (i.e. no limit)

        # PICKUPS
        # Pickups per App Class is being test here
        PickupsPerApp(input_file, output_file, 1440, [], weekdays_split, z_scores, day_lim, user_lim=user_lim,
                      agg=agg).process_data()
        MeanDurationBetweenPickups(input_file, output_file, 1440, [], weekdays_split, z_scores, day_lim,
                                   user_lim=user_lim,
                                   agg=agg).process_data()
        PickupsAppClass(input_file, output_file, time_bins, [], weekdays_split, z_scores, day_lim, user_lim=user_lim,
                        agg=agg).process_data()

        files_to_test[f'{output_file}'] = output_file

        bin_print_name = f'{print_name} \n{time_bin}h resolution'
        files_to_plot[f'{bin_print_name}'] = f'{name}_{time_bin}h.pickle'


    # once all the data has been generated, call the classifier with these files
    agg_classifier(files_to_test, n_trees=100)

    title = 'The effect of Temporal Resolution - Pickups per App Class'
    weekdays_plot(files_to_plot, title)

    return files_to_test


# function that analyzes the impact of adding more data
def data_time_analysis():
    files_to_test = {}
    files_to_plot = {}

    # number of days of data to use
    time_bin_list = [7, 14, 21, 28, 35]

    name = 'csv_files/ofcom_duration'
    print_name = 'Duration'

    for day_lim in time_bin_list:
        # analysis that investigates different activations/thresholds
        # stores results in results.txt, will all be plotted later

        input_file = 'csv_files/ofcom_processed.csv'
        output_file = f'{name}_{day_lim}_days.csv'
        # clear output, as required
        f = open(output_file, "w+")
        f.close()

        pop_apps = []
        weekdays_split = False
        z_scores = False
        agg = False
        user_lim = 778  # limit number of users, default: 10000 (i.e. no limit)

        # metrics to test
        DurationsPerApp(input_file, output_file, 180, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                        agg=agg).process_data()
        DurationsOverall(input_file, output_file, 1440, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                         agg=agg).process_data()
        AverageDurationPerApp(input_file, output_file, 1440, pop_apps, weekdays_split, z_scores, day_lim,
                              user_lim=user_lim,
                              agg=agg).process_data()
        DurationsAppClass(input_file, output_file, 1440, pop_apps, weekdays_split, z_scores, day_lim,
                              user_lim=user_lim,
                              agg=agg).process_data()

        files_to_test[f'{output_file}'] = output_file

        bin_print_name = f'{print_name} {day_lim} Days'
        files_to_plot[f'{bin_print_name}'] = f'{name}_{day_lim}_days.pickle'


    # once all the data has been generated, call the classifier with these files
    all_data_classifier(files_to_test)

    print(files_to_plot)

    title = 'The effect of adding more data - Durations'

    weekdays_plot(files_to_plot, title)

    return files_to_test


# checking how unique the fingerprints are from a given file(s)
def check_fingerprint_uniqueness(files_to_test):

    dupes_list = []

    # do analysis for each given file
    for f_name, f_path in files_to_test.items():
        # load data from CSV
        input_df = pd.read_csv(f_path)

        users = list(set([row.Person for row in input_df.itertuples()]))

        # then, extract the fingerprint information
        fingerprints = {u:[] for u in users}
        for row in input_df.itertuples():

            row = list(row)
            user = row[1]
            data = row[3:]  # this is for a user for a certain day
            fingerprints[user].append(data)

        # now each user will have a list of fingerprints, one per day
        # I am merging these --> like in the other works
        for u, prints in fingerprints.items():

            new_prints = [0 for _ in range(len(prints[0]))]

            for printe in prints:
                for i, p in enumerate(printe):
                    if p == 1:
                        new_prints[i] = 1

            fingerprints[u] = new_prints


        # Dictionary of fingerprints to their counts
        prints = {}

        for f in fingerprints.values():
            encoding = encode_fingerprint(f)

            # increment the number of times this fingerprint has been seen
            prints[encoding] = prints.get(encoding, 0) + 1

        # count the number of duplicates i.e. fingerprints with a count of more than 1
        dupes = sum([x for _, x in prints.items() if x > 1])

        dupes_list.append(dupes)

        print(f'There were {dupes} out of {len(list(fingerprints.values()))}')

    return dupes_list


# unique encoding for a fingerprint - easier comparison
def encode_fingerprint(fingerprint):

    encoding = []

    for i, p in enumerate(fingerprint):
        if p == 1:
            encoding.append(str(i))
            encoding.append('-')

    encoding = ''.join(encoding)

    return encoding


# helper to plot the results in results.txt
def plot_results():
    with open('results.txt', 'r') as f:
        lines = f.readlines()

        # to make the printing a bit clearer
        # targ_indices = [0, 8, 9]  # sig and tanh v2
        # targ_indices = [0, 3]  # mean v bin
        # targ_indices = [0, 1, 2, 3]  # bin w/ different thresholds

        # targ_indices = [19, 21]  # one data point vs z-score
        # targ_indices = [19, 22]  # one data point vs norm
        # targ_indices = [19, 23]  # one data point vs sig
        # targ_indices = [19, 24]  # one data point vs tanh
        # targ_indices = [19, 18]  # one data point vs relu
        # targ_indices = [19, 25]  # one data point vs extra info (1 point)

        # targ_indices = [25, 27]  # overall duration vs overall pickups
        # targ_indices = [19, 26]  # duration vs pickups per app
        # targ_indices = [34, 36]  # duration + pickups w/out and w/ pickups per class
        # targ_indices = [36, 38, 39]  # best one (36) w/ avg. duration/first time of use per bin
        targ_indices = [36, 40]  # best one vs best one but w/ diff. order

        lines = [lines[i] for i in targ_indices]

        for line in lines:
            line = line.strip()
            line_split = line.split(';')
            name = line_split[0]
            labels = ['1', '3', '6', '12', '24']
            means = [float(x) for x in line_split[1].split(',')]
            sems = [float(x) for x in line_split[2].split(',')]

            plt.errorbar(x=labels, y=means, yerr=sems, capsize=10, fmt='o', label=f'{name}')


# helper to calculate the average minimum hamming distance for each file
def check_hamming_distances(files):

    distances = []

    # do analysis for each given file
    for f_name, f_path in files.items():
        # load data from CSV
        input_df = pd.read_csv(f_path)

        users = list(set([row.Person for row in input_df.itertuples()]))

        # then, extract the fingerprint information
        fingerprints = {u: [] for u in users}
        for row in input_df.itertuples():
            row = list(row)
            user = row[1]
            data = row[3:]  # this is for a user for a certain day
            fingerprints[user].append(data)

        # now each user will have a list of fingerprints, one per day
        # I am merging these --> like in the other works
        for u, prints in fingerprints.items():

            new_prints = [0 for _ in range(len(prints[0]))]

            for printe in prints:
                for i, p in enumerate(printe):
                    if p == 1:
                        new_prints[i] = 1

            fingerprints[u] = new_prints

        minimum_distances = []

        # given the fingerprints, need to find the minimum hamming distance for each user
        for u, prints in fingerprints.items():

            # init to big value
            min_distance = 100000

            # iterate through all other users, calc distance, update min if required
            for o, o_prints in fingerprints.items():

                # skip themselves
                if u == o:
                    continue

                # ham distance is len(union) - len(intersection)
                union = [min(x + y, 1) for x, y in zip(prints, o_prints)]  # 0 if none, 1 if either or both
                inter = [int((x + y) == 2) for x, y in zip(prints, o_prints)]  # 1 if both used

                ham_distance = sum(union) - sum(inter)

                min_distance = min(ham_distance, min_distance)

            minimum_distances.append(min_distance)

        # average
        avg_min_ham_dist = sum(minimum_distances) / len(minimum_distances)

        distances.append(avg_min_ham_dist)

        print(f'Resolution of {f_name} had a minimum Hamming Distance of {avg_min_ham_dist}')

    return distances


# plot a line graph with 2 axes
def dual_line_plot(xs, y1, y2):
    fig, ax1 = plt.subplots(figsize=(9, 6), tight_layout=True)
    ax2 = ax1.twinx()

    COLOR_1 = "#69b3a2"
    COLOR_2 = "#3399e6"

    ax2.plot(xs, y1, color=COLOR_1, lw=4, label='Average Minimum Hamming Distance')
    ax1.plot(xs, y2, color=COLOR_2, lw=4, label='Number of Anonymous Users')

    ax1.set_xlabel("Number of Weeks of Data")

    ax1.set_ylabel("Number of Anonymous Users", color=COLOR_2, fontsize=18)
    ax1.tick_params(axis="y", labelcolor=COLOR_2)

    ax2.set_ylabel("Average Minimum Hamming Distance", color=COLOR_1, fontsize=18)
    ax2.tick_params(axis="y", labelcolor=COLOR_1)

    ax1.xaxis.labelpad = 15
    ax1.yaxis.labelpad = 15
    ax2.yaxis.labelpad = 15

    # ax1.set_xticklabels(list(xs), rotation=25, fontsize=16)

    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    fig.suptitle("Effect of Considering Different Amounts of Data \n on Min. Hamming Distance & # Anonymous Users", fontsize=20)

    plt.show()


if __name__ == "__main__":
    # sample
    files = time_gran_analysis()


