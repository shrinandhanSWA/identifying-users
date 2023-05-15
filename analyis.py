# Python file that has various statistical analyses

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from process_data import DurationsPerApp, DurationsOverall, PickupsPerApp, PickupsOverall, FirstUseTimeApp, \
    FirstUseTime, AverageDurationPerApp, DurationsAppClass
from rf_classifier import agg_classifier as rf_classifier
from lr_classifier import classifier as lr_classifier

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


def time_gran_analysis(name):
    files_to_test = {}
    time_bin_list = [60, 180, 360, 720, 1440]

    for time_bins in time_bin_list:
        # analysis that investigates different activations/thresholds
        # stores results in results.txt, will all be plotted later
        input_file = 'csv_files/ofcom_processed.csv'
        output_file = f'csv_files/AllDays_{time_bins // 60}h.csv'
        # clear output, as required
        f = open(output_file, "w+")
        f.close()

        pop_apps = []
        weekdays_split = False
        z_scores = False
        agg = True
        day_lim = 7  # limit each person to only day_lim days of data
        user_lim = 778  # limit number of users, default: 10000 (i.e. no limit)

        # for FirstUseTime, 3h time bins looks to be the best
        FirstUseTime(input_file, output_file, 180, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                     agg=agg).process_data()
        DurationsPerApp(input_file, output_file, 1440, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                        agg=agg).process_data()
        DurationsOverall(input_file, output_file, 360, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                         agg=agg).process_data()
        AverageDurationPerApp(input_file, output_file, 1440, pop_apps, weekdays_split, z_scores, day_lim,
                              user_lim=user_lim,
                              agg=agg).process_data()
        PickupsPerApp(input_file, output_file, 1440, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                      agg=agg).process_data()
        PickupsOverall(input_file, output_file, 360, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                       agg=agg).process_data()
        files_to_test[f'{time_bins // 60}h'] = output_file

    # once all the data has been generated, call the classifier with these files
    names, accs, sems = rf_classifier(files_to_test, n_trees=500)

    # write to results.txt - to be loaded up later
    with open('results.txt', 'a') as f:
        to_write = [name, ';']
        for acc in accs:
            to_write.append(str(acc))
            to_write.append(',')
        to_write = to_write[:-1]
        to_write.append(';')
        for sem in sems:
            to_write.append(str(sem))
            to_write.append(',')
        to_write = to_write[:-1]
        to_write.append('\n')

        f.write(''.join(to_write))

    return


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

    plt.xlabel('time bins (h)')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


# Helper to execute the current best configuration
def get_curr_best():
    input_file = 'csv_files/ofcom_processed.csv'
    output_file = f'csv_files/ofcom_best.csv'

    files_to_test = {'curr_best': output_file}

    # clear output, as required
    f = open(output_file, "w+")
    f.close()

    pop_apps = []
    weekdays_split = False
    z_scores = False
    agg = True
    day_lim = 7  # limit each person to only day_lim days of data
    user_lim = 778  # limit number of users, default: 10000 (i.e. no limit)

    FirstUseTime(input_file, output_file, 180, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                 agg=agg).process_data()
    DurationsPerApp(input_file, output_file, 1440, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                    agg=agg).process_data()
    DurationsOverall(input_file, output_file, 360, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                     agg=agg).process_data()
    AverageDurationPerApp(input_file, output_file, 1440, pop_apps, weekdays_split, z_scores, day_lim,
                          user_lim=user_lim,
                          agg=agg).process_data()
    PickupsPerApp(input_file, output_file, 1440, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                  agg=agg).process_data()
    PickupsOverall(input_file, output_file, 360, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                   agg=agg).process_data()

    _, acc, sem = rf_classifier(files_to_test, n_trees=100)

    print(f'Best accuracy is {acc[0]}% with sem of {sem[0]*100}%')

    # return model, test_data


# function that does a SHAP analysis on a given model on the given data (needs to be only the data i.e. no labels)
def network_analysis(model, data):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
    explainer = shap.explainers.Linear(model, data)
    shap_values = explainer(data)

    shap.plots.scatter(shap_values[:, 0])


if __name__ == "__main__":
    # ofcom_data_analysis()
    # hsu_app_analysis()
    # time_gran_analysis('more-trees')
    # plot_results()
    get_curr_best()
    # network_analysis(net, data)

    print()
