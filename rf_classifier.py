import pandas as pd
import numpy as np
import shap
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import sem
from statistics import mean, median
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import top_k_accuracy_score

plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.titlepad'] = 20
plt.rcParams["font.family"] = "sans serif"
plt.rcParams['axes.facecolor'] = 'aliceblue'

DAYS_NAMES = {'Day1': 'Monday', 'Day2': 'Tuesday', 'Day3': 'Wednesday', 'Day4': 'Thursday', 'Day5': 'Friday',
              'Day6': 'Saturday', 'Day7': 'Sunday'}


def network_analysis(model, data, feature_names, labels):

    print('Doing a SHAP Analysis...')

    explainer = shap.TreeExplainer(model, feature_names=feature_names)
    shap_values = explainer(data)
    shap.plots.beeswarm(shap_values[:, :, 1])

    # Importance Analysis
    importances = list(model.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_names, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # plot feature importances in a basic bar plot
    features = [x for x, _ in reversed(feature_importances)]
    vals = [x for _, x in reversed(feature_importances)]

    fig, ax1 = plt.subplots(figsize=(8, 4.8), tight_layout=True)

    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title='Importance Analysis - top 18 apps CRAWDAD ',
        xlabel='Importance',
        ylabel='App (Only top 18 shown)',
    )

    plt.barh(features, vals)

    ax1.xaxis.labelpad = 15
    ax1.yaxis.labelpad = 15

    plt.show()

    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# weekday classifier - for csv files with non-aggregated days
# in other words, those generated from process_data with AGG as FALSE
def all_data_classifier(files_to_test, n_trees=100, max_depth=None):

    # do analysis for each given file
    for f_name, f_path in files_to_test.items():
        # load data from CSV
        print('loading data...')
        input_df = pd.read_csv(f_path)

        # Calculate number of users
        users = list(set([row.Person for row in input_df.itertuples()]))

        # for each user, generate a list (for their accuracies across each day in test_days)
        users_data = {}
        for user in users:
            users_data[user] = []

        # Calculate the days - slightly different than for aggregated data
        # test_days = sorted(list(set([row.Day.split('-')[0] for row in input_df.itertuples()])))
        days_per_user = len(input_df) // len(users)

        # Get all test days
        test_days = sorted(list(set([row.Day.split('-')[0] for row in input_df.itertuples()])))

        # Run analysis across each day in test_days
        for test_day in test_days:

            # split data into training and test set
            training_set = []
            test_set = []
            training_labels = []
            test_labels = []

            # # this is to make sure there is only 1 test day always
            # person_to_test = {i: False for i in range(1, len(users) + 1)}

            # iterate through the processed data, to construct the sets
            for row in input_df.itertuples():

                row = list(row)

                person = row[1]
                day = row[2].split('-')[0]
                data = row[3:]

                if day == test_day:
                    test_labels.append(person)
                    test_set.append(data)
                    # person_to_test[person] = True
                else:
                    training_labels.append(person)
                    training_set.append(data)

            # convert everything to numpy arrays
            training_set = np.array(training_set)
            training_labels = np.array(training_labels)
            test_set = np.array(test_set)
            test_labels = np.array(test_labels)

            # Shuffling training data
            seed = 1843
            np.random.seed(seed)
            np.random.shuffle(training_set)
            np.random.seed(seed)
            np.random.shuffle(training_labels)

            rf = RandomForestClassifier(n_estimators=n_trees, random_state=46, max_depth=max_depth)

            rf.fit(training_set, training_labels)

            predictions = rf.predict(test_set)

            predictions_proba = rf.predict_proba(test_set)

            k = 10
            top_k_acc = top_k_accuracy_score(test_labels, predictions_proba, k=10)

            for i, user in enumerate(test_labels):
                users_data[user].append(1 if user == predictions[i] else 0)

            N = test_labels.shape[0]
            accuracy = (test_labels == predictions).sum() / N

            print(f'accuracy when the test day is {test_day}: {accuracy * 100}%, top {k} acc of {top_k_acc * 100}%')

        users_data['test_days'] = test_days

        # At this point, the current file has been processed for all the test days
        # Save the dictionary to a file via pickle to the name of the file.txt
        output_location = f_path.split('.')[0] + '.pickle'
        with open(output_location, 'wb') as f:
            pickle.dump(users_data, f, protocol=pickle.HIGHEST_PROTOCOL)


# Helper to aggregate several days of data according to the METHOD argument
def aggregate_days(data, agg_method, users):

    data_size = len(data[0])
    data_per_user = len(data) // users
    new_labels = []
    weeks_per_user = data_per_user // 7

    index = 0

    new_data = []

    if agg_method == 'MAX':
        for u in range(users):
            # each user has weeks_per_user weeks, so for each week, need one list
            for w in range(weeks_per_user):
                # initialize a list
                week_data = []
                for _ in range(data_size):
                    week_data.append(0)
                # fill in the list from the next 7 entries in data
                user_week_data_all = data[index:index+7]
                for d in user_week_data_all:
                    for i, n in enumerate(d):
                        if n == 1:
                            week_data[i] = 1

                index += 7

                new_data.append(week_data)
                new_labels.append(u+1)  # so labels are 1-778, not 0-777

    if agg_method == 'AVG':
        for u in range(users):
            # each user has weeks_per_user weeks, so for each week, need one list
            for w in range(weeks_per_user):
                # initialize a list
                week_data = []
                for _ in range(data_size):
                    week_data.append(0)
                # fill in the list from the next 7 entries in data
                user_week_data_all = data[index:index + 7]
                for d in user_week_data_all:
                    for i, n in enumerate(d):
                        week_data[i] += n
                # take average for each entry in week_data after summing them
                # averaged over 1 week, so divide by 7
                week_data = [x / 7 for x in week_data]

                index += 7

                new_data.append(week_data)
                new_labels.append(u + 1)  # so labels are 1-778, not 0-777

    return new_data, new_labels


def calc_user_distances(f_path):

    input_df = pd.read_csv(f_path)

    # Calculate number of users
    users = list(set([row.Person for row in input_df.itertuples()]))

    # for each user, generate a list
    users_data = {}
    for user in users:
        users_data[user] = [[], []]  # list 1: their own distances, list 2: other distances

    # Use week 1 fingerprint to do the matching
    test_week = '1'
    no_users = len(set([row.Person for row in input_df.itertuples()]))

    test_set = []
    test_labels = []
    comp_set = []
    comp_labels = []

    # iterate through the processed data, to construct the sets
    for row in input_df.itertuples():

        row = list(row)

        person = row[1]
        week = row[2].split('-')[1]
        data = row[3:]

        if week == test_week:  # and not person_to_test[person]:
            test_labels.append(person)
            test_set.append(data)
        else:
            comp_labels.append(person)
            comp_set.append(data)


    test_set, test_labels = aggregate_days(test_set, 'MAX', no_users)
    comp_set, comp_labels = aggregate_days(comp_set, 'MAX', no_users)

    # Basically, for each user (i.e. test_set)
    for u_finger, u in zip(test_set, test_labels):
        # I need to check the Hamming Distance to their own other fingerprint(s)
        # Also to other users!
        for o_finger, o in zip(comp_set, comp_labels):
            # calculate hamming distance between fingerprints
            # ham distance is len(union) - len(intersection)
            union = [min(x + y, 1) for x, y in zip(u_finger, o_finger)]  # 0 if none, 1 if either or both
            inter = [int((x + y) == 2) for x, y in zip(u_finger, o_finger)]  # 1 if both used
            ham_distance = sum(union) - sum(inter)
            if o == u:
                users_data[u][0].append(ham_distance)
            else:
                users_data[u][1].append(ham_distance)

    max_own_distances = [max(x[0]) for x in users_data.values()]
    min_other_distances = [min(x[1]) for x in users_data.values()]

    # basically if max_own < min_other, they can be identified, so determine how many can be
    identifiable = sum([own < other for own, other in zip(max_own_distances, min_other_distances)])

    print(f'out of {no_users}, {identifiable} are 100% identifiable')

    # basically, it shows their most dissimilar fingerprints are closer then the closes other fingerprint


# this classifier aggregates weekly information and tests this
def weekly_classifier(files_to_test, n_trees=100):

    # do analysis for each given file
    for f_name, f_path in files_to_test.items():
        # load data from CSV
        print('loading data...')
        input_df = pd.read_csv(f_path)

        # Calculate number of users
        users = list(set([row.Person for row in input_df.itertuples()]))

        # for each user, generate a list (for their accuracies across each day in test_days)
        users_data = {}
        for user in users:
            users_data[user] = []

        test_weeks = sorted(list(set([row.Day.split('-')[1] for row in input_df.itertuples()])))
        users = len(set([row.Person for row in input_df.itertuples()]))

        # Run analysis across each day in test_days
        for test_week in test_weeks:

            # split data into training and test set
            training_set = []
            test_set = []
            training_labels = []
            test_labels = []

            # iterate through the processed data, to construct the sets
            for row in input_df.itertuples():

                row = list(row)

                person = row[1]
                week = row[2].split('-')[1]
                data = row[3:]

                if week == test_week:
                    test_labels.append(person)
                    test_set.append(data)
                else:
                    training_labels.append(person)
                    training_set.append(data)

            # KEY DIFFERENCE: AGGREGATE INFORMATION
            test_set, test_labels = aggregate_days(test_set, 'MAX', users)
            training_set, training_labels = aggregate_days(training_set, 'MAX', users)

            # convert everything to numpy arrays
            training_set = np.array(training_set)
            training_labels = np.array(training_labels)
            test_set = np.array(test_set)
            test_labels = np.array(test_labels)

            # Shuffling training data
            seed = 1843
            np.random.seed(seed)
            np.random.shuffle(training_set)
            np.random.seed(seed)
            np.random.shuffle(training_labels)

            rf = RandomForestClassifier(n_estimators=n_trees, random_state=46)

            rf.fit(training_set, training_labels)

            predictions = rf.predict(test_set)

            for i, user in enumerate(test_labels):
                users_data[user].append(1 if user == predictions[i] else 0)

            N = test_labels.shape[0]
            accuracy = (test_labels == predictions).sum() / N

            print(f'accuracy when the test week is {test_week}: {accuracy * 100}%')

        users_data['test_days'] = test_weeks

        # At this point, the current file has been processed for all the test days
        # Save the dictionary to a file via pickle to the name of the file.txt
        output_location = f_path.split('.')[0] + '.pickle'
        with open(output_location, 'wb') as f:
            pickle.dump(users_data, f, protocol=pickle.HIGHEST_PROTOCOL)


# Analyzing overall accuracy vs days
def per_day_plot(file, title):
    # since we are analyzing it per day, we need to get an accuracy per day per user
    with open(file, 'rb') as f:
        file_dict = pickle.load(f)

    # extract information about the test_days
    test_days = file_dict.pop('test_days')

    # get the unique days as well as their indices
    unique_days = list(set([x.split('-')[0] for x in test_days]))
    indices = {u:[] for u in unique_days}
    accs = {u:[] for u in unique_days}

    for i, day in enumerate(test_days):
        for u in unique_days:
            if u in day:
                indices[u].append(i)
                break

    for day, name in DAYS_NAMES.items():

        if day not in indices:
            continue

        day_indices = indices[day]

        for data in file_dict.values():
            for i in day_indices:
                if i >= len(data):
                    break
                accs[day].append(data[i])

    accs = {a: mean(x) for a, x in accs.items()}

    # sort
    accs = {key: accs[key] for key in sorted(accs.keys())}

    # ------------------------------------------------------------
    # Actually doing the plotting
    x = [DAYS_NAMES[x] for x in list(accs.keys())]
    y = list(accs.values())

    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=96, tight_layout=True)

    # Plot bars
    bar1 = ax.bar(x, y)

    ax.bar_label(bar1, labels=[round(ys * 100, 2) for ys in y], padding=3, color='black', fontsize=18)

    ax.set_xlabel('Test Day', fontsize=24, labelpad=20)  # No need for an axis label
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_major_formatter(lambda s, i: f'{s:,.0f}')
    ax.xaxis.set_tick_params(pad=2, labelbottom=True, bottom=True, labelsize=18, labelrotation=0)
    plt.xticks(x, x, rotation=45, fontsize='18')

    ax.spines[['top', 'left', 'bottom', 'right']].set_visible(False)

    plt.ylim(bottom=0.34)

    # Determine the y-limits of the plot
    ymin, ymax = ax.get_ylim()
    # Calculate a suitable y position for the text label
    average = mean(list(y))
    y_pos = average / ymax - 0.41
    plt.axhline(y=average, color='grey', linewidth=3)
    ax.text(0.96, y_pos, f'Average = {average*100:.2f}', ha='right', va='center', transform=ax.transAxes, size=18, zorder=3)

    ax.set_ylabel('Top 1 accuracy', fontsize=24, labelpad=20)

    # Add in title and subtitle
    ax.text(x=0.12, y=.93, s=title, transform=fig.transFigure,fontsize=14, weight='bold', alpha=.8, size=36)

    # Adjust the margins around the plot area
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.85, wspace=None, hspace=None)

    # Set a white background
    fig.patch.set_facecolor('white')

    plt.show()


def extract_accs(files):

    overall_accuracies = []
    names = []

    for name, fpath in files.items():
        with open(fpath, 'rb') as f:
            file_dict = pickle.load(f)
            if 'test_days' in file_dict:
                file_dict.pop('test_days')  # discard this

        test_days = max([len(data) for data in file_dict.values()])

        accuracies_by_day = [[] for _ in range(test_days)]
        for _, data in file_dict.items():
            for i, d in enumerate(data):
                accuracies_by_day[i].append(d)

        accuracies_by_day = [mean(x) for x in accuracies_by_day]

        accuracies_by_day = list(sorted(accuracies_by_day))[-2]
        names.append(name)

        # reduces to one average accuracy
        overall_accuracies.append(accuracies_by_day)

    return names, overall_accuracies


def line_plot(finger, dur, pickup, title):

    # extract results
    names, finger = extract_accs(finger)
    _, dur = extract_accs(dur)
    _, pickup = extract_accs(pickup)

    # plot all the results in a line graph!

    # start the actual plotting
    fig, ax1 = plt.subplots(figsize=(8, 4.8), tight_layout=True)

    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel='Number of Apps',
        ylabel='Top 1 accuracy',
    )

    ax1.xaxis.labelpad = 15
    ax1.yaxis.labelpad = 15

    plt.plot(names, dur, c='red', label='Durations per App')
    plt.plot(names, pickup, c='blue', label='Pickups per App')
    plt.plot(names, finger, c='green', label='Fingerprints')

    plt.legend()

    plt.show()


# Analyzing overall accuracy across files
def weekdays_plot(files, title):

    overall_accuracies = []

    for name, fpath in files.items():
        with open(fpath, 'rb') as f:
            file_dict = pickle.load(f)
            if 'test_days' in file_dict:
                file_dict.pop('test_days')  # discard this

        # test_days = max([len(data) for data in file_dict.values()])

        test_days = 7
        max_days = max([len(data) for data in file_dict.values()])

        accuracies_by_day = [[] for _ in range(test_days)]
        for _, data in file_dict.items():


            for i, d in enumerate(data):
                accuracies_by_day[i].append(d)

        accuracies_by_day = [mean(x) for x in accuracies_by_day]

        overall_accuracies.append(accuracies_by_day)


    # start the actual plotting
    fig, ax1 = plt.subplots(figsize=(8, 4.8), tight_layout=True)

    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=title,
        xlabel='Technique Employed',
        ylabel='Top 1 accuracy',
    )

    ax1.xaxis.labelpad = 15
    ax1.yaxis.labelpad = 15

    bp_dict = ax1.boxplot(overall_accuracies, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    ax1.set_xticklabels(files.keys(), rotation=25, fontsize=16)

    medians = [round(median(acc) * 100, 2) for acc in overall_accuracies]

    annotate_params = dict(xytext=(115, -5), textcoords='offset points', fontsize=16)
    # annotate_params = dict(xytext=(115, -5), textcoords='offset points', fontsize=16)
    for i, m in enumerate(medians):
        plt.annotate(m, (bp_dict['medians'][i].get_xdata()[0], bp_dict['medians'][i].get_ydata()[0]), **annotate_params)

    plt.show()


# weekday classifier - for csv files with aggregated days
def agg_classifier(files_to_test, n_trees=100, pca_components=0.0, max_depth=None):
    avg_accs = []
    avg_sems = []

    # do analysis for each given file
    for f_name, f_path in files_to_test.items():
        # load data from CSV
        print('loading data...')
        input_df = pd.read_csv(f_path)

        # extract the features used
        feature_names = list(input_df.columns)[2:]

        # Calculate number of users
        users = list(set([row.Person for row in input_df.itertuples()]))

        # for each user, generate a list (for their accuracies across each day in test_days)
        users_data = {}
        for user in users:
            users_data[user] = []

        # Calculate the days
        test_days = sorted(list(set([row.Day.split('-')[0] for row in input_df.itertuples()])))

        # Run analysis across each day in test_days
        for test_day in test_days:

            # split data into training and test set
            training_set = []
            test_set = []
            training_labels = []
            test_labels = []

            # do PCA if required
            if pca_components > 0:
                # Perform PCA to keep only the interesting dimensions if pca_components is used

                pca = PCA(n_components=pca_components, svd_solver='full')
                pca.fit(input_df.iloc[:, 2:])
                data_transformed = list(pca.transform(input_df.iloc[:, 2:]))

                for row, data in zip(input_df.itertuples(), data_transformed):

                    row = list(row)

                    person = row[1]
                    day = row[2]

                    if day == test_day:
                        test_labels.append(person)
                        test_set.append(data)
                    else:
                        training_labels.append(person)
                        training_set.append(data)

            else:

                # iterate through the processed data, to construct the sets
                for row in input_df.itertuples():

                    row = list(row)

                    person = row[1]
                    day = row[2].split('-')[0]
                    data = row[3:]

                    if day == test_day:
                        test_labels.append(person)
                        test_set.append(data)
                    else:
                        training_labels.append(person)
                        training_set.append(data)


            small_training_set = training_set[:778]
            small_training_labels = training_labels[:778]

            # convert everything to numpy arrays
            training_set = np.array(training_set)
            training_labels = np.array(training_labels)
            test_set = np.array(test_set)
            test_labels = np.array(test_labels)

            # Shuffling training data
            seed = 1843  # np.random.randint(0, 10000)
            np.random.seed(seed)
            np.random.shuffle(training_set)
            np.random.seed(seed)
            np.random.shuffle(training_labels)

            rf = RandomForestClassifier(n_estimators=n_trees, random_state=46, max_depth=max_depth)

            rf.fit(training_set, training_labels)

            # network_analysis(rf, small_training_set, feature_names, small_training_labels)

            predictions = rf.predict(test_set)

            predictions_proba = rf.predict_proba(test_set)

            k = 10
            top_k_acc = top_k_accuracy_score(test_labels, predictions_proba, k=10)

            for i, user in enumerate(test_labels):
                users_data[user].append(1 if user == predictions[i] else 0)

            N = test_labels.shape[0]
            accuracy = (test_labels == predictions).sum() / N

            print(f'accuracy when the test day is {test_day}: {accuracy * 100}%, top {k} was {top_k_acc * 100}%')

        # At this point, the current file has been processed for all the test days
        # Save the dictionary to a file via pickle to the name of the file.txt
        output_location = f_path.split('.')[0] + '.pickle'
        with open(output_location, 'wb') as f:
            pickle.dump(users_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        avg_acc = mean(mean(data) for _, data in users_data.items())
        avg_sem = mean(sem(data) for _, data in users_data.items())

        avg_accs.append(avg_acc)
        avg_sems.append(avg_sem)

        print(f'done analyzing {f_name} - avg accuracy was {avg_acc * 100}% with sem of {avg_sem * 100}%')



if __name__ == '__main__':

    # create dictionary of files to classify, then classify them
    files_to_test = {'778': 'csv_files/ofcom_combined_target_times.csv'}
    agg_classifier(files_to_test, n_trees=100, max_depth=None)

    # create dictionary of files to plot, and a title
    files_to_plot = {'Baseline - Durations\n per App': 'csv_files/ofcom_durations_simple.pickle',
                     'Baseline - Pickups\n per App': 'csv_files/ofcom_pickups_simple.pickle',
                     'Introducing & Combining\n new metrics': 'csv_files/ofcom_combined_simple_times.pickle',
                     'Optimized Temporal\n Resolutions for each\n metric': 'csv_files/ofcom_combined_temporal_res.pickle',
                     'SHAP Analysis': 'csv_files/ofcom_combined_target_times.pickle'}

    title = 'More Temporal Information - 778 OFCOM Users with 1 week of data'
    weekdays_plot(files_to_plot, title)

