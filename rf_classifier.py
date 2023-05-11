import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import sem
from statistics import mean
import matplotlib.pyplot as plt


# weekday classifier - for csv files with non-aggregated days
# in other words, those generated from process_data with AGG as FALSE
def all_data_classifier(files_to_test):
    names = list(files_to_test.keys())
    avg_accs = []
    avg_sems = []

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
        test_days = sorted(list(set([row.Day.split('-')[0] for row in input_df.itertuples()])))

        # Run analysis across each day in test_days
        for test_day in test_days:

            # split data into training and test set
            training_set = []
            test_set = []
            training_labels = []
            test_labels = []

            # this is to make sure there is only 1 test day always
            person_to_test = {i: False for i in range(1, len(users) + 1)}

            # iterate through the processed data, to construct the sets
            for row in input_df.itertuples():

                row = list(row)

                person = row[1]
                day = row[2].split('-')[0]
                data = row[3:]

                if day == test_day and not person_to_test[person]:
                    test_labels.append(person)
                    test_set.append(data)
                    person_to_test[person] = True
                else:
                    training_labels.append(person)
                    training_set.append(data)

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

            rf = RandomForestClassifier(n_estimators=100, random_state=46)

            rf.fit(training_set, training_labels)

            predictions = rf.predict(test_set)

            for i, user in enumerate(test_labels):
                users_data[user].append(1 if user == predictions[i] else 0)

            N = test_labels.shape[0]
            accuracy = (test_labels == predictions).sum() / N

            print(f'accuracy when the test day is {test_day}: {accuracy * 100}%')

            # try sigmoid/other activation functions for the thresholding

        avg_acc = mean(mean(data) for _, data in users_data.items())
        avg_sem = mean(sem(data) for _, data in users_data.items())

        avg_accs.append(avg_acc)
        avg_sems.append(avg_sem)

        print(f'done analyzing {f_name} - avg accuracy was {avg_acc*100}% with sem of {avg_sem*100}%')

    # plot accs and sem for this analysis
    plt.errorbar(x=names, y=avg_accs, yerr=avg_sems, capsize=10, fmt='o', label='default(mean)', c='g')

    plt.show()

    # return some information if the caller wishes to use it
    return names, avg_accs, avg_sems


# weekday classifier - for csv files with aggregated days
def agg_classifier(files_to_test, n_trees=100):
    names = list(files_to_test.keys())
    avg_accs = []
    avg_sems = []

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

        # Calculate the days
        test_days = sorted(list(set([row.Day for row in input_df.itertuples()])))

        # Run analysis across each day in test_days
        for test_day in test_days:

            # split data into training and test set
            training_set = []
            test_set = []
            training_labels = []
            test_labels = []

            # iterate through the processed data, to construct the sets
            for row in input_df.itertuples():

                row = list(row)

                person = row[1]
                day = row[2]
                data = row[3:]

                if day == test_day:
                    test_labels.append(person)
                    test_set.append(data)
                else:
                    training_labels.append(person)
                    training_set.append(data)

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

            rf = RandomForestClassifier(n_estimators=n_trees, random_state=46)

            rf.fit(training_set, training_labels)

            predictions = rf.predict(test_set)

            for i, user in enumerate(test_labels):
                users_data[user].append(1 if user == predictions[i] else 0)

            N = test_labels.shape[0]
            accuracy = (test_labels == predictions).sum() / N

            print(f'accuracy when the test day is {test_day}: {accuracy * 100}%')

            # try sigmoid/other activation functions for the thresholding

        avg_acc = mean(mean(data) for _, data in users_data.items())
        avg_sem = mean(sem(data) for _, data in users_data.items())

        avg_accs.append(avg_acc)
        avg_sems.append(avg_sem)

        print(f'done analyzing {f_name} - avg accuracy was {avg_acc}% with sem of {avg_sem}%')

    # plot accs and sem for this analysis
    plt.errorbar(x=names, y=avg_accs, yerr=avg_sems, capsize=10, fmt='o', label='default(mean)', c='g')

    plt.show()

    # return some information if the caller wishes to use it
    return names, avg_accs, avg_sems


if __name__ == '__main__':
    files_to_test = {'all_days': 'csv_files/ofcom_10_days.csv',
                     'weekdays': 'csv_files/ofcom_10_weekdays.csv',
                     'weekends': 'csv_files/ofcom_10_weekends.csv'}

    all_data_classifier(files_to_test)
