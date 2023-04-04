import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import sem
from statistics import mean
import matplotlib.pyplot as plt


# weekday classifier
def classifier(files_to_test):

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

            lr = LogisticRegression(multi_class='multinomial', max_iter=500)

            lr.fit(training_set, training_labels)

            predictions = lr.predict(test_set)

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

        print(f'done analyzing {f_name}')

    # plot accs and sem for this analysis
    plt.errorbar(x=names, y=avg_accs, yerr=avg_sems, capsize=10, fmt='o', label='default(mean)', c='g')

    plt.show()

    # return some information if the caller wishes to use it
    return names, avg_accs, avg_sems


if __name__ == '__main__':

    files_to_test = {'OFCOM': 'csv_files/AllDays.csv', 'OFCOM_WEEKDAY_ONLY': 'csv_files/Weekdays.csv'}

    classifier(files_to_test)