import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


INPUT_WEEKDAY = 'csv_files/output_weekday.csv'
INPUT_TEST = 'csv_files/TimeBehavProf46.csv'
INPUT = 'csv_files/TimeBehavProf.csv'


# weekday classifier
def weekday_classifier():
    # load data from CSV
    print('loading data...')
    input_df = pd.read_csv(INPUT_TEST)

    # split data into training and test set
    # for now, test set will contain one entry for every person
    print('splitting data...')
    training_set = []
    test_set = []
    training_labels = []
    test_labels = []

    # to be used later
    cols = input_df.columns

    info = input_df.iloc[:, 2:]
    idx = input_df.iloc[:, :2]

    # iterate through the processed data
    for id, inf in zip(idx.itertuples(), info.itertuples(index=False)):

        # Day 1 --> testing, other days --> training
        day = id.Day
        if day == 'Day1':
            test_labels.append(id.Person)
            test_set.append(list(np.array(inf)))
        else:
            training_labels.append(id.Person)
            training_set.append(list(np.array(inf)))


        # splitting data in the same way as the original analysis
        # day = id.Day
        # if day in ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6']:
        #     training_labels.append(id.Person)
        #     training_set.append(list(np.array(inf)))
        # elif day == 'Day7':
        #     test_labels.append(id.Person)
        #     test_set.append(list(np.array(inf)))


    # convert everything to numpy arrays
    training_set = np.array(training_set)
    training_labels = np.array(training_labels)
    test_set = np.array(test_set)
    test_labels = np.array(test_labels)

    # Shuffling training data
    seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    np.random.shuffle(training_set)
    np.random.seed(seed)
    np.random.shuffle(training_labels)


    print('training classifier...')
    rf = RandomForestRegressor(n_estimators=3120)
    # original analysis used 3120 trees (n_estimators)

    rf.fit(training_set, training_labels)

    print('testing classifier...')
    predictions = rf.predict(test_set)

    # round the predictions off
    predictions = np.round(predictions)

    N = test_labels.shape[0]
    accuracy = (test_labels == predictions).sum() / N

    print(f'accuracy is {accuracy*100}%')

    print('done')


if __name__ == '__main__':

    weekday_classifier()

