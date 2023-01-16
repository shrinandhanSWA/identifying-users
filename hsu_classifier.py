import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


INPUT_WEEKDAY = 'output_weekday.csv'
INPUT_TEST = 'TimeBehavProf46.csv'


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

    curr_person = 0

    # iterate through the processed data
    for id, inf in zip(idx.itertuples(), info.itertuples()):

        person = id.Person
        if person != curr_person:
            curr_person = person
            test_labels.append(person)
            test_set.append(list(np.array(inf)))
        else:
            training_labels.append(person)
            training_set.append(list(np.array(inf)))

    # convert everything to numpy arrays
    training_set = np.array(training_set)
    training_labels = np.array(training_labels)
    test_set = np.array(test_set)
    test_labels = np.array(test_labels)


    print('training classifier')
    rf = RandomForestRegressor(n_estimators=3120)  # original one used 3120 trees
    rf.fit(training_set, training_labels)

    print('testing classifier')
    predictions = rf.predict(test_set)

    # round the predictions off
    predictions = np.round(predictions)

    N = test_labels.shape[0]
    accuracy = (test_labels == predictions).sum() / N

    print(f'accuracy is {accuracy*100}%')

    print('done')


if __name__ == '__main__':

    weekday_classifier()

