import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_WEEKDAY = 'output_weekday.csv'
INPUT_WEEKEND = 'output_weekend.csv'

POP_APPS = ['Calculator', 'Calendar', 'Camera', 'Clock', 'Contacts', 'Facebook',
            'Gallery', 'Gmail', 'Google Play Store', 'Google Search', 'Instagram',
            'Internet', 'Maps', 'Messaging', 'Messanger', 'Phone', 'Photos',
            'Settings', 'Twitter', 'WhatsApp', 'YouTube']
DAYS = 7
HOURS = 7

# weekday classifier
def weekday_classifier():

    # load data from CSV
    print('loading data...')
    input_df = pd.read_csv(INPUT_WEEKDAY)

    # shuffle and split data
    print('splitting data...')

    # users are the class(y label) - time information is the x label
    users_df = input_df.reset_index()
    users_df.rename({'Unnamed: 0': 'User'}, axis=1, inplace=True)

    df_cols = users_df.columns # extracting columns, lost in the for loop

    # training and testing dataframes
    train_df = None
    test_df = None

    # unwrap dataframe - original one has entire day split (I will split the same for now)
    for row in users_df.itertuples():
        # for each app, the first 96 entries are training and the last 24 are testing
        for i in range(0, 2520, 120):
            continue
        break

    # X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1),
    #                                                     train['Survived'], test_size=0.30,
    #                                                     random_state=101)

    # print('processing data')
    #
    # print('training classifier')
    #
    # print('testing classifier')
    #
    # print('accuracies:')

    print('done')

# weekend classifier
def weekend_classifer():
    return

if __name__ == '__main__':

    weekday_classifier()

