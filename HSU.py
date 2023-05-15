# Script that generates time event matrices for HSU data
from process_data import DurationsPerApp
import pandas as pd


# Helper function to merge Messenger + Messaging, also Photos + Gallery
def unify_apps():

    original_hsu = pd.read_csv('csv_files/EveryonesAppData.csv')

    new_hsu = original_hsu.copy(deep=True)

    for row in new_hsu.itertuples():
        if row.event == ' Photos':
            new_hsu.at[row.Index, 'event'] = ' Gallery'
        if row.event == ' Messaging':
            new_hsu.at[row.Index, 'event'] = ' Messenger'
        if row.event in [' Chrome', ' Firefox']:
            new_hsu.at[row.Index, 'event'] = ' Internet'


    new_hsu.to_csv('csv_files/EveryonesAppDataUnified.csv', index=False)


# process raw data
def process():

    # set up arguments, then call the function
    input_file = 'csv_files/EveryonesAppData.csv'

    output_file = 'csv_files/TimeBehavProf46.csv'  # needs to be a tuple when weekdays_split is True
    # output_file = ('csv_files/ofcom_10_weekdays.csv', 'csv_files/ofcom_10_weekends.csv')

    time_bins = 1440  # in minutes

    pop_apps = ['Calculator', 'Calendar', 'Camera', 'Clock', 'Contacts', 'Facebook',
                'Gallery', 'Gmail', 'Google Play Store', 'Google Search', 'Instagram',
                'Internet', 'Maps', 'Messaging', 'Messenger', 'Phone', 'Photos',
                'Settings', 'Twitter', 'WhatsApp', 'YouTube']

    weekdays_split = False

    z_scores = False

    day_lim = 7  # limit each person to only day_lim days of data

    user_lim = 778  # limit number of users, default: 10000 (i.e. no limit)

    selected_days = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7']

    agg = True

    # remember, caller has to clear output_file
    f = open(output_file, "w+")
    f.close()

    # f = open(output_file[0], "w+")
    # f.close()
    # f = open(output_file[1], "w+")
    # f.close()

    # Do the analysis
    DurationsPerApp(input_file, output_file, time_bins, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                    agg=agg).process_data()

if __name__ == "__main__":
    # unify_apps()
    process()