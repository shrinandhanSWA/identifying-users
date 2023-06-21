# Script that generates time event matrices for HSU data
from merge_datasets import compute_N_pop_apps
from process_data import DurationsPerApp, Fingerprint, PickupsPerApp
import pandas as pd


# Helper function to merge Messenger + Messaging, also Photos + Gallery
from rf_classifier import agg_classifier, line_plot


def unify_apps():
    original_hsu = pd.read_csv('csv_files/EveryonesAppData.csv')

    new_hsu = original_hsu.copy(deep=True)

    # list of all the renaming that needs to be done for HSU apps
    for row in new_hsu.itertuples():
        if row.event in [' Open Camera', 'SafeCamera']:
            new_hsu.at[row.Index, 'event'] = ' Camera'
        if row.event in [' Gallery', ' Photobooks']:
            new_hsu.at[row.Index, 'event'] = ' Photos'
        if row.event in [' Messaging', ' Messages']:
            new_hsu.at[row.Index, 'event'] = ' Messenger'
        if row.event in [' Chrome', ' Firefox', ' Samsung Internet']:
            new_hsu.at[row.Index, 'event'] = ' Internet'
        if row.event == ' Puzzle Alarm Clock':
            new_hsu.at[row.Index, 'event'] = ' Clock'
        if row.event == ' My Maps':
            new_hsu.at[row.Index, 'event'] = ' Maps'
        if row.event == ' Dialler':
            new_hsu.at[row.Index, 'event'] = ' Phone'
        if row.event in [' Audio settings', ' com.qualcomm.qti.networksetting', ' Settings Suggestions']:
            new_hsu.at[row.Index, 'event'] = ' Settings'
        if row.event == ' Email':
            new_hsu.at[row.Index, 'event'] = ' Inbox'

    new_hsu.to_csv('csv_files/EveryonesAppDataUnified.csv', index=False)


# process raw data
def process():
    # set up arguments, then call the function
    input_file = 'csv_files/EveryonesAppData.csv'

    output_file = 'csv_files/TimeBehavProf46AllApps.csv'  # needs to be a tuple when weekdays_split is True

    time_bins = 1440  # in minutes

    # original set of popular apps (21 - 2 that were merged)
    pop_apps = ['Calculator', 'Calendar', 'Camera', 'Clock', 'Contacts', 'Facebook',
                'Gallery', 'Gmail', 'Google Play Store', 'Google Search', 'Instagram',
                'Internet', 'Maps', 'Messaging', 'Messenger', 'Phone', 'Photos',
                'Settings', 'Twitter', 'WhatsApp', 'YouTube']

    weekdays_split = False

    z_scores = False

    day_lim = 5  # limit each person to only day_lim days of data

    user_lim = 778  # limit number of users, default: 10000 (i.e. no limit)

    selected_days = ['Day1', 'Day2', 'Day4', 'Day5', 'Day6']

    agg = True

    # remember, caller has to clear output_file
    f = open(output_file, "w+")
    f.close()

    # Do the analysis
    DurationsPerApp(input_file, output_file, time_bins, pop_apps, weekdays_split, z_scores, day_lim, user_lim=user_lim,
                    selected_days=selected_days, agg=agg).process_data()