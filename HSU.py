# Script that generates time event matrices for HSU data
from process_data import process_data


if __name__ == "__main__":
    # set up arguments, then call the function
    input_file = 'csv_files/EveryonesAppData.csv'

    output_file = 'csv_files/TimeBehavProf46.csv'  # needs to be a tuple when weekdays_split is True
    # output_file = ('csv_files/Weekdays.csv', 'csv_files/Weekends.csv')

    time_bins = 1440  # in minutes

    # pop_apps = []
    pop_apps = ['Calculator', 'Calendar', 'Camera', 'Clock', 'Contacts', 'Facebook',
                'Gallery', 'Gmail', 'Google Play Store', 'Google Search', 'Instagram',
                'Internet', 'Maps', 'Messaging', 'Messenger', 'Phone', 'Photos',
                'Settings', 'Twitter', 'WhatsApp', 'YouTube']

    weekdays_split = False

    z_scores = False

    day_lim = 7  # limit each person to only day_lim days of data

    selected_days = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7']

    process_data(input_file, output_file, time_bins, pop_apps, weekdays_split, z_scores, day_lim)
