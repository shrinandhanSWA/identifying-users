# Python file that has various statistical analyses

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


def hsu_day_distribution():
    input = pd.read_csv('csv_files/EveryonesAppData.csv')

    day_spreads = [[] for _ in range(7)]
    curr_user = None
    curr_user_days_seen = []
    day_count = 0

    for row in input.itertuples():

        if row.participantnumber != curr_user:
            curr_user = row.participantnumber
            day_count = 1
            curr_user_days_seen = []


        timestamp = row.timestamp
        timestamp_datetime = datetime.fromtimestamp(int(timestamp))
        actual_day = timestamp_datetime.day
        day_of_week = datetime.weekday(timestamp_datetime)
        # day_of_week - 0 --> Monday, ... , 6 --> Sunday

        # Days have multiple entries, so need to consider only first entry of the day
        if actual_day in curr_user_days_seen:
            continue

        # only want to consider the first 7 days
        if day_count > 7:
            continue

        curr_user_days_seen.append(actual_day)
        day_spreads[day_count - 1].append(day_of_week)
        day_count += 1

    # ------------------------------------

    input = pd.read_csv('csv_files/EveryonesAppData.csv')

    weekday_day_spreads = [[] for _ in range(7)]
    curr_user = None
    curr_user_days_seen = []
    day_count = 0

    for row in input.itertuples():

        if row.participantnumber != curr_user:
            curr_user = row.participantnumber
            day_count = 1
            curr_user_days_seen = []

        timestamp = row.timestamp
        timestamp_datetime = datetime.fromtimestamp(int(timestamp))
        actual_day = timestamp_datetime.day
        day_of_week = datetime.weekday(timestamp_datetime)
        # day_of_week - 0 --> Monday, ... , 6 --> Sunday

        # Days have multiple entries, so need to consider only first entry of the day
        if actual_day in curr_user_days_seen:
            continue

        # only want to consider the first 7 days
        if day_count > 7:
            continue

        # ignore weekends
        if day_of_week in [5, 6]:
            continue

        curr_user_days_seen.append(actual_day)
        weekday_day_spreads[day_count - 1].append(day_of_week)
        day_count += 1


    fig, axs = plt.subplots(3, 3)
    axs[0, 0].hist(day_spreads[0], label='all days')
    axs[0, 0].hist(weekday_day_spreads[0], label='weekdays')
    axs[0, 0].legend(prop={'size': 5})
    axs[0, 0].set_title('Day1', size=10)
    axs[0, 1].hist(day_spreads[1], label='all days')
    axs[0, 1].hist(weekday_day_spreads[1], label='weekdays')
    axs[0, 1].legend(prop={'size': 5})
    axs[0, 1].set_title('Day2', size=10)
    axs[0, 2].hist(day_spreads[2], label='all days')
    axs[0, 2].hist(weekday_day_spreads[2], label='weekdays')
    axs[0, 2].legend(prop={'size': 5})
    axs[0, 2].set_title('Day3', size=10)
    axs[1, 0].hist(day_spreads[3], label='all days')
    axs[1, 0].hist(weekday_day_spreads[3], label='weekdays')
    axs[1, 0].legend(prop={'size': 5})
    axs[1, 0].set_title('Day4', size=10)
    axs[1, 1].hist(day_spreads[4], label='all days')
    axs[1, 1].hist(weekday_day_spreads[4], label='weekdays')
    axs[1, 1].legend(prop={'size': 5})
    axs[1, 1].set_title('Day5', size=10)
    axs[1, 2].hist(day_spreads[5], label='all days')
    axs[1, 2].hist(weekday_day_spreads[5], label='weekdays')
    axs[1, 2].legend(prop={'size': 5})
    axs[1, 2].set_title('Day6', size=10)
    axs[2, 0].hist(day_spreads[6], label='all days')
    axs[2, 0].hist(weekday_day_spreads[6], label='weekdays')
    axs[2, 0].legend(prop={'size': 5})
    axs[2, 0].set_title('Day7', size=10)
    plt.show()


if __name__ == "__main__":
    hsu_day_distribution()
