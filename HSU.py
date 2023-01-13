# Script that generates time event matrices for HSU data
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.decomposition import PCA

MINS_IN_DAY = 1440


# Helper to perform compression on user data, TODO
def process_user_info(data):
    return data


# small helper func to add on new_data to input_df
def update_df(input_df, new_data):
    # process new_data (this is the user's data for all the days)
    new_data = process_user_info(new_data)

    new_df = pd.DataFrame(data=new_data)

    return pd.concat([input_df, new_df], ignore_index=True)


if __name__ == "__main__":

    input = pd.read_csv('EveryonesAppData.csv')

    # temporary cropping
    # input = input.iloc[:10]

    # first step: identify all the apps
    all_apps = sorted(list(set(input['event'])))  # sorting for consistency
    # removing leading spaces
    all_apps = [x.strip() for x in all_apps]

    # popular apps
    pop_apps = ['Calculator', 'Calendar', 'Camera', 'Clock', 'Contacts', 'Facebook',
                'Gallery', 'Gmail', 'Google Play Store', 'Google Search', 'Instagram',
                'Internet', 'Maps', 'Messaging', 'Messenger', 'Phone', 'Photos',
                'Settings', 'Twitter', 'WhatsApp', 'YouTube']

    all_apps = pop_apps  # popular apps only (for testing)

    # all time bins - split into weekdays and weekends
    # granularity of time bins is determined by the time_gran variable (min)

    time_gran = 1440  # e.g. an hour (60 min)
    no_bins = MINS_IN_DAY // time_gran
    weekend_bins = []
    weekday_bins = []
    for app in all_apps:
        for time in range(no_bins):
            for day in range(7):  # day 0: monday, ..., day 6: sunday
                bin_name = app + '-' + str(day) + '-' + str(time)
                if day < 5:
                    weekday_bins.append(bin_name)
                else:
                    weekend_bins.append(bin_name)

    # helper variables
    curr_user = None
    curr_user_weekday_dict = {}
    curr_user_weekend_dict = {}
    curr_user_start_day = 0
    curr_user_day_count = 0

    # overall dataframe for all users (will be built up) - weekday and weekend info separated
    users_weekday_df = pd.DataFrame(data=curr_user_weekday_dict)
    users_weekend_df = pd.DataFrame(data=curr_user_weekend_dict)

    # iterate through all the inputs
    for row in input.itertuples():
        user = row.participantnumber
        timestamp = row.timestamp
        app = row.event.strip()
        duration = row.duration

        if app not in pop_apps:
            continue

        if user != curr_user:
            # moved on to next user, so save previous user's information
            users_weekday_df = update_df(users_weekday_df, curr_user_weekday_dict)
            users_weekend_df = update_df(users_weekend_df, curr_user_weekend_dict)

            # set the rolling parameters for the new user
            curr_user = user
            curr_user_weekday_dict = {b: [0] for b in weekday_bins}
            curr_user_weekend_dict = {b: [0] for b in weekend_bins}

        # calculate day and hour of this event
        curr_event_time = datetime.fromtimestamp(int(timestamp))
        curr_event_day = datetime.weekday(curr_event_time)

        # determine which time bin the current time is in
        curr_event_min = curr_event_time.hour * 60 + curr_event_time.minute
        curr_event_bin = curr_event_min // time_gran

        # create bin name
        bin_name = app + '-' + str(curr_event_day) + '-' + str(curr_event_bin)

        if curr_event_day in [5, 6]:  # weekend
            new_duration = curr_user_weekend_dict.get(bin_name, [0])[0] + duration
            curr_user_weekend_dict[bin_name] = [new_duration]
        else:  # weekday
            new_duration = curr_user_weekday_dict.get(bin_name, [0])[0] + duration
            curr_user_weekday_dict[bin_name] = [new_duration]

    # updating info of the last user
    users_weekday_df = update_df(users_weekday_df, curr_user_weekday_dict)
    users_weekend_df = update_df(users_weekend_df, curr_user_weekend_dict)

    # ensure no NaNs are present - might be redundant
    users_weekday_df.fillna(0)  # temporary code
    users_weekend_df.fillna(0)  # temporary code

    # apply PCA - potential alternatives: multi dimensional scaling (MDS)/t-sne plot
    users_df_cols = users_weekday_df.columns
    data = users_weekday_df.to_numpy()
    # issue: the input (data) seems to have NaNs, idk how it happened, but turn it into 0s
    pca = PCA(n_components=46)  # TODO: PCA on per user stuff, so each user's day matrix looks different than before
    pca.fit(data)
    new_data = pca.singular_values_

    compressed_df = pd.DataFrame(data=new_data)
    compressed_df.fillna(0)

    # users_weekday_df = compressed_df  # to save the compressed version

    print('saving to CSV')

    # save both dataframes to separate CSVs
    users_weekday_df.to_csv('output_weekday.csv')
    users_weekend_df.to_csv('output_weekend.csv')
