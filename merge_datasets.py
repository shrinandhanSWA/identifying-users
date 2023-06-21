# merging datasets together
# for now, it's HSU and Ofcom

import pandas as pd
from process_data import DurationsPerApp, ProcessData


# To be excluded from popular apps calculation
SYSTEM_APPS = ['System UI-Dur-1', 'Samsung Experience Home-Dur-1', 'TouchWiz Home-Dur-1', 'Xperia Home-Dur-1',
               'Android System-Dur-1', 'Nova Launcher-Dur-1', 'x-Dur-1', 'Pixel Launcher-Dur-1,'
                'org.pielot.sns-Dur-1', '-android-Dur-1', 'com.android.keyguard-Dur-1'
               '+com.sec.android.app.launcher-Dur-1', 'com.cleanmaster.security-Dur-1',
               '+com.android.launcher3-Dur-1', 'com.huawei.systemmanager-Dur-1', 'org.pielot.sns2-Dur-1',
               '-com.android.systemui-Dur-1', '=com.android.inputmethod.latin-Dur-1',
               'org.pielot.sns-Dur-1', 'com.android.keyguard-Dur-1',
               '+com.sec.android.app.launcher-Dur-1', '+com.android.launcher-Dur-1',
               '+com.huawei.android.launcher-Dur-1', '+com.miui.home-Dur-1', '+com.sonyericsson.home-Dur-1',
               'com.sec.android.app.sbrowser-Dur-1', 'com.cleanmaster.mguard-Dur-1', 'com.kingroot.kinguser-Dur-1',
               'com.android.dreams.basic-Dur-1', 'com.lge.shutdownmonitor-Dur-1', '+com.android.launcher5.bird-Dur-1',
               '=com.google.android.inputmethod.latin-Dur-1']


# generate required data
def generate_data():
    processor = DurationsPerApp('input_file', 'output_file', 1440, [], False, False, agg=True)

    # -------------------------------------------------------------------------------------------

    # OFCOM!!!
    # set up arguments, then call the function
    input_file = 'csv_files/ofcom_processed.csv'
    output_file = 'csv_files/ofcom_merge.csv'
    user_lim = 732  # limit number of users, default: 10000 (i.e. no limit)

    f = open(output_file, "w+")
    f.close()

    processor.input_file = input_file
    processor.output_file = output_file
    processor.user_lim = user_lim

    processor.process_data()

    # -------------------------------------------------------------------------------------------
    # HSU!!!

    # set up arguments, then call the function
    input_file = 'csv_files/EveryonesAppData.csv'
    output_file = 'csv_files/hsu_merge.csv'

    f = open(output_file, "w+")
    f.close()

    processor.input_file = input_file
    processor.output_file = output_file

    processor.process_data()


# merge raw data - deal with app merge conflicts
def merge_raw_data():
    # load in required csv files (generated by generate_data)
    hsu = pd.read_csv('csv_files/hsu_merge.csv')
    ofcom = pd.read_csv('csv_files/ofcom_merge.csv')

    # Merging requires the same apps.txt but with different names (hereby merge conflicts) to be resolved
    # Since OFCOM only has 19, I only have to deal with these 19 (will need to be expanded later)
    # OFCOM apps.txt: BBC News, Facebook, Gmail, Inbox, Instagram, Internet, Maps, Messenger, Nova Launcher,
    # Outlook, Photos, Snapchat, Spotify, Twitter, WhatsApp, Yahoo Mail, YouTube, eBay
    # Conflicts: Inbox --> Email(?), Internet --> Samsung Internet,
    # Maps --> Maps (not My Maps, just an interesting case)
    # Another note: The names are actually Inbox-Bin1, so need to tackle this

    # calculate bins per day
    bins_per_day = len(ofcom.count()) // 18  # OFCOM only has 18 apps.txt
    app_names = {'Inbox': 'Email', 'Internet': 'Samsung Internet'}
    app_map = {}

    for i in range(1, bins_per_day + 1):
        for old_name, new_name in app_names.items():
            app_map[f'{old_name}-Bin{i}'] = f'{new_name}-Bin{i}'

    # Manually assign any OFCOM app that has a different name in the HSU data
    # Rename columns in ofcom
    ofcom.rename(columns=app_map, inplace=True)

    # Now, actually do the merging, which is actually just a concat operation
    merged = pd.concat([hsu, ofcom], axis=0, ignore_index=True)

    # 2 things now
    # 1. turn NAs to 0
    merged = merged.fillna(0)

    # 2. The indices will mess up, we need one index from 1 to N (merging breaks this smooth index)
    # first, make sure the tuples are increasing from 1 to N, then use set_index
    user_index = 0
    curr_user = -1  # never going to be possible
    day_index = 1

    for i, row in enumerate(merged.itertuples()):
        if curr_user != row.Person:
            curr_user = row.Person
            user_index += 1
            day_index = 1
        curr_day = 'Day' + str(day_index)
        merged.at[i, 'Person'] = user_index
        merged.at[i, 'Day'] = curr_day
        day_index += 1

    merged.set_index(['Person', 'Day'], inplace=True)

    # finally, save the merged data
    merged.to_csv('csv_files/merged_data.csv')


# recompute the most popular apps
# TODO: need to sum up if there are multiple bins for the same day
# the current implementation assumes there is only one bin per day per app
def compute_N_pop_apps(no_apps, data_path):
    merged = pd.read_csv(data_path)

    app_durations = {}

    for col_name in merged:

        if col_name in ['Person', 'Day']:  # skip the indices
            continue

        if col_name in SYSTEM_APPS:  # skip system apps
            continue

        col = merged[col_name]
        total_duration = int(col.mean())



        app_durations[col_name] = total_duration


    sorted_durations = sorted(app_durations.items(), key=lambda x: x[1], reverse=True)

    top_apps = [x.split('-')[0] for x, _ in sorted_durations[:no_apps]]

    return top_apps


# get rid of unpopular apps.txt + calculate z scores
def crop_pop_apps(pop_apps, data_path):
    # also get the z scores at the same time
    # for some reason, loading up the merged data introduces its own index
    merged = pd.read_csv(data_path, index_col=[0, 1])

    to_be_removed = []

    for col_name in merged:
        if col_name in ['Person', 'Day']:
            continue
        if col_name not in pop_apps:
            to_be_removed.append(col_name)

    pop_apps_only = merged.drop(columns=to_be_removed)

    pop_apps_only = pop_apps_only.fillna(0)

    pop_apps_z = ProcessData.calc_z_scores(pop_apps_only)

    # save
    pop_apps_z.to_csv('csv_files/merged.csv')


def go(gen_data=True, no_apps=18):
    if gen_data:
        generate_data()

    merge_raw_data()

    pop_apps = compute_N_pop_apps(no_apps, 'csv_files/merged_data.csv')

    print(f'The popular apps are: {pop_apps}')

    crop_pop_apps(pop_apps, 'csv_files/merged_data.csv')


if __name__ == "__main__":
    go(True, 18)
