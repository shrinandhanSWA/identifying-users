# code for creating multiple binary classifiers, one for each user
# the big challenge is dealing with imbalanced data
# Option 1: Feed in everything, and then give class weights
# Option 2: pick a datapoint from another user for every datapoint (basically downsampling)
# Both have been implemented

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_curve, auc, accuracy_score
from tqdm import tqdm

TEST_DAYS = ['Day1', 'Day1-1', 'Day1-2', 'Day1-3', 'Day1-4', 'Day1-5']


def compute_acc_auc(y_true, y_pred, p_scores, sigm=False):
    """
    Compute the accuracy and the AUC of the ROC curve for the MIA's predicitions on a fixed target
    Input:
        y_true: true labels of the attacked groups
        y_pred: scores given as output of the classifier (all entries should be in [0,1])
        p_scores: probability scores of membership given by classifier
        OPTIONAL
        sigm: boolean determining if a sigmoid transformation should be applied to the p_scores to confine them in [0,1]
    Output:
        acc: Accuracy of the attack
        area: Area Under the ROC Curve
    """
    if sigm:
        p_scores = np.exp(p_scores)/(1+np.exp(p_scores))
    assert all(0<=y<=1 for y in p_scores), "the classifier scores should be within [0,1]"
    fpr, tpr, thresholds = roc_curve(y_true, p_scores)
    area = auc(fpr, tpr)
    acc = accuracy_score(y_true, y_pred)
    return acc, area


# -----------------------------------------------------
# OPTION 1
@ignore_warnings(category=ConvergenceWarning)
def create_classifier(user_data, other_data):

    # user_data needs to be transformed with person --> 1
    user_data = user_data.assign(Person=1)

    test_labels = []
    test_set = []
    training_labels = []
    training_set = []

    for row in user_data.itertuples():

        row = list(row)

        person = row[1]
        day = row[2]
        data = row[3:]

        if day in TEST_DAYS:
            test_labels.append(person)
            test_set.append(data)
        else:
            training_labels.append(person)
            training_set.append(data)

    # other_data needs to be transformed with person --> 0
    other_data = other_data.assign(Person=0)

    # downsampling attempt of training data of the majority class
    # other_test_data = other_test_data.iloc[:100]

    for row in other_data.itertuples():

        row = list(row)

        person = row[1]
        day = row[2]
        data = row[3:]

        if day in TEST_DAYS:
            test_labels.append(person)
            test_set.append(data)
        else:
            training_labels.append(person)
            training_set.append(data)

    # convert everything to numpy arrays
    training_set = np.array(training_set)
    training_labels = np.array(training_labels).ravel()
    test_set = np.array(test_set)
    test_labels = np.array(test_labels).ravel()

    # Shuffling training data
    seed = 1843
    np.random.seed(seed)
    np.random.shuffle(training_set)
    np.random.seed(seed)
    np.random.shuffle(training_labels)

    lr = LogisticRegression(class_weight='balanced')

    lr.fit(training_set, training_labels)

    predictions = lr.predict(test_set)
    p_scores = lr.predict_proba(test_set)[:, 1]

    # print(predictions)
    # print()
    # print('-----')

    # evaluate
    accuracy = compute_acc_auc(test_labels, predictions, p_scores, sigm=True)

    return accuracy, lr


def bin_list(data, bin_edges):

    hist, edges = np.histogram(data, bins=bin_edges)

    plt.hist(data, bins=bin_edges)

    # Annotate frequency above each bar
    bar_widths = np.diff(bin_edges)  # Calculate the width of each bar
    for i in range(len(hist)):
        x = edges[i] + (bar_widths[i] / 2)  # Calculate the x-coordinate for annotation
        plt.text(x, hist[i] + 3, str(hist[i]), ha='center', va='bottom', fontsize=15)


    plt.gca().set_facecolor((1, 1, 1, 0.9))

    plt.xlabel('AUC Score', fontsize=20, labelpad=15)
    plt.ylabel('Frequency', fontsize=20, labelpad=15)
    plt.title('Distribution of AUC scores', fontsize=30, pad=10)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.show()

    print("Histogram:", hist)
    print("Bin Edges:", edges)


# Runners
def go(f_path):
    input_data = pd.read_csv(f_path)

    # step 1: identify all the users
    users = list(set(input_data['Person']))

    user_accs_map = {u: (0, 0) for u in users}

    classifiers = []

    # step 2: for each user, split the data and train a classifier
    for user in tqdm(users):

        this_user = input_data[input_data['Person'] == user]
        other_user = input_data[input_data['Person'] != user]

        user_acc, classifier = create_classifier(this_user, other_user)

        classifiers.append(classifier)

        user_accs_map[user] = user_acc


    # step 3: evaluate results
    accs = [x for x, _ in user_accs_map.values()]
    aucs = [x for _, x in user_accs_map.values()]

    print(accs)
    print()
    print('---------------------------------')
    print()
    print(aucs)

    avg_acc = sum(accs) / len(accs)
    avg_auc = sum(aucs) / len(aucs)

    print(f'Average accuracy was {round(avg_acc * 100, 2)} & the average AUC was {round(avg_auc * 100, 2)}')

    bin_list(aucs, [0.4, 0.5, 0.9, 0.95, 0.99, 1])


if __name__ == "__main__":

    go('csv_files/ofcom_combined_target_times_5_weeks.csv')
