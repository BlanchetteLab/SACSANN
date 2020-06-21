"""
Script to read and format data for SACCSANN training
"""

import csv
import os
import pickle
import random

from loguru import logger
from sklearn.preprocessing import StandardScaler

N_NEIGHBORS = 10


def get_chromosome_list(genome):
    """
    Get the list of chromosomes corresponding to the input genome excluding sex
    chromosomes.
    Args:
        genome (str): genome of interest, mouse "mm" and human "hg" are the only
            available as of now

    Returns:
        List[int]: list of chromosomes number
    """
    if genome.startswith("mm"):
        chromosomes = [i for i in range(1, 20)]
    elif genome.startswith("hg"):
        chromosomes = [i for i in range(1, 23)]
    else:
        logger.info("Sorry, this code can only handle mouse and human genomes for now.")
        return []
    return chromosomes


def balance_classes(X, y):
    """
    Balance the two classes in the input data set.

    Args:
        X (List[List]): features
        y (List): corresponding labels.

    Returns:
        X_new (List[List]): balanced randomly selected bins' features
        y_new (List): corresponding labels
        selected_A (List): indexes of the selected A bins
        selected_B (List): indexes of the selected B bins
    """
    # Count number of A and B bins
    n_A = 0
    n_B = 0
    index_A = []
    index_B = []
    for i in range(len(y)):
        if y[i] == 1:
            n_A += 1
            index_A.append(i)
        else:
            n_B += 1
            index_B.append(i)
    min_class = min(
        n_A, n_B
    )  # take the minimum to be the number of training samples per class

    # Randomly sample in the list indices
    random.seed(2)
    train_A = random.sample(range(len(index_A)), min_class)
    train_B = random.sample(range(len(index_B)), min_class)
    selected_A = [index_A[train_A[i]] for i in range(len(train_A))]
    selected_B = [index_B[train_B[i]] for i in range(len(train_B))]
    X_new = []
    y_new = []
    for i in range(min_class):
        X_new.append(X[index_A[train_A[i]]])
        X_new.append(X[index_B[train_B[i]]])
        y_new.append(y[index_A[train_A[i]]])
        y_new.append(y[index_B[train_B[i]]])
    return X_new, y_new, selected_A, selected_B


def format_test_data(features_directory, chromosomes, scaler=None):
    X = []
    chromosomes_length = []
    for chromosome in chromosomes:
        features_file = os.path.join(features_directory, f"chr{chromosome}.csv")
        feature_data = list(csv.reader(open(features_file, "r"), delimiter=","))
        number_of_bins = 0
        for row in feature_data:
            number_of_bins += 1
            X.append([float(row[i]) for i in range(len(row))])
        chromosomes_length.append(number_of_bins)
    # Data normalization
    if scaler is not None:
        X = scaler.transform(X)
    return X, chromosomes_length


def get_data(
    labels_directory,
    features_directory,
    train_chromosomes,
    test_chromosomes,
    scaling,
    balance,
    save_model=False,
    output_folder=None,
):
    """
    Read and process input data
    """
    X_train, y_train, X_test, y_test = [], [], [], []
    chromosomes = train_chromosomes + test_chromosomes
    testchrlen = []

    for chromosome in chromosomes:
        labels_file = os.path.join(labels_directory, f"chr{chromosome}.csv")
        label_data = list(csv.reader(open(labels_file, "r"), delimiter=","))
        features_file = os.path.join(features_directory, f"chr{chromosome}.csv")
        feature_data = list(csv.reader(open(features_file, "r"), delimiter=","))

        if chromosome in train_chromosomes:
            for row in label_data:
                y_train.append(int(row[0]))
            for row in feature_data:
                X_train.append([float(row[i]) for i in range(len(row))])
        else:
            l = 0
            for row in label_data:
                l += 1
                y_test.append(int(row[0]))
            for row in feature_data:
                X_test.append([float(row[i]) for i in range(len(row))])
            testchrlen.append(l)
    # Data normalization
    if scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if save_model:
            pickle.dump(scaler, open(os.path.join(output_folder, "scaler.p"), "wb"))
        if X_test != []:
            X_test = scaler.transform(X_test)
    else:
        scaler = None

    # Balance A and B bins in the training set
    if balance:
        X_train, y_train, A_indexes, B_indexes = balance_classes(X_train, y_train)
    else:
        A_indexes, B_indexes = [], []

    return X_train, y_train, X_test, y_test, A_indexes, B_indexes, scaler, testchrlen


def get_proba_features(nb_neighbors, probas, X):
    """
    Build the probability vectors for the input probas vector.
    Each bin is assigned a (2*nb_neighbors + 1)-long vector which is added to the input
    X matrix.
    """
    vec_len = 2 * nb_neighbors + 1
    for i in range(nb_neighbors):
        line = []
        for k in range(nb_neighbors - i):
            line.append(0.0)
        k = 0
        while i - k > 0:
            line.append(probas[k][0])
            k += 1
        line.append(probas[i][0])
        j = 1
        while j <= nb_neighbors:
            line.append(probas[i + j][0])
            j += 1
        X.append(line)

    for i in range(nb_neighbors, len(probas) - nb_neighbors):
        line = []
        j = nb_neighbors
        while j > 0:
            line.append(probas[i - j][0])
            j -= 1
        line.append(probas[i][0])
        j = 1
        while j <= nb_neighbors:
            line.append(probas[i + j][0])
            j += 1
        X.append(line)

    for i in range(len(probas) - nb_neighbors, len(probas)):
        line = []
        j = nb_neighbors
        while j > 0:
            line.append(probas[i - j][0])
            j -= 1
        line.append(probas[i][0])
        j = 1
        while (i + j) < len(probas):
            line.append(probas[i + j][0])
            j += 1
        while len(line) < vec_len:
            line.append(0.0)
        X.append(line)
    return X


def build_proba_features(
    features_directory,
    train_chromosomes,
    test_chromosomes,
    scaler,
    intermediate_classifier,
    a_indexes,
    b_indexes,
):
    """
    Build features for the smoothing network given the output of the intermediate
    network
    """
    X_proba_train = []
    X_proba_test = []
    chromosomes = train_chromosomes + test_chromosomes
    for chromosome in chromosomes:
        X = []
        features_file = os.path.join(features_directory, f"chr{chromosome}.csv")
        feature_data = list(csv.reader(open(features_file, "r"), delimiter=","))
        for row in feature_data:
            X.append([float(row[i]) for i in range(len(row))])
        X = scaler.transform(X)
        probas = intermediate_classifier.predict_proba(X)
        if chromosome not in test_chromosomes:
            X_proba_train = get_proba_features(N_NEIGHBORS, probas, X_proba_train)
        else:
            X_proba_test = get_proba_features(N_NEIGHBORS, probas, X_proba_test)

    if len(a_indexes) != 0:
        X_train_balanced = []
        for i in range(len(a_indexes)):
            X_train_balanced.append(X_proba_train[a_indexes[i]])
            X_train_balanced.append(X_proba_train[b_indexes[i]])
        return X_train_balanced, X_proba_test
    else:
        return X_proba_train, X_proba_test


def write_predictions(y_pred, output_file):
    """
    Write input predictions to csv file
    """
    with open(output_file, "w") as file:
        writer = csv.writer(
            file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for i in range(len(y_pred)):
            writer.writerow([str(y_pred[i])])
