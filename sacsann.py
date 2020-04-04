#! /usr/bin/env python3
# -*- coding: utf8 -*-
import argparse
import os
import sys

from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import process_data

N_FEATURES = 100
N_LAYER_INT = 1
N_NODES_INT = 256
ALPHA_INT = 0.001
LEARNING_RATE_INT = 0.0001
N_LAYER_TOP = 2
N_NODES_TOP = (64, 64)
ALPHA_TOP = 0.001
LEARNING_RATE_TOP = 0.01


def parse_arguments():
    """
    Parse the arguments from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "labels_path",
        help="Path to directory containing per chromosomes compartment labels",
    )
    parser.add_argument(
        "features_path", help="Path to directory containing per chromosome features",
    )
    parser.add_argument("genome", help="Genome to analyze")
    parser.add_argument(
        "--train_chromosomes",
        default=[],
        nargs="+",
        help="Training chromosomes, between 1 and 19 for mouse genome and 1 and 22 for"
        " human genome. If none is given, SACCSANN will be trained on the whole genome.",
    )
    parser.add_argument(
        "--test_chromosomes",
        default=[],
        nargs="+",
        help="Test chromosome, between 1 and 19 for mouse genome and 1 and 22 for human "
        "genome. If none is given, SACCSANN will not be tested.",
    )
    parser.add_argument(
        "--output_folder",
        default="output",
        help="Path to the directory where to store SACSANN results",
    )
    parser.add_argument(
        "-s",
        "--save_model",
        type=bool,
        default=False,
        help="""Whether or not to save the trained model""",
    )
    return parser.parse_args()


def predict_compartments(
    features_directory,
    train_chromosomes,
    test_chromosomes,
    test_chromosomes_length,
    X_train,
    y_train,
    X_test,
    y_test,
    a_indexes,
    b_indexes,
    scaler,
    output_folder,
    save_model=False,
):
    """
    Train SACCSANN architecture for the provided training and test data

    Args:
        features_directory (str): directory with one features file per chromosome
        train_chromosomes (List[int]): list of chromosomes used for training
        test_chromosomes (List[int]): number of the test chromosome if provided by the
            user. Default values is 0 and SACCANN is trained on the whole genome
        test_chromosomes_length:
        X_train (List[List]): training features
        y_train (List[int]): binary training labels (1: A bin, 0: B bin)
        X_test (List[List]): test features
        y_test (List[int]): binary test labels (1: A bin, 0: B bin)
        a_indexes (List): if the 'balance_class' variable is True, contains A bin
            indexes used for training the network. Otherwise empty
        b_indexes (List): same for B bins
        scaler (sklearn.preprocessing): scaler used for feature normalization
        output_folder (str): path to the output folder
        save_model (bool): whether or not to save current model

    Returns:
        Opt[float]: AUC score on test data, if provided
    """
    logger.info("Training the intermediate network...")
    intermediate_classifier = MLPClassifier(
        activation="logistic",
        solver="adam",
        alpha=ALPHA_INT,
        max_iter=500,
        shuffle=True,
        tol=0.00001,
        learning_rate_init=LEARNING_RATE_INT,
        random_state=1,
        hidden_layer_sizes=N_NODES_INT,
    )
    intermediate_classifier.fit(X_train, y_train)

    logger.info("Extracting probability predictions...")
    X_train_proba, X_test_proba = process_data.build_proba_features(
        features_directory,
        train_chromosomes,
        test_chromosomes,
        scaler,
        intermediate_classifier,
        a_indexes,
        b_indexes,
    )
    scaler_top = StandardScaler()
    X_train_proba = scaler_top.fit_transform(X_train_proba)

    logger.info("Training the smoothing network...")
    final_classifier = MLPClassifier(
        activation="logistic",
        solver="adam",
        alpha=ALPHA_TOP,
        max_iter=500,
        shuffle=True,
        tol=0.00001,
        learning_rate_init=LEARNING_RATE_TOP,
        random_state=1,
        hidden_layer_sizes=N_NODES_TOP,
    )
    final_classifier.fit(X_train_proba, y_train)
    logger.info("Done")

    if save_model:
        process_data.write_model(
            intermediate_classifier, os.path.join(output_folder, "mlp_int_weights.csv")
        )
        process_data.write_model(
            final_classifier, os.path.join(output_folder, "mlp_top_weights.csv")
        )

    if X_test != []:
        y_pred_int = intermediate_classifier.predict(X_test)
        X_test_proba = scaler_top.fit_transform(X_test_proba)
        y_pred_top = final_classifier.predict(X_test_proba)
        logger.info(f"Intermediate AUC score: {roc_auc_score(y_test, y_pred_int)}")
        logger.info(f"Final smoothed AUC score: {roc_auc_score(y_test, y_pred_top)}")

        # Save predictions
        process_data.write_predictions(
            y_pred_top[0 : test_chromosomes_length[0]],
            os.path.join(output_folder, f"chr{test_chromosomes[0]}_predictions.csv"),
        )
        cum_length = test_chromosomes_length[0]
        for i in range(1, len(test_chromosomes_length)):
            chr_test = test_chromosomes[i]
            process_data.write_predictions(
                y_pred_top[cum_length : cum_length + test_chromosomes_length[i]],
                os.path.join(output_folder, f"chr{chr_test}_predictions.csv"),
            )
            cum_length += test_chromosomes_length[i]
        return roc_auc_score(y_test, y_pred_top)


def run_sacssan(args):
    """
    Run SACSANN
    """
    train_chromosomes = [int(i) for i in args.train_chromosomes[0].split(",")]
    if args.test_chromosomes != []:
        test_chromosomes = [int(i) for i in args.test_chromosomes[0].split(",")]
    else:
        test_chromosomes = []
    chromosomes = train_chromosomes + test_chromosomes
    possible_chrs = process_data.get_chromosome_list(args.genome)

    for i in range(len(chromosomes)):
        if chromosomes[i] not in possible_chrs:
            logger.warning(
                f"Unvalid chromosome, "
                f"possible chromosomes for the input genome are {possible_chrs}"
            )
            sys.exit()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    (
        X_train,
        y_train,
        X_test,
        y_test,
        A_indexes,
        B_indexes,
        scaler,
        testChrLen,
    ) = process_data.get_data(
        args.labels_path,
        args.features_path,
        train_chromosomes,
        test_chromosomes,
        scaling=True,
        balance=True,
    )
    predict_compartments(
        args.features_path,
        train_chromosomes,
        test_chromosomes,
        testChrLen,
        X_train,
        y_train,
        X_test,
        y_test,
        A_indexes,
        B_indexes,
        scaler,
        args.output_folder,
        args.save_model,
    )


if __name__ == "__main__":
    args = parse_arguments()
    run_sacssan(args)
