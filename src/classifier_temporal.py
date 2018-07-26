# --------------------------------------------------------
# Failing to Learn
# Copyright (c) 2017 FCAV University of Michigan
# Licensed under The MIT License [see LICENSE for details]
# Written by Manikandasriram S.R. and Cyrus Anderson
# --------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.externals import joblib


def write_kitti_results(results, output_folder, only_tp=False):
    print('writing KITTI style results of size {} to {}'.format(results.shape[0], output_folder))
    os.makedirs(output_folder, exist_ok=True)
    errors = dict()
    for index, row in results.iterrows():
        [label, p, x, y, w, h, fr, t, img_width, img_height, seq_name] = row[['valid', 'preds', 'x', 'y', 'w', 'h',
                                                                'frame', 'type', 'width', 'height', 'seq_name']]
        fr = int(fr)
        # if only_tp and (label == 0 or p < 0.5):
        #     continue
        x1 = x*img_width + img_width/2
        y1 = img_height/2 - y*img_width
        w1 = w*img_width
        h1 = h*img_width
        [x1, y1, x2, y2] = [x1-w1/2, y1-h1/2, x1+w1/2, y1+h1/2]
        err = [fr, -1, t, -1, -1, -1, x1, y1, x2, y2, -1, -1, -1, -1, -1, -1, -1, p]
        if seq_name in errors.keys():
            # append to errors
            errors[seq_name].append(err)
        else:
            # create the list
            errors[seq_name] = [err]
    for seq in errors.keys():
        if type(seq) == int:
            filename = os.path.join(output_folder, '{0:04d}.txt'.format(seq))
        else:
            filename = os.path.join(output_folder, '{}.txt'.format(seq))
        with open(filename, 'w') as f:
            for err in errors[seq]:
                f.write(' '.join(str(e) for e in err) + '\n')
    return errors


def evaluate_classifier(Y_test, preds, name='', color='b', linestyle='solid'):
    ap = average_precision_score(Y_test, preds)
    print(ap)
    precision, recall, thresholds = precision_recall_curve(Y_test, preds)

    plt.step(recall, precision, color=color, linestyle=linestyle, where='post', label='{name} AP={0:0.2f}'.format(
        ap, name=name))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    ind = np.where(preds >= 0.5)[0]
    tp = np.sum(Y_test[ind])
    fp = ind.shape[0]-tp
    fn = np.sum(Y_test)-tp
    print("Total mistakes predicted: {} Total mistakes present: {}".format(ind.shape[0], np.sum(Y_test)))
    print("TP: {} FP: {} FN: {}".format(tp, fp, fn))
    print("TP/(TP+FP): {}, TP/(TP+FN): {}".format(tp/(tp+fp), tp/(tp+fn)))


def train_classifier(train_hypotheses_df, test_hypotheses_df, model_filename, output_folder, show_impts=False,
                     show_pr=False):
    # column_names = ['x', 'y', 'w', 'h', 'r', 'det_cnt', 'med_det_ov', 'min_det_ov', 'max_det_ov', 'med_det_cnf',
    #                 'min_det_cnf', 'max_det_cnf', 'hyp_cnt', 'med_hyp_ov', 'min_hyp_ov', 'max_hyp_ov', 'med_hyp_cnf',
    #                 'min_hyp_cnf', 'max_hyp_cnf', 'n1', 'n2']
    column_names = ['x', 'y', 'w', 'h', 'r', 'det_cnt', 'med_det_ov', 'med_det_cnf', 'hyp_cnt', 'med_hyp_ov',
                    'med_hyp_cnf', 'n2']
    X_train_df = train_hypotheses_df[column_names]
    Y_train_df = train_hypotheses_df['valid'].astype(bool)
    X_train = X_train_df.as_matrix()
    Y_train = Y_train_df.as_matrix().reshape(-1)

    X_test_df = test_hypotheses_df[column_names]
    Y_test_df = test_hypotheses_df['valid'].astype(bool)

    X_test = X_test_df.as_matrix()
    Y_test = Y_test_df.as_matrix().reshape(-1)

    print("Training on a dataset of size: {}".format(X_train.shape[0]))

    # Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=30, class_weight="balanced_subsample")
    clf.fit(X_train, Y_train)

    if show_impts:
        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        keys = X_train_df.keys()
        for f in range(X_train.shape[1]):
            print("%d. feature %d %s (%f)" % (f + 1, indices[f], keys[indices[f]], importances[indices[f]]))

        fig = plt.figure(figsize=(7, 7))
        # Plot the feature importances of the clf
        plt.bar(np.arange(X_train.shape[1]), importances, yerr=std, color="g", align="center", width=0.5)
        plt.xticks(np.arange(X_train.shape[1]), keys, rotation=45)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()

    if show_pr:
        font = {'family': 'normal',
                'size': 18}
        matplotlib.rc('font', **font)
        fig = plt.figure(figsize=(8, 8))
        preds_baseline = np.ones(len(X_test_df))
        evaluate_classifier(Y_test, preds_baseline, name='RRC (Naive)', color='r', linestyle='dashed')
        preds = clf.predict_proba(X_test)[:, 1]
        evaluate_classifier(Y_test, preds, name='RRC (Classifier)', color='r')
        plt.legend()
        plt.show()

    if model_filename:
        joblib.dump(clf, model_filename)

    if output_folder:
        preds = clf.predict_proba(X_test)[:, 1]
        results = test_hypotheses_df.copy()
        results.loc[:, 'preds'] = preds
        write_kitti_results(results, output_folder)
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Binary classifier to predict if an hypothesis is a false negative '
                                                 'of the detector or not')
    parser.add_argument(
        '--train_dataset', help='list of CSV files for training the classifier', default=None, type=str, required=True)
    parser.add_argument(
        '--test_dataset', help='list of CSV files for evaluating the classifier', default=None, type=str, required=True)
    parser.add_argument(
        '--ignore_size', help='minimum height of the boxes to be included in training/testing', default=25, type=int)
    parser.add_argument(
        '--model_file', help="filename to save the trained model", type=str, required=True)
    parser.add_argument(
        '--show_pr', help='show precision-recall curve', default=True, type=bool)
    parser.add_argument(
        '--show_impts', help='show feature importances histogram', default=True, type=bool)
    parser.add_argument(
        '--output_folder', help="folder where missed detections will be written to in KITTI tracking format", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.train_dataset, 'r') as f:
        lines = f.readlines()
        train_files = [line.strip() for line in lines]

    with open(args.test_dataset, 'r') as f:
        lines = f.readlines()
        test_files = [line.strip() for line in lines]

    train_hypotheses_df = pd.DataFrame()
    for f in train_files:
        tmp_df = pd.read_csv(f)
        tmp_df = tmp_df[(tmp_df.h >= args.ignore_size/1920)]
        train_hypotheses_df = train_hypotheses_df.append(tmp_df)

    test_hypotheses_df = pd.DataFrame()
    for f in test_files:
        if os.path.exists(f):
            tmp_df = pd.read_csv(f)
            tmp_df = tmp_df[(tmp_df.h >= args.ignore_size/1920)]
            seq_name = os.path.basename(f).split('.')[0]
            tmp_df['seq_name'] = [seq_name]*len(tmp_df)
            test_hypotheses_df = test_hypotheses_df.append(tmp_df)

    train_classifier(train_hypotheses_df, test_hypotheses_df, args.model_file, args.output_folder, show_pr=args.show_pr,
                     show_impts=args.show_impts)