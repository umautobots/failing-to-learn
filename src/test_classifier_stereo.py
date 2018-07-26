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
import time

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.externals import joblib


def write_kitti_results(results, output_folder, only_tp=False):
    print('writing KITTI style results of size {} to {}'.format(results.shape[0], output_folder))
    os.makedirs(output_folder, exist_ok=True)
    errors = dict()
    for index, row in results.iterrows():
        [label, p, x, y, w, h, fr, t, img_width, img_height, seq_name] = row[['labels', 'preds', 'x', 'y', 'w', 'h',
                                                                'frame', 'type', 'width', 'height', 'seq_name']]
        fr = int(fr)
        if only_tp and (label == 0 or p < 0.5):
            continue
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

    plt.step(recall, precision, color=color, linestyle=linestyle, where='post', label='{name} {0:0.2f}'.format(ap,
        name=name))
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


def parse_args():
    parser = argparse.ArgumentParser(description='Use the trained classifier to make predictions on a test set')
    parser.add_argument(
        '--test_dataset', help='list of CSV files for evaluating the classifier', default=None, type=str, required=True)
    parser.add_argument(
        '--ignore_size', help='minimum height of the boxes to be included in training/testing', default=25, type=int)
    parser.add_argument(
        '--model_file', help="filename to save the trained model", type=str, required=True)
    parser.add_argument(
        '--show_pr', help='show precision-recall curve', default=False, type=bool)
    parser.add_argument(
        '--output_folder', help="folder where missed detections will be written to in KITTI tracking format", type=str)
    parser.add_argument(
        '--width', help="width in pixels to normalize coordinates", default=1242, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.test_dataset, 'r') as f:
        lines = f.readlines()
        test_files = [line.strip() for line in lines]

    test_hypotheses_df = pd.DataFrame()
    for f in test_files:
        if os.path.exists(f):
            tmp_df = pd.read_csv(f)
            tmp_df = tmp_df[(tmp_df.h >= args.ignore_size/args.width)]
            seq_name = os.path.basename(f).split('.')[0]
            tmp_df['seq_name'] = [seq_name]*len(tmp_df)
            test_hypotheses_df = test_hypotheses_df.append(tmp_df)

    column_names = ['x', 'y', 'w', 'h', 'r', 'det_cnt', 'med_det_ov', 'med_det_cnf', 'hyp_cnt', 'med_hyp_ov',
                    'med_hyp_cnf']
    X_test_df = test_hypotheses_df[column_names]
    X_test = X_test_df.as_matrix()

    clf = joblib.load(args.model_file)

    font = {'family' : 'normal',
            'size'   : 18}

    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(7, 7))
    if args.show_pr:
        Y_test_df = test_hypotheses_df['valid'].astype(bool)
        Y_test = Y_test_df.as_matrix().reshape(-1)

        preds_baseline = np.ones(len(X_test_df))
        evaluate_classifier(Y_test, preds_baseline, name='RRC (Naive)', color='b', linestyle='dashed')
        preds = clf.predict_proba(X_test)[:, 1]
        evaluate_classifier(Y_test, preds, name='RRC (Classifier)', color='b')
        plt.legend()
        plt.show()

    if args.output_folder:
        preds = clf.predict_proba(X_test)[:, 1]
        results = test_hypotheses_df.copy()
        results.loc[:, 'preds'] = preds
        write_kitti_results(results, args.output_folder)