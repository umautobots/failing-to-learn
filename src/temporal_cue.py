# --------------------------------------------------------
# Failing to Learn
# Copyright (c) 2017 FCAV University of Michigan
# Licensed under The MIT License [see LICENSE for details]
# Written by Manikandasriram S.R. and Cyrus Anderson
# --------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from utils import load_detections, bbox_overlaps, pix_to_xy


def add_labels_temporal(dets_df, hyp_df, gt_df, fr, ov_thresh=0.5):
    dc_df = gt_df[gt_df.type == 'DontCare']
    gt_df = gt_df[gt_df.type != 'DontCare']
    gt_df = gt_df.set_index(np.arange(len(gt_df)))

    # use gt to determine missed detections
    dets_bbox = dets_df[['bb_left', 'bb_top', 'bb_right', 'bb_bottom']].values
    gt_bbox = gt_df[['bb_left', 'bb_top', 'bb_right', 'bb_bottom']].values
    overlaps = bbox_overlaps(gt_bbox, dets_bbox)
    C = 1 - overlaps
    COST_MAX = 1e9
    C[overlaps <= ov_thresh] += COST_MAX
    [row, col] = linear_sum_assignment(C)
    inds_valid = C[row, col] < COST_MAX

    # use gt to determine valid hypotheses
    hyp_bbox = hyp_df[['bb_left', 'bb_top', 'bb_right', 'bb_bottom']].values
    hyp_overlaps = bbox_overlaps(hyp_bbox, gt_bbox)
    C_hyp = 1 - hyp_overlaps
    C_hyp[hyp_overlaps <= ov_thresh] += COST_MAX
    [row_h, col_h] = linear_sum_assignment(C_hyp)
    inds_valid_hyp = C_hyp[row_h, col_h] < COST_MAX

    labels = np.zeros(len(hyp_df))

    # if associated to ground truth, verify that it was not detected
    for i, r in enumerate(row_h):
        if not inds_valid_hyp[i]:
            continue
        if col_h[i] not in row[inds_valid]:
            labels[r] = True
    if len(dc_df) and len(hyp_df):
        dc_bbox = dc_df[['bb_left', 'bb_top', 'bb_right', 'bb_bottom']].values
        dc_overlaps = bbox_overlaps(dc_bbox, hyp_bbox, 'b')
        for index, hyp in hyp_df.iterrows():
            if np.max(dc_overlaps[:, index]) >= 0.5 and labels[index] == 0:
                labels[index] = -1
    hyp_df['valid'] = labels
    hyp_df = hyp_df[hyp_df.valid >= 0]
    return hyp_df


def extract_temporal_hypotheses(dets_df, tracks_df, width, height, ov_thresh=0.5):
    # ensure indices of dataframes are [0,n]
    dets_df = dets_df.set_index(np.arange(len(dets_df)))
    tracks_df = tracks_df.set_index(np.arange(len(tracks_df)))

    # use hungarian to determine inconsistencies
    dets_bbox = dets_df[['bb_left', 'bb_top', 'bb_right', 'bb_bottom']].values
    tracks_bbox = tracks_df[['bb_left', 'bb_top', 'bb_right', 'bb_bottom']].values
    # clip tracks to image size
    np.clip(tracks_bbox[:, 0], 0, width)
    np.clip(tracks_bbox[:, 1], 0, height)
    np.clip(tracks_bbox[:, 2], 0, width)
    np.clip(tracks_bbox[:, 3], 0, height)

    overlaps = bbox_overlaps(dets_bbox, tracks_bbox)
    C = 1 - overlaps
    COST_MAX = 1e9
    C[overlaps <= ov_thresh] += COST_MAX
    [row, col] = linear_sum_assignment(C)
    inds_valid = C[row, col] < COST_MAX
    inds_inconsistencies = np.union1d(col[~inds_valid], np.setdiff1d(np.arange(tracks_bbox.shape[0]), col[inds_valid]))
    hypotheses = tracks_df.iloc[inds_inconsistencies, :]

    # compute features for each hypothesis
    tracks_overlaps = bbox_overlaps(tracks_bbox, tracks_bbox)
    np.fill_diagonal(tracks_overlaps, -1)
    hypotheses_df = pd.DataFrame(columns=['x', 'y', 'w', 'h', 'r', 'det_cnt', 'mean_det_ov', 'med_det_ov', 'min_det_ov',
                                          'max_det_ov', 'mean_det_cnf', 'med_det_cnf', 'min_det_cnf', 'max_det_cnf',
                                          'hyp_cnt', 'mean_hyp_ov', 'med_hyp_ov', 'min_hyp_ov', 'max_hyp_ov',
                                          'mean_hyp_cnf', 'med_hyp_cnf', 'min_hyp_cnf', 'max_hyp_cnf', 'type',
                                          'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'width', 'height', 'id'])
    for index, h in hypotheses.iterrows():
        row = pix_to_xy(h[['bb_left', 'bb_top', 'bb_right', 'bb_bottom']].values, width, height)
        row += [h['conf']]

        # detections overlapping hypothesis
        det_ov = overlaps[:, index]
        inds_ov = det_ov > 0
        det_cnt = np.sum(inds_ov)
        if np.any(inds_ov):
            mean_det_ov = np.mean(det_ov[inds_ov])
            med_det_ov = np.median(det_ov[inds_ov])
            min_det_ov = np.min(det_ov[inds_ov])
            max_det_ov = np.max(det_ov[inds_ov])
            mean_det_cnf = np.mean(dets_df.loc[inds_ov, ['conf']].values)
            med_det_cnf = np.median(dets_df.loc[inds_ov, ['conf']].values)
            min_det_cnf = np.min(dets_df.loc[inds_ov, ['conf']].values)
            max_det_cnf = np.max(dets_df.loc[inds_ov, ['conf']].values)
        else:
            mean_det_ov = 0
            med_det_ov = 0
            min_det_ov = 0
            max_det_ov = 0
            mean_det_cnf = 0
            med_det_cnf = 0
            min_det_cnf = 0
            max_det_cnf = 0
        row += [det_cnt, mean_det_ov, med_det_ov, min_det_ov, max_det_ov, mean_det_cnf, med_det_cnf, min_det_cnf,
                max_det_cnf]

        # other shifted detections overlapping current hypothesis
        hyp_ov = tracks_overlaps[index, :]
        inds_hyp_ov = hyp_ov > 0
        hyp_cnt = np.sum(inds_hyp_ov)
        if np.any(inds_hyp_ov):
            mean_hyp_ov = np.mean(hyp_ov[inds_hyp_ov])
            med_hyp_ov = np.median(hyp_ov[inds_hyp_ov])
            min_hyp_ov = np.min(hyp_ov[inds_hyp_ov])
            max_hyp_ov = np.max(hyp_ov[inds_hyp_ov])
            mean_hyp_cnf = np.mean(tracks_df.loc[inds_hyp_ov, ['conf']].values)
            med_hyp_cnf = np.median(tracks_df.loc[inds_hyp_ov, ['conf']].values)
            min_hyp_cnf = np.min(tracks_df.loc[inds_hyp_ov, ['conf']].values)
            max_hyp_cnf = np.max(tracks_df.loc[inds_hyp_ov, ['conf']].values)
        else:
            mean_hyp_ov = 0
            med_hyp_ov = 0
            min_hyp_ov = 0
            max_hyp_ov = 0
            mean_hyp_cnf = 0
            med_hyp_cnf = 0
            min_hyp_cnf = 0
            max_hyp_cnf = 0
        row += [hyp_cnt, mean_hyp_ov, med_hyp_ov, min_hyp_ov, max_hyp_ov, mean_hyp_cnf, med_hyp_cnf, min_hyp_cnf,
                max_hyp_cnf]

        # add meta information
        row += [h['type']]
        row += h[['bb_left', 'bb_top', 'bb_right', 'bb_bottom']].values.tolist()
        row += [width, height]
        row += [h['id']]

        # append to dataframe
        hypotheses_df.loc[len(hypotheses_df)] = row
    return hypotheses_df


def parse_args():
    parser = argparse.ArgumentParser(description='Uses temporal inconsistencies to extract hypotheses')
    parser.add_argument(
        '--dets', help='text file containing detections for left camera image', required=True, type=str)
    parser.add_argument(
        '--tracks', help='text file containing tracks for left camera image', required=True, type=str)
    parser.add_argument(
        '--labels', help='text file containing ground truth labels for left camera image', required=False, type=str)
    parser.add_argument(
        '--width', help='width of image', required=True, type=float)
    parser.add_argument(
        '--height', help='height of image', required=True, type=float)
    parser.add_argument(
        '--thresh', help='confidence threshold for detections', default=0.5, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dets_df = load_detections(args.dets, thresh=args.thresh)
    tracks_df = load_detections(args.tracks)

    if args.labels:
        gt_df = load_detections(args.labels, filter_columns=False)
    else:
        gt_df = None

    h = extract_temporal_hypotheses(dets_df, tracks_df, args.width, args.height)

    h = add_labels_temporal(dets_df, h, gt_df)
