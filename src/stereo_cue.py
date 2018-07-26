# --------------------------------------------------------
# Failing to Learn
# Copyright (c) 2017 FCAV University of Michigan
# Licensed under The MIT License [see LICENSE for details]
# Written by Manikandasriram S.R. and Cyrus Anderson
# --------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
import skimage.io
from scipy.optimize import linear_sum_assignment
from utils import load_detections, bbox_overlaps, pix_to_xy


def add_labels_stereo(left_df, hyp_df, gt_df, ov_thresh=0.5):
    dc_df = gt_df[gt_df.type == 'DontCare']
    gt_df = gt_df[gt_df.type != 'DontCare']
    gt_df = gt_df.set_index(np.arange(len(gt_df)))

    # use gt to determine missed detections
    left_bbox = left_df[['bb_left', 'bb_top', 'bb_right', 'bb_bottom']].values
    gt_bbox = gt_df[['bb_left', 'bb_top', 'bb_right', 'bb_bottom']].values
    overlaps = bbox_overlaps(gt_bbox, left_bbox)
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


def shift_detections(right_df, disparity):
    # ensure indices of dataframes are [0,n]
    right_df = right_df.set_index(np.arange(len(right_df)))

    # construct disparity dataframe
    height, width = disparity.shape
    lx, ly = np.meshgrid(np.arange(width), np.arange(height))
    lx = lx.ravel()
    ly = ly.ravel()
    disparity = disparity.ravel()
    # disparity val, L_x, L_y, R_x (disparity only works on x-shifts)
    # all px coords -> L_x, L_y
    # L_x + shift = R_x
    disp_df = pd.DataFrame(np.array([lx, ly, disparity, lx + disparity]).T, columns=['L_x', 'L_y', 'disparity', 'R_x'])

    # compute shifted_df
    shifted_df = pd.DataFrame(columns=['type', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'conf'])
    for index, det_right in right_df.iterrows():
        matching_disp_df = disp_df[
            disp_df['R_x'].between(det_right.bb_left, det_right.bb_right) &
            disp_df['L_y'].between(det_right.bb_top, det_right.bb_bottom)
        ]
        matching_disp_df = matching_disp_df[matching_disp_df['disparity'] >= 0]
        if len(matching_disp_df) == 0:
            # print('Skipping detection {} due to invalid disparity'.format(index))
            continue
        median_disparity = np.median(matching_disp_df['disparity'])
        shifted_det = det_right.copy()
        shifted_det.bb_left = np.clip(shifted_det.bb_left + median_disparity, 0, width)
        shifted_det.bb_right = np.clip(shifted_det.bb_right + median_disparity, 0, width)
        shifted_df = shifted_df.append(shifted_det)
    return shifted_df


def extract_stereo_hypotheses(left_df, shifted_df, width, height, ov_thresh=0.5):
    # ensure indices of dataframes are [0,n]
    left_df = left_df.set_index(np.arange(len(left_df)))
    shifted_df = shifted_df.set_index(np.arange(len(shifted_df)))

    # use hungarian to determine inconsistencies
    left_bbox = left_df[['bb_left', 'bb_top', 'bb_right', 'bb_bottom']].values
    shifted_bbox = shifted_df[['bb_left', 'bb_top', 'bb_right', 'bb_bottom']].values
    overlaps = bbox_overlaps(left_bbox, shifted_bbox)
    C = 1 - overlaps
    COST_MAX = 1e9
    C[overlaps <= ov_thresh] += COST_MAX
    [row, col] = linear_sum_assignment(C)
    inds_valid = C[row, col] < COST_MAX
    inds_inconsistencies = np.union1d(col[~inds_valid], np.setdiff1d(np.arange(shifted_bbox.shape[0]), col[inds_valid]))
    hypotheses = shifted_df.iloc[inds_inconsistencies, :]

    # compute features for each hypothesis
    shifted_det_overlaps = bbox_overlaps(shifted_bbox, shifted_bbox)
    np.fill_diagonal(shifted_det_overlaps, -1)
    hypotheses_df = pd.DataFrame(columns=['x', 'y', 'w', 'h', 'r', 'det_cnt', 'mean_det_ov', 'med_det_ov', 'min_det_ov',
                                          'max_det_ov', 'mean_det_cnf', 'med_det_cnf', 'min_det_cnf', 'max_det_cnf',
                                          'hyp_cnt', 'mean_hyp_ov', 'med_hyp_ov', 'min_hyp_ov', 'max_hyp_ov',
                                          'mean_hyp_cnf', 'med_hyp_cnf', 'min_hyp_cnf', 'max_hyp_cnf', 'type',
                                          'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'width', 'height'])
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
            mean_det_cnf = np.mean(left_df.loc[inds_ov, ['conf']].values)
            med_det_cnf = np.median(left_df.loc[inds_ov, ['conf']].values)
            min_det_cnf = np.min(left_df.loc[inds_ov, ['conf']].values)
            max_det_cnf = np.max(left_df.loc[inds_ov, ['conf']].values)
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
        hyp_ov = shifted_det_overlaps[index, :]
        inds_hyp_ov = hyp_ov > 0
        hyp_cnt = np.sum(inds_hyp_ov)
        if np.any(inds_hyp_ov):
            mean_hyp_ov = np.mean(hyp_ov[inds_hyp_ov])
            med_hyp_ov = np.median(hyp_ov[inds_hyp_ov])
            min_hyp_ov = np.min(hyp_ov[inds_hyp_ov])
            max_hyp_ov = np.max(hyp_ov[inds_hyp_ov])
            mean_hyp_cnf = np.mean(shifted_df.loc[inds_hyp_ov, ['conf']].values)
            med_hyp_cnf = np.median(shifted_df.loc[inds_hyp_ov, ['conf']].values)
            min_hyp_cnf = np.min(shifted_df.loc[inds_hyp_ov, ['conf']].values)
            max_hyp_cnf = np.max(shifted_df.loc[inds_hyp_ov, ['conf']].values)
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

        # append to dataframe
        hypotheses_df.loc[len(hypotheses_df)] = row
    return hypotheses_df


def parse_args():
    parser = argparse.ArgumentParser(description='Uses stereo inconsistencies to extract hypotheses')
    parser.add_argument(
        '--left_dets', help='text file containing detections for left camera image', required=True, type=str)
    parser.add_argument(
        '--right_dets', help='text file containing detections for right camera image', required=True, type=str)
    parser.add_argument(
        '--labels', help='text file containing ground truth labels for left camera image', required=False, type=str)
    parser.add_argument(
        '--disparity', help='pre-computed disparity matrix (should be same size as camera images)',
        required=True, type=str)
    parser.add_argument(
        '--thresh', help='confidence threshold for detections', default=0.5, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    left_df = load_detections(args.left_dets, thresh=args.thresh)
    right_df = load_detections(args.right_dets, thresh=args.thresh)

    if args.labels:
        gt_df = load_detections(args.labels, filter_columns=False)
    else:
        gt_df = None

    disparity_img = skimage.io.imread(args.disparity)   # Note: -1 is saved as 0 in image
    disparity = disparity_img.astype(float)
    # disparity = np.load(args.disparity)

    shifted_df = shift_detections(right_df, disparity)
    height, width = disparity.shape

    h = extract_stereo_hypotheses(left_df, shifted_df, width, height)

    h = add_labels_stereo(left_df, h, gt_df)
