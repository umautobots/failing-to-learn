# --------------------------------------------------------
# Failing to Learn
# Copyright (c) 2017 FCAV University of Michigan
# Licensed under The MIT License [see LICENSE for details]
# Written by Manikandasriram S.R. and Cyrus Anderson
# --------------------------------------------------------

import os
import skimage.io
import argparse
import numpy as np
import pandas as pd
import time
from utils import load_detections_sequence
from stereo_cue import extract_stereo_hypotheses, add_labels_stereo, shift_detections
from temporal_cue import extract_temporal_hypotheses, add_labels_temporal


def process_sequence_temporal(dets, tracks, thresh, width, height, output, labels=None):
    dets_df = load_detections_sequence(dets, thresh=thresh)
    tracks_df = load_detections_sequence(tracks, thresh=thresh)

    if labels:
        gt_df = load_detections_sequence(labels, filter_columns=False)
    else:
        gt_df = None

    hypotheses_df = pd.DataFrame()

    num_frames = max(dets_df.frame.max(), tracks_df.frame.max()) + 1
    # process each frame
    for fr in range(num_frames):
        print("Processing frame {}\r".format(fr), end='')
        # extract per frame dataframe
        dets_df_fr = dets_df[dets_df.frame == fr]
        tracks_df_fr = tracks_df[tracks_df.frame == fr]
        dets_df_fr = dets_df_fr.loc[:, ['type', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'conf']]
        tracks_df_fr = tracks_df_fr.loc[:, ['type', 'id', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'conf']]

        h = extract_temporal_hypotheses(dets_df_fr, tracks_df_fr, width, height)

        if labels:
            # extract per frame dataframe
            gt_df_fr = gt_df[gt_df.frame == fr]
            gt_df_fr = gt_df_fr.loc[:, ['type', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'conf']]

            # determine ground truth labels for the binary classification problem
            h = add_labels_temporal(dets_df_fr, h, gt_df_fr, fr)

        # add sequence information
        hyp_frame = []
        hyp_n1 = []
        hyp_n2 = []
        for index, hyp in h.iterrows():
            hyp_frame.append(fr)
            hyp_n1.append(len(tracks_df[tracks_df.id == hyp.id]))
            hyp_n2.append(len(tracks_df[(tracks_df.id == hyp.id) & (tracks_df.frame <= fr)]))

        h['frame'] = hyp_frame
        h['n1'] = hyp_n1
        h['n2'] = hyp_n2

        hypotheses_df = hypotheses_df.append(h)

    hypotheses_df.to_csv(output, index=False)
    return hypotheses_df


def process_sequence_shift(right_dets, thresh, disparity_folder, disparity_type, shifted_dets):
    right_df = load_detections_sequence(right_dets, thresh=thresh)
    shifted_df = pd.DataFrame()
    num_frames = right_df.frame.max() + 1

    # process each frame
    for fr in range(num_frames):
        print("Processing frame {}\r".format(fr), end='')
        # extract per frame dataframe
        right_df_fr = right_df[right_df.frame == fr]
        right_df_fr = right_df_fr.loc[:, ['type', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'conf']]

        if disparity_type == "npz":
            disparity_filename = os.path.join(disparity_folder, "{:06d}.npz".format(fr))
            disparity = np.load(disparity_filename)['disparity']
        elif disparity_type == "png":
            disparity_filename = os.path.join(disparity_folder, "{:06d}.png".format(fr))
            disparity_img = skimage.io.imread(disparity_filename)  # Note: -1 is saved as 0 in image
            disparity = disparity_img.astype(float)
        else:
            raise ValueError("disparity_type unknown. Either use npz file or png")

        shifted_df_fr = shift_detections(right_df_fr, disparity)

        num_shifted_dets = len(shifted_df_fr)
        shifted_df_fr['frame'] = [fr]*num_shifted_dets
        shifted_df_fr['id'] = [-1]*num_shifted_dets

        shifted_df = shifted_df.append(shifted_df_fr, ignore_index=True)

    os.makedirs(os.path.dirname(shifted_dets), exist_ok=True)
    shifted_df.to_csv(shifted_dets, index=False)
    return shifted_df


def process_sequence_stereo(left_dets, shifted_dets, thresh, shifted_thresh, width, height, output, labels=None):
    left_df = load_detections_sequence(left_dets, thresh=thresh)
    shifted_df = pd.read_csv(shifted_dets)
    shifted_df = shifted_df[shifted_df.conf >= shifted_thresh]

    if labels:
        gt_df = load_detections_sequence(labels, filter_columns=False)
    else:
        gt_df = None

    hypotheses_df = pd.DataFrame()

    num_frames = int(max(left_df.frame.max(), shifted_df.frame.max())) + 1
    # process each frame
    for fr in range(num_frames):
        print("Processing frame {}\r".format(fr), end='')
        # extract per frame dataframe
        left_df_fr = left_df[left_df.frame == fr]
        shifted_df_fr = shifted_df[shifted_df.frame == fr]
        left_df_fr = left_df_fr.loc[:, ['type', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'conf']]
        shifted_df_fr = shifted_df_fr.loc[:, ['type', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'conf']]

        h = extract_stereo_hypotheses(left_df_fr, shifted_df_fr, width, height)

        if labels:
            # extract per frame dataframe
            gt_df_fr = gt_df[gt_df.frame == fr]
            gt_df_fr = gt_df_fr.loc[:, ['type', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'conf']]

            # determine ground truth labels for the binary classification problem
            h = add_labels_stereo(left_df_fr, h, gt_df_fr)

        # add meta information
        num_hyp = len(h)
        h['id'] = [-1]*num_hyp
        h['frame'] = [fr]*num_hyp
        hypotheses_df = hypotheses_df.append(h)

    hypotheses_df.to_csv(output, index=False)
    return hypotheses_df


def parse_args():
    parser = argparse.ArgumentParser(description='Uses stereo or temporal cue for a sequence')
    parser.add_argument(
        '--left_dets', help='text file containing detections for left camera image', type=str)
    parser.add_argument(
        '--right_dets', help='text file containing detections for right camera image', type=str)
    parser.add_argument(
        '--shifted_dets', help='text file containing shifted detections from right camera image', type=str)
    parser.add_argument(
        '--tracks', help='text file containing tracks for left camera image', type=str)
    parser.add_argument(
        '--labels', help='text file containing ground truth labels for left camera image', type=str)
    parser.add_argument(
        '--mode', help='stereo or temporal or shift', required=True, type=str)
    parser.add_argument(
        '--disparity', help='folder containing disparity images for every frame', type=str)
    parser.add_argument(
        '--disparity_type', help='npz or png', default='npz', type=str)
    parser.add_argument(
        '--thresh', help='confidence threshold for detections', default=0.5, type=float)
    parser.add_argument(
        '--shifted_thresh', help='confidence threshold for shifted detections', default=0.5, type=float)
    parser.add_argument(
        '--hypotheses', help='output CSV file for (un)labeled hypotheses', type=str)
    parser.add_argument(
        '--width', help='width of image', type=float)
    parser.add_argument(
        '--height', help='height of image', type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode.lower() == "shift":
        process_sequence_shift(args.right_dets, args.thresh, args.disparity, args.disparity_type, args.shifted_dets)
    elif args.mode.lower() == "stereo":
        process_sequence_stereo(args.left_dets, args.shifted_dets, args.thresh, args.shifted_thresh, args.width,
                                args.height, args.hypotheses, args.labels)
    elif args.mode.lower() == "temporal":
        process_sequence_temporal(args.left_dets, args.tracks, args.thresh, args.width, args.height, args.hypotheses,
                                  args.labels)
