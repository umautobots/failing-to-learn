# --------------------------------------------------------
# Failing to Learn
# Copyright (c) 2017 FCAV University of Michigan
# Licensed under The MIT License [see LICENSE for details]
# Written by Manikandasriram S.R. and Cyrus Anderson
# --------------------------------------------------------

import numpy as np
import pandas as pd


def load_detections_sequence(filename, filter_columns=True, thresh=0.5):
    # for KITTI tracking format
    column_names = ['frame', 'id', 'type', 'truncated', 'occluded', 'alpha', 'bb_left', 'bb_top', 'bb_right',
                    'bb_bottom', '3D height', '3D width', '3D length', '3D x', '3D y', '3D z', 'rotation y', 'conf']
    df = pd.read_csv(filename, delim_whitespace=True, names=column_names)

    # filter based on score (if available)
    if np.all(df.conf.notna()):
        df = df[df.conf > thresh]

    # filter based on type
    df = df[(df.type == 'Car') | (df.type == 'car') | (df.type == 'DontCare') | (df.type == 'Van') | (df.type ==
                                                                                                      'Truck')]

    if filter_columns:
        # filter out unused columns
        df = df.loc[:, ['frame', 'id', 'type', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'conf']]
    return df


def load_detections(filename, filter_columns=True, thresh=0.):
    # for KITTI object recognition format
    column_names = ['type', 'truncated', 'occluded', 'alpha', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom',
                    '3D height', '3D width', '3D length', '3D x', '3D y', '3D z', 'rotation y', 'conf']
    df = pd.read_csv(filename, delim_whitespace=True, names=column_names)

    # filter based on score (if available)
    if np.all(df.conf.notna()):
        df = df[df.conf > thresh]

    # filter based on type
    df = df[(df.type == 'Car') | (df.type == 'car')]

    if filter_columns:
        # filter out unused columns
        df = df.loc[:, ['type', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom', 'conf']]
    return df


def bbox_overlaps(bbox_a, bbox_b, type='iou'):
    # bbox_a and bbox_b are Nx4 matrices with each row being [xmin, ymin, xmax, ymax]
    overlaps = np.zeros((bbox_a.shape[0], bbox_b.shape[0]))
    area_a = (bbox_a[:, 2] - bbox_a[:, 0] + 1) * (bbox_a[:, 3] - bbox_a[:, 1] + 1)
    area_b = (bbox_b[:, 2] - bbox_b[:, 0] + 1) * (bbox_b[:, 3] - bbox_b[:, 1] + 1)
    for i, bbox in enumerate(bbox_a):
        xx1 = np.maximum(bbox[0], bbox_b[:, 0])
        yy1 = np.maximum(bbox[1], bbox_b[:, 1])
        xx2 = np.minimum(bbox[2], bbox_b[:, 2])
        yy2 = np.minimum(bbox[3], bbox_b[:, 3])
        w = xx2 - xx1 + 1
        h = yy2 - yy1 + 1
        area_int = w*h
        nonzero_inds = (w > 0) & (h > 0)
        area_int[~nonzero_inds] = 0
        area_union = area_a[i] + area_b - area_int
        ov = area_int/area_union
        if type == 'iou':
            overlaps[i, :] = ov
        elif type == 'b':
            overlaps[i, :] = area_int/area_b
        elif type == 'a':
            overlaps[i, :] = area_int / area_a
        else:
            raise ValueError
    return overlaps


def pix_to_xy(bbox, img_width, img_height):
    [xmin, ymin, xmax, ymax] = bbox

    # bbox to [x_c, y_c, w, h]
    x_px = (xmin + xmax + 1) / 2.
    y_px = (ymin + ymax + 1) / 2.
    w_px = xmax - xmin + 1
    h_px = ymax - ymin + 1

    # shift origin from top left to center of image
    x_px = x_px - img_width/2.
    y_px = img_height/2. - y_px

    # scale to unit img width
    # since we want to preserve aspect ratio, we are scaling everything by width
    x = x_px/img_width
    y = y_px/img_width
    w = w_px/img_width
    h = h_px/img_width

    return [x, y, w, h]