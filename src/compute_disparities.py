# --------------------------------------------------------
# Failing to Learn
# Copyright (c) 2017 FCAV University of Michigan
# Licensed under The MIT License [see LICENSE for details]
# Written by Manikandasriram S.R. and Cyrus Anderson
# --------------------------------------------------------

# compute disparity using OpenCV for the given left and right image directories
import numpy as np
import os
import argparse
import cv2
import glob
from multiprocessing import Pool
# import matplotlib.pyplot as plt

def compute_disparity(disp_args):
    left_image_path = disp_args[0]
    right_image_path = disp_args[1]
    disp_image_path = disp_args[2]
    disp_ext = disp_args[3]

    imgL = cv2.imread(left_image_path, 0)
    imgR = cv2.imread(right_image_path, 0)

    stereo = cv2.StereoSGBM_create(preFilterCap=63, blockSize=3, P1=36, P2=288, minDisparity=0, numDisparities=128,
                                   uniquenessRatio=10, speckleWindowSize=100, speckleRange=32, disp12MaxDiff=1,
                                   mode=cv2.StereoSGBM_MODE_HH)
    disparity = stereo.compute(imgL, imgR).astype(np.float32)/16.0

    # fig = plt.figure(1)
    # ax1 = fig.add_subplot(131)
    # ax1.imshow(imgL)
    # ax2 = fig.add_subplot(132)
    # ax2.imshow(imgR)
    # ax3 = fig.add_subplot(133)
    # ax3.imshow(disparity)
    # plt.show()

    np.savez_compressed(disp_image_path, disparity=disparity)


def main():
    args = parse_args()
    left_img_list = sorted(glob.glob(os.path.join(args.image_left_dir,"*."+args.image_ext)))
    right_img_list = sorted(glob.glob(os.path.join(args.image_right_dir,"*."+args.image_ext)))
    disp_args = []
    for left_img, right_img in zip(left_img_list, right_img_list):
        disp_args.append((left_img, right_img, os.path.join(args.disparity_dir,
                         left_img.split('/')[-1].replace(args.image_ext, args.disp_ext)), args.disp_ext))
    # print(disp_args[0])
    p = Pool(4)
    p.map(compute_disparity, disp_args)
    # compute_disparity(disp_args[0])

def parse_args():
    parser = argparse.ArgumentParser(description='computes disparities using OpenCV with multiprocessing')
    parser.add_argument(
        '--image_left_dir', help='directory containing left images', default=None, type=str)
    parser.add_argument(
        '--image_right_dir', help='directory containing right images', default=None, type=str)
    parser.add_argument(
        '--image_ext', help="file extension of images", default="jpg", type=str)
    parser.add_argument(
        '--disparity_dir', help='directory to output disparities', default=None, type=str)
    parser.add_argument(
        '--disp_ext', help="file extension of disparity", default="npz", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
