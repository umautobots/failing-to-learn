# Failing to Learn: Autonomously Identifying Perception Failures for Self-driving Cars

by Manikandasriram S.R. and Cyrus Anderson at [UM FCAV](https://fcav.engin.umich.edu/)

### Introduction

The paper presents a method to identify false negatives made by single frame
object detectors on unlabeled datasets. Our paper has been accepted in RA-L and will be presented 
at IROS 2018 in Madrid, Spain. For more details, 
please refer to our published paper in IEEE Xplore (https://ieeexplore.ieee.org/document/8412512/) or
the accepted version in arxiv(https://arxiv.org/abs/1707.00051).

### License

Failing to Learn code is released under the MIT License (refer to the LICENSE file for details)

### Citation

If you find this paper or data helpful, please consider citing:

```
@article{srinivasanramanagopal2018failing, 
    title={Failing to Learn: Autonomously Identifying Perception Failures for Self-driving Cars}, 
    author={M. Srinivasan Ramanagopal and C. Anderson and R. Vasudevan and M. Johnson-Roberson}, 
    journal={IEEE Robotics and Automation Letters}, 
    year={2018}, 
    doi={10.1109/LRA.2018.2857402}, 
}
```

### GTA Dataset

We also release a new tracking dataset with 104 sequences totaling 80,655 labeled pairs of stereo images from a game engine along with ground truth disparity to facilitate further research. Note this data can only be used for non-commercial applications.
* [Left Camera Images and Annotations](http://www.umich.edu/~fcav/GTA_Tracking_Dataset.tar.gz) (22.5GB)
* [Right Camera Images](https://s3.us-east-2.amazonaws.com/ngv.datasets/GTA_Tracking_right_images.tar.gz) (54.5GB)
* [Disparity Images](https://s3.us-east-2.amazonaws.com/ngv.datasets/GTA_Tracking_disparity.tar.gz) (4.8GB)

The right camera images were obtained by reprojecting the left camera images using the depth buffer from the game engine. The resulting holes were filled using the Navier Stokes Inpainting method from OpenCV. 

### Directory Setup

In this demo, we will use an [RRC](https://github.com/xiaohaoChen/rrc_detection) object detector trained on [Sim10k](https://fcav.engin.umich.edu/research/driving-in-the-matrix) dataset and use our approach to identify missed detections on KITTI tracking dataset. 

1. Clone the repository. Let `$FAILING_TO_LEARN` be the root folder.
```
git clone https://github.com/umautobots/failing-to-learn 
cd failing-to-learn
export FAILING_TO_LEARN=</path/to/failing-to-learn>
```

2. Download and extract the GTA dataset (from above) and [KITTI tracking dataset](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) into folder named `data` as shown below. 

3. For reproducing the results from our paper, we provide the `results` folder [here](https://drive.google.com/file/d/1e7tjhQL4eEpTmEWmRC77zEMrAONDp6Jq/view?usp=sharing) with detections and tracks (constructed using MDP Tracker) for the GTA and KITTI tracking datasets. You can replace this with your own detections and tracks in the KITTI tracking format. 

The resulting folder structure should look as follows:

```
.
├── data
│   ├── GTA
│   │   ├── image_02 -> <path/to/GTA_Dataset>/image_02/
│   │   ├── image_03 -> <path/to/GTA_Dataset>/image_03/
│   │   └── label_02 -> <path/to/GTA_Dataset>/label_02/
│   └── KITTI
│       ├── training
│       │   ├── image_02 -> <path/to/KITTI Dataset>/training/image_02/
│       │   ├── image_03 -> <path/to/KITTI Dataset>/training/image_03/
│       │   └── label_02 -> <path/to/KITTI_Dataset>/training/label_02/
│       └── testing
│           ├── image_02 -> <path/to/KITTI Dataset>/training/image_02/
│           └── image_03 -> <path/to/KITTI Dataset>/training/image_03/
├── scripts
│   └── ...
├── src
│   └── ...
├── LICENSE
├── README.md
└── results
    ├── GTA
    │   ├── dets
    │   │   ├── 0000.txt
    │   │   ├── ...
    │   │   └── 0103.txt
    │   ├── dets_right
    │   │   ├── 0000.txt
    │   │   ├── ...
    │   │   └── 0103.txt
    │   └── tracks
    │       ├── 0000.txt
    │       ├── ...
    │       └── 0103.txt
    └── KITTI
        ├── training
        │   ├── dets
        │   │   ├── 0000.txt
        │   │   ├── ...
        │   │   └── 0020.txt
        │   ├── dets_right
        │   │   ├── 0000.txt
        │   │   ├── ...
        │   │   └── 0020.txt
        │   └── tracks
        │       ├── 0000.txt
        │       ├── ...
        │       └── 0020.txt
        └── testing
            ├── dets
            │   ├── 0000.txt
            │   ├── ...
            │   └── 0020.txt
            ├── dets_right
            │   ├── 0000.txt
            │   ├── ...
            │   └── 0020.txt
            └── tracks
                ├── 0000.txt
                ├── ...
                └── 0020.txt
```

### Temporal Cue

0. Use your favourite Multi-Object Tracker to construct tracks from the detections. We use the MDP Tracker in our experiments. 

1. Compute the hypotheses for the GTA dataset. This will output the hypotheses as a CSV file in `results/GTA/false_negatives/temporal_cue`
```
bash scripts/temporal/process_gta_temporal_rrc.sh
```

2. Compute the hypotheses for the KITTI tracking dataset. This will output the hypotheses as a CSV file in `results/KITTI/training/false_negatives/temporal_cue`
```
bash scripts/temporal/process_kitti_training_temporal_rrc.sh
```

3. Train and test the binary classifier.
```
python3 src/classifier_temporal.py --train_dataset scripts/temporal/train_dataset_temporal_rrc.txt --test_dataset scripts/temporal/test_dataset_temporal_rrc_kitti.txt --model_file results/temporal_classifier.pkl --output_folder results/KITTI/training/false_negatives/temporal_cue/kitti_format
```

4. (Optional) Compute the hypotheses for KITTI testing dataset (unlabeled).
```
bash scripts/temporal/process_kitti_testing_temporal_rrc.sh
```

5. (Optional) Run classifier on KITTI testing dataset (unlabeled).
```
python3 src/test_classifier_temporal.py --test_dataset scripts/temporal/test_dataset_temporal_rrc_kitti_test.txt --model_file results/temporal_classifier.pkl --output_folder results/KITTI/testing/false_negatives/temporal_cue/kitti_format
```

### Stereo Cue

1. Compute disparities for the GTA dataset.
```
bash scripts/compute_gta_disparities.sh
```

2. Compute disparities for the KITTI dataset.
```
bash scripts/compute_kitti_training_disparities.sh
```

3. Shift detections using the disparities for the GTA dataset. The script uses a threshold filter to shift only confident detections. This threshold should be lower than the threshold used in step 5 below.
```
bash scripts/stereo/process_gta_shift_rrc.sh
```

4. Shift detections using the disparities for the KITTI dataset. The script uses a threshold filter to shift only confident detections. This threshold should be lower than the threshold used in step 5 below.
```
bash scripts/stereo/process_kitti_training_shift_rrc.sh
```

5. Compute the hypotheses for the GTA dataset.
```
bash scripts/stereo/process_gta_stereo_rrc.sh
```

6. Compute the hypotheses for the KITTI dataset.
```
bash scripts/stereo/process_kitti_training_stereo_rrc.sh
```

7. Train and test the binary classifier.
```
python3 src/classifier_stereo.py --train_dataset scripts/stereo/train_dataset_stereo_rrc.txt --test_dataset scripts/stereo/test_dataset_stereo_rrc_kitti.txt --model_file results/stereo_classifier.pkl --output_folder results/KITTI/training/false_negatives/stereo_cue/kitti_format
```

8. (Optional) Shift detections using the disparities for the KITTI testing dataset (unlabeled). 
```
bash scripts/stereo/process_kitti_testing_shift_rrc.sh
```

9. (Optional) Compute the hypotheses for KITTI testing dataset (unlabeled).
```
bash scripts/stereo/process_kitti_testing_stereo_rrc.sh
```

10. (Optional) Run trained classifier on KITTI testing dataset (unlabeled).
```
python3 src/test_classifier_stereo.py --test_dataset scripts/stereo/test_dataset_stereo_rrc_kitti_test.txt --model_file results/stereo_classifier.pkl --output_folder results/KITTI/testing/false_negatives/stereo_cue/kitti_format
```
