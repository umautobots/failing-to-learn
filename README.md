# Failing to Learn: Autonomously Identifying Perception Failures for Self-driving Cars

by Manikandasriram S.R. and Cyrus Anderson at FCAV in University of Michigan.


### Introduction

The paper presents a method to identify false negatives made by CNN based 
object detectors on unlabeled datasets. For more details, 
please refer to our paper on arxiv(https://arxiv.org/pdf/1707.00051.pdf).

### License

Failing to Learn code is released under the MIT License (refer to the LICENSE file for details)


### Citation

If you find this paper or data helpful, please consider citing:

```
@article{srinivasan2017failing,
	title={Failing to Learn: Autonomously Identifying Perception Failures for Self-driving Cars}, 
    author={Srinivasan Ramanagopal, Manikandasriram and Anderson, Cyrus and Vasudevan, Ram and Johnson-Roberson, Matthew},	
    journal={arXiv preprint arXiv:1707.00051},
    year={2017}
}
```
### Dataset

We also release a new tracking dataset with over 100 sequences totaling more than 80,000 labeled images from a game engine to facilitate further research. 
* [Sequences and Annotations](http://www.umich.edu/~fcav/GTA_Tracking_Dataset.tar.gz) (23G)

### Temporal Cue

Given a trained Object Detector and a Multi Object Tracker, we determine missed objects from object detector predictions on unlabeled datasets. In this demo, we will use an [RRC](https://github.com/xiaohaoChen/rrc_detection) object detector trained on the KITTI dataset and use our algorithm to identify mistakes on a GTA tracking dataset. We use the ground truth labels from the GTA dataset only to evaluate the precision and recall of our algorithm.

1. Clone the repository. Let `$REPO` be the root folder.
```
git clone https://github.com/umautobots/failing-to-learn 
cd failing-to-learn
```
3. Download and extract KITTI Tracking dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and our GTA Tracking dataset from [here](https://fcav.engin.umich.edu/research/failing-to-learn). Create symlinks to the datasets like shown below. 
```
mkdir -p $REPO/data/KITTI/training/ && cd $REPO/data/KITTI/training
ln -s <path/to/KITTI_Tracking_Dataset>/training/image_02 image_02
ln -s <path/to/KITTI_Tracking_Dataset>/training/label_02 label_02
mkdir -p $REPO/data/GTA && cd $REPO/data/GTA
ln -s <path/to/GTA_Tracking_Dataset>/image_02 image_02
ln -s <path/to/GTA_Tracking_Dataset>/label_02 label_02
```
4. Use the trained object detector to make predictions for the KITTI training dataset as well as the GTA dataset in the KITTI tracking dataset format and place them in the results folder as shown below. For convenience, we provide the predictions made by an RRC object detector (Drive link: https://goo.gl/2UKwKQ).
5. You should now have a folder structure such as in `directory_structure.txt`
6. Compile MDP tracker by running `$REPO/MDP_Tracking/compile.m`
7. Run the script `$REPO/lib/test_failing_to_learn.m`. This is a simple script that invokes the necessary scripts in sequence. 

**Note:** This script will potentially take an entire day to process all the data. The MDP Tracker creates a custom data structure to store images in `.mat` files and store them inside `$REPO/MDP_Tracking/results_kitti/data` folder. In our experiments, this folder alone has a size of `~1TB`. You can use symlinks to store this data in a different disk while maintaining the same folder structure. The final results will be placed in `$REPO/results/GTA/false_negatives`. 


### Stereo Cue
This will be added soon.
