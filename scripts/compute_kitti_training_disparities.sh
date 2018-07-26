set -e
cd $FAILING_TO_LEARN
declare -a arr=('0000' '0001' '0002' '0003' '0004' '0005' '0006' '0007' '0008' '0009' '0010' '0011' '0012' '0013' '0014' '0015' '0016' '0017' '0018' '0019' '0020')
for i in "${arr[@]}"
do
	echo "Processing sequence $i"
	mkdir -p data/KITTI/training/disparity/$i
	python3 src/compute_disparities.py --image_left_dir data/KITTI/training/image_02/$i --image_right_dir data/KITTI/training/image_03/$i --image_ext png --disparity_dir data/KITTI/training/disparity/$i --disp_ext npz
done