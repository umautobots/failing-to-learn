set -e
cd $FAILING_TO_LEARN
declare -a arr=('0000' '0001' '0002' '0003' '0004' '0005' '0006' '0007' '0008' '0009' '0010' '0011' '0012' '0013' '0014' '0015' '0016' '0017' '0018' '0019' '0020' '0022' '0023' '0024' '0025' '0026' '0027' '0028')
RESULTS="results"
for i in "${arr[@]}"
do
	echo "Processing sequence $i"
	python3 src/process_sequence.py --right_dets $RESULTS/KITTI/testing/dets_right/$i.txt --disparity data/KITTI/testing/disparity/$i --disparity_type npz --shifted_dets $RESULTS/KITTI/testing/shifted_dets/$i.txt --mode shift --thresh 0.3
done
