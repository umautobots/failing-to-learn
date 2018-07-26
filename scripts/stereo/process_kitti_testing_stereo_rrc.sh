set -e
cd $FAILING_TO_LEARN
declare -a arr=('0022' '0023' '0024' '0025' '0026' '0027' '0028')
RESULTS="results"
mkdir -p $RESULTS/KITTI/testing/false_negatives/stereo_cue
for i in "${arr[@]}"
do
	echo "Processing sequence $i"
	python3 src/process_sequence.py --left_dets $RESULTS/KITTI/testing/dets/$i.txt --shifted_dets $RESULTS/KITTI/testing/shifted_dets/$i.txt --hypotheses $RESULTS/KITTI/testing/false_negatives/stereo_cue/$i.csv --thresh 0.5 --shifted_thresh 0.5 --mode stereo --width 1242 --height 375
done
