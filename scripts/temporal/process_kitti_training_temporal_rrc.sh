set -e
cd $FAILING_TO_LEARN
declare -a arr=('0000' '0001' '0002' '0003' '0004' '0005' '0006' '0007' '0008' '0009' '0010' '0011' '0012' '0013' '0014' '0015' '0016' '0018' '0019' '0020')
RESULTS="results"
mkdir -p $RESULTS/KITTI/training/false_negatives/temporal_cue
for i in "${arr[@]}"
do
	echo "Processing sequence $i"
	python3 src/process_sequence.py --left_dets $RESULTS/KITTI/training/dets/$i.txt --tracks $RESULTS/KITTI/training/tracks/$i.txt --labels data/KITTI/training/label_02/$i.txt --hypotheses $RESULTS/KITTI/training/false_negatives/temporal_cue/$i.csv --mode temporal --width 1242 --height 375 --thresh 0.5
done
