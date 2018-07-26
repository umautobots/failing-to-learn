set -e
cd $FAILING_TO_LEARN
declare -a arr=('0000' '0001' '0002' '0003' '0004' '0005' '0006' '0007' '0008' '0009' '0010' '0011' '0012' '0013' '0014' '0015' '0016' '0017' '0018' '0019' '0020' '0021' '0022' '0023' '0024' '0025' '0026' '0027' '0028' '0029' '0030' '0031' '0032' '0033' '0034' '0035' '0036' '0037' '0038' '0039' '0040' '0041' '0042' '0043' '0044' '0045' '0046' '0047' '0048' '0049' '0050' '0051' '0052' '0053' '0054' '0055' '0056' '0057' '0058' '0059' '0060' '0061' '0062' '0063' '0064' '0065' '0066' '0067' '0068' '0069' '0070' '0071' '0072' '0073' '0074' '0075' '0076' '0077' '0078' '0079' '0080' '0081' '0082' '0083' '0084' '0085' '0086' '0087' '0088' '0089' '0090' '0091' '0092' '0093' '0094' '0095' '0096' '0097' '0098' '0099' '0100' '0101' '0102' '0103')
RESULTS="results"
mkdir -p $RESULTS/GTA/false_negatives/stereo_cue
for i in "${arr[@]}"
do
	echo "Processing sequence $i"
	python3 src/process_sequence.py --left_dets $RESULTS/GTA/dets/$i.txt --shifted_dets $RESULTS/GTA/shifted_dets/$i.txt --labels data/GTA/label_02/$i.txt --hypotheses $RESULTS/GTA/false_negatives/stereo_cue/$i.csv --thresh 0.5 --shifted_thresh 0.5 --mode stereo --width 1920 --height 1080
done
