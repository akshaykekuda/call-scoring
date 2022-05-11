#!/bin/bash

# This is so one file can be run by each server. The servers all have 
# name P-CTAPP8DXNN, where NN is 01 .. 05. server_num=NN
# This solution only works because of the hostname convention.
server_num=$(grep -Poh "[0-9]{2,2}" /etc/hostname)

# You can set -d particular date to force the script 
# to back date.
while getopts d: option
do
case "${option}"
in
d) filedate=${OPTARG} ;;
esac
done

if [ -z "$filedate" ]; then
 filedate=$(date -d "1 days ago" +"%Y-%m-%d")
fi

# clean out build path
build_output=/home/mcmurw/transcriber/build/output/
# uncomment later
rm -rf $build_output
mkdir -p $build_output

# archive any output in the build output folder- we can use these transcriptions
# for new examples of good or bad calls for the model.
destination=/mnt/callratings/transcriptions/"$filedate"/$server_num
mkdir -p /mnt/callratings/transcriptions/"$filedate"/
mkdir -p $destination


# path to model file
model_path=/mnt/models/my_doc2vec_model_trained_2

# Directories for sales calls
sales_lo=/mnt/models/scoring/sale_below60/
sales_hi=/mnt/models/scoring/sale_above85/
sales_output=/mnt/callratings/daily_scores/"$filedate"/sale"$server_num".csv

# directories for customer service calls
cs_lo=/mnt/models/scoring/below60/
cs_hi=/mnt/models/scoring/above85/
cs_output=/mnt/callratings/daily_scores/"$filedate"/cs"$server_num".csv

# define directory where to put wav files that have been assigned to this server.
wavpath=/mnt/callratings/run/"$filedate"/test"$server_num"/
mkdir -p $wavpath
cd /mnt/callratings/

# This loop converts all *.opus files to *.wav
for f in "$wavpath"*.opus; do
    #get the basename of file, and replace the ".opus" with ".wav"
        f2=${f%.*}".wav"

    # use opusdec to convert the opus file to a wav file with sampling rate 8000
       opusdec --rate 8000 $f $f2
done

# Delete all .opus files, since they are no longer needed
rm -f "$wavpath"*.opus

# Create 15 directories, one for each core
for i in {0..15}; do
   mkdir -p "$wavpath"$i
done

# initialize count variable
count=1

# iterate through each wave, and mv it to its directory
for f in "$wavpath"*.wav; do
        mv $f "$wavpath"$(($count%16))
        (( count++ ))
done

# call batch.sh on each directory
# 16 directories because 16 cores
# This may seem dumb but it forces the server to use all 16 cores.

for j in {0..15}; do
	/home/mcmurw/transcriber/batch.sh "$wavpath""$j"/* &
	sleep 1
done
wait
sleep 1

cp "$build_output"* $destination

# Clean up and reorganize the wav files.
find $wavpath -name '*.wav' -exec mv {} $wavpath \;

# Once this particular servers workload is done it will then score the calls it transcribed
mkdir -p /mnt/callratings/daily_scores/"$filedate"/

# customer service score
python3 transcripts_to_scores_withday.py $wavpath $build_output $model_path $cs_lo $cs_hi $cs_output

# sales score
python3 transcripts_to_scores_withday.py $wavpath $build_output $model_path $sales_lo $sales_hi $sales_output

# At this point, the wav files are no longer needed and can be deleted.
#This command has been commented out for now.
rm -f "$wavpath"*.wav


# Clear all the files that are older than 2 weeks to ensure that the server does not run out of space.

current_date=$(date +"%Y-%m-%d")

cd /mnt/callratings/

last_date=$(date --date="$current_date -2 week" +"%Y-%m-%d")
find run/ ! -newermt "$last_date" | xargs rm -rfv
find run/ -empty -type d -delete

last_date=$(date --date="$current_date -1 month" +"%Y-%m-%d")
# find transcriptions/ ! -newermt "$last_date" | xargs rm -rfv
# find transcriptions/ -empty -type d -delete


rm -rf /home/mcmurw/transcriber/build/output
rm -rf /home/mcmurw/transcriber/build/audio
rm -rf /home/mcmurw/transcriber/build/trans
rm -rf /home/mcmurw/transcriber/build/diarization
