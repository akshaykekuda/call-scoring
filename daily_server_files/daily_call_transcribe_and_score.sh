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

asr_base=/home/mcmurw/

# clean out build path
build_output="$asr_base"transcriber/build/output/


# archive any output in the build output folder- we can use these transcriptions
# for new examples of good or bad calls for the model.
csr_destination=/mnt/callratings/transcriptions/"$filedate"/$server_num/csr/
customer_destination=/mnt/callratings/transcriptions/"$filedate"/$server_num/customer/

mkdir -p $csr_destination
mkdir -p $customer_destination


# path to model file
model_path=/mnt/models/attention_model/best_model
vocab_path=/mnt/models/attention_model/call_vocab


# directory for output score fil

output=/mnt/callratings/daily_scores/"$filedate"/server"$server_num".csv

# define directory where to put wav files that have been assigned to this server.
wavpath=/mnt/callratings/run/"$filedate"/test"$server_num"/
mkdir -p $wavpath

# This loop converts all *.opus files to *.wav
for f in "$wavpath"*.opus; do
    #get the basename of file, and replace the ".opus" with ".wav"
        f2=${f%.*}".wav"

    # use opusdec to convert the opus file to a wav file with sampling rate 8000
       opusdec --rate 8000 $f $f2
done

# Delete all .opus files, since they are no longer needed
rm -f "$wavpath"*.opus

#split wav file into customer and csr channel

python split_arrange.py $wavpath

# call batch.sh on each directory
# 16 directories because 16 cores
# This may seem dumb but it forces the server to use all 16 cores.

sudo rm -rf "$asr_base"transcriber/build/
sudo mkdir -p $build_output

for dir in "$wavpath"csr/*/; do
	sudo "$asr_base"transcriber/batch.sh "$dir"* &
	sleep 1
done
wait
sleep 1

# move transcripts to destination

sudo find $build_output -type f -exec mv {} $csr_destination \;

# rm -rf "$asr_base"transcriber/build/*
# mkdir -p $build_output

# for j in {0..15}; do
# 	"$asr_base"transcriber/batch.sh "$wavpath"customer/"$j"/* &
# 	sleep 1
# done
# wait
# sleep 1

# cp "$build_output"* $customer_destination


# Clean up and reorganize the wav files.
# find $wavpath -name '*.wav' -exec mv {} $wavpath \;

# Once this particular servers workload is done it will then score the calls it transcribed
mkdir -p /mnt/callratings/daily_scores/"$filedate"/

# call score
source activate call_scoring

parent_dir=$(cd ../ && pwd)

PYTHONPATH=$parent_dir python transcripts_to_score.py --csr_path $csr_destination --customer_path $customer_destination --model_path $model_path --vocab_path $vocab_path \
--out_path $output  

# At this point, the wav files are no longer needed and can be deleted.
#This command has been commented out for now.
# rm -f "$wavpath"*.wav


# Clear all the files that are older than 2 weeks to ensure that the server does not run out of space.

current_date=$(date +"%Y-%m-%d")

cd /mnt/callratings/

last_date=$(date --date="$current_date -2 week" +"%Y-%m-%d")
find run/ ! -newermt "$last_date" | xargs rm -rfv
find run/ -empty -type d -delete

# last_date=$(date --date="$current_date -1 month" +"%Y-%m-%d")
# find transcriptions/ ! -newermt "$last_date" | xargs rm -rfv
# find transcriptions/ -empty -type d -delete

sudo rm -rf "$asr_base"transcriber/build

# sudo rm -rf "$asr_base"transcriber/build/output
# sudo rm -rf "$asr_base"transcriber/build/audio
# sudo rm -rf "$asr_base"transcriber/build/trans
# sudo rm -rf "$asr_base"transcriber/build/diarization
