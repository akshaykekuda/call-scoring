#!/bin/bash
# This script combines the scoring files fromt the previous day into one csv
# that is to be uploaded to OneDrive.
# After the file is created, the script will preprocess *.opus files for the daily run.

# Create file to be uploaded to onedrive


# Begin preprocessing.

# Default to one day ago.
folder_date="$(date -d "1 days ago" '+%Y-%m-%d')"

# You can set -d particular date to force the script 
# to back date.
while getopts d: option
do
case "${option}"
in
d) folder_date=${OPTARG};;
esac
done

# replace - with _ to get a new day of files.
read_date=$(tr '-' '_' <<<"$folder_date")

BASEDIR=/mnt/callratings


# clean up old files if they exist
rm -rf $BASEDIR/run/$folder_date/test01 $BASEDIR/run/$folder_date/test02 
rm -rf $BASEDIR/run/$folder_date/test03 $BASEDIR/run/$folder_date/test04 
rm -rf $BASEDIR/run/$folder_date/test05

# make new directories for each server
mkdir -p $BASEDIR/run/$folder_date/test01 $BASEDIR/run/$folder_date/test02 
mkdir -p $BASEDIR/run/$folder_date/test03 $BASEDIR/run/$folder_date/test04 
mkdir -p $BASEDIR/run/$folder_date/test05

# copy the days files that are meant to be run
yes | cp -rf $BASEDIR/CustomerService/$read_date/*.opus $BASEDIR/run/$folder_date/
yes | cp -rf $BASEDIR/Sales/$read_date/*.opus $BASEDIR/run/$folder_date/

# Rename each file so that they can be assigned to a server
$BASEDIR/rename.sh $BASEDIR/run/$folder_date/
$BASEDIR/random.sh $BASEDIR/run/$folder_date/

# Move the opus files to the locations.
mv $BASEDIR/run/$folder_date/1-*.opus $BASEDIR/run/$folder_date/test01/
mv $BASEDIR/run/$folder_date/2-*.opus $BASEDIR/run/$folder_date/test02/
mv $BASEDIR/run/$folder_date/3-*.opus $BASEDIR/run/$folder_date/test03/
mv $BASEDIR/run/$folder_date/4-*.opus $BASEDIR/run/$folder_date/test04/
mv $BASEDIR/run/$folder_date/5-*.opus $BASEDIR/run/$folder_date/test05/
