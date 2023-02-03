#Purpose: merge the 500nt RF baseline model windows with the 140nt LR_cryptic model windows to get the ensemble scores

#!/bin/bash

function merge_files () {
    """
    This function takes in a file path as an argument and performs the following steps:
    1. Creates a temporary file from the input file path
    2. Searches for a file with a similar name in a specified directory
    3. Creates a temporary file from the found file
    4. Sorts both temporary files based on the midpoint of each line
    5. Merges the two sorted files based on the midpoint
    6. Adds a new column to the merged data with the product of two specific columns
    7. Returns the merged data
    8. Deletes both temporary files
    """
    FILE_PATH=$(mktemp)
    FILE_PATH_name=$(realpath "$1")
    cp $FILE_PATH_name $FILE_PATH
    FILE_SEARCH=$(echo $(basename "$FILE_PATH_name" ) | cut -d\( -f1)
    FILE_139_name=$(find  /lustre07/scratch/spour98/full_feat_vects/results_04112022_bed -name "$FILE_SEARCH*r139\.fix" )
    FILE_139=$(mktemp)
    cp $FILE_139_name $FILE_139
    FILE_PATH_MID=$( awk -vOFS="\t" '{a=($2+$3)/2; print $0,a}' "$FILE_PATH" | sort -k7 )
    FILE_139_MID=$(awk -vOFS="\t" '{a=($2+$3)/2; print $0,a}' "$FILE_139" | sort -k7)
    MERGED=$(join  -t $'\t' -j 7 -o 1.1,1.2,1.3,1.4,1.5,1.6,1.7,2.1,2.2,2.3,2.4,2.5,2.6,2.7  <(echo "$FILE_PATH_MID")   <(echo "$FILE_139_MID"))
    MERGED=$(echo "$MERGED" | awk -vOFS="\t" '{print $0,$5*$12}' )
    echo "$MERGED"
    rm -f $FILE_PATH $FILE_139
}
#you must supply with file of paths to the 500nt window scored bed files, $2 is the directory, $3 is the suffix e.g. "*merged")
while read line
    do
    output="$(basename $line .bed.r499.fix)"
    merge_files $line > "$2""/""$output""$3"
done <$1
