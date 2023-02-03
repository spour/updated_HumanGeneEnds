#!/bin/bash

# This bash script is designed to perform the following steps on a BED file from "...step1.py":

# Convert the BED file to use tab as the separator
# Add "(" and ")" to the 4th column
# Extend the regions by 500 bp to the right using bedtools slop
# Shift the right end of the region if the strand is "+"
# Sort the resulting file based on chromosome name and starting position


module load bedtools
#1 is file containing paths of files outputted from "...step1.py", so bed files
#where "$2" is the size of the window the model originally scored - 1, e.g. 500 bp in the baseline model would get 499, 140 in the cryptic model is 139
#$3 is output suffix e.g. ".fix"

fix_bed () {

	fixed_bed=$(awk -vOFS='\t' -F'[(|)|\t]' '{print $1,$2,$3,$4"("$5")"$6,$8,$5}' "$1" | bedtools slop -l 0 -r $2 -s -i - -g <path to hg38.genome>  | awk -vOFS='\t' '{if($6=="+")print $1,$2+1,$3+1,$4,$5,$6; else if($6=="-") print $0}' | sort -k1,1 -k2,2n)
  
	echo "$fixed_bed"
}

while read line; do echo $line; fix_bed $line $2 > "$line""$3"; done <$1
#fix_bed $1 > $2

