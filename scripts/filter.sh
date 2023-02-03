#Purpose: filter genome windows to those above the average score
#This script is used to identify lines in the scored_beds files that meet a certain threshold for the 5th column and writes them to a new file. This can be useful for identifying specific patterns or characteristics in data.
for f in scoredd_beds/*; do awk '{if ($5>=0.689899)print $0}' $f >> collected_global_cryptic_sites; done
