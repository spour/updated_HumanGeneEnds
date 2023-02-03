#Purpose: get average score of the baseline scores for all the CPA sites in a bed file
awk '!seen[$1$2$3$4$5$6]++' $bed | awk '{ total += $11 } END { print total/NR }'
