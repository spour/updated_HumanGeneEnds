#Purpose: merge overlapping/adjacent regions 
#Reads the input file "collected_global_cryptic_sites" using the "cat" command.
#Sorts the data by the first and second columns using the "sort" command.
#Merges overlapping regions using the "bedtools merge" command with a maximum distance of 30 and preserving the strand information.
#Combines the values in columns 5, 2, 3, 6, and 4 using the "collapse" option.
#Writes the merged and combined data to the output file "collected_global_cryptic_sites.merged".
cat collected_global_cryptic_sites | sort -k1,1 -k2,2n  | bedtools merge -d 30 -s -i - -c  5,2,3,6,4 -o collapse,collapse,collapse,distinct,collapse > collected_global_cryptic_sites.merged
