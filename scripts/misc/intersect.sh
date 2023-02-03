

##collecting cryptic sites from the baseline scores of each window for all genes
# Purpose: To intersect two BED files and append the results to another BED file, e.g. to get the constitutive sites scored by the model for each gene to get average for thresholding

# Define the input file 1
input_file_1 = "<POSITIVE_CPA_sites.bed>"

# Define the output file
output_file = " "

# Loop through all files in a given directory <dir> with a <ext> extension, these are the bed files with all the baseline scores for each window, average with all CPA sites in genome
for f in <dir>/*<ext>:
    # Use the bedtools intersect command to intersect the two BED files
    # -wa: Write the original entry in A for each overlap
    # -wb: Write the original entry in B for each overlap
    # -s: force strandedness. If the feature in B is stranded, use the opposite strand for the feature in A.
    # -f: Minimum overlap as a fraction of A.
    # -r: Require same strandedness.
    # -a: The A input file
    # -b: The B input file
    bedtools intersect -wa -wb -s -f 1.0 -r -a $input_file_1 -b $f >> $output_file
