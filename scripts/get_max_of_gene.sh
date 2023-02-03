#Purpose: from the ensemble model mutiplied scores, get the max for each gene and write to file.
# This function takes in two parameters:
# 1. A directory containing files that end with "*merged"
# 2. A file name to write the top line of each merged file to

find_top_line() {
  local dir=$1
  local out_file=$2

  # loop through each file in the directory
  for f in $dir/*merged; do
    # check if the file exists and is not empty
    if [ -s $f ]; then
      # sort the file by column 15 in descending order and take the first line (top line)
      top=$(sort -k15,15 -gr $f | head -1)
      # write the top line to the output file
      echo $top >> $out_file
    fi
  done
}
