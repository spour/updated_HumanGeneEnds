import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='convert scored format to bed and return score coordinates')
parser.add_argument("-i", "--infile", type=str, help="list of infile paths of RF_scored output e.g. ../../full_feat_vects/results_13102022/ZNF648::chr1:182054069-182061712(-).scored", required=True)
parser.add_argument("-out", "--outfile", type = str, help="scored out file location", required=True)
args = parser.parse_args()

def is_unique(s):
    """Check if all values in a series are the same.
    
    Arguments:
        s {pandas Series} -- Series to check.
    
    Returns:
        bool -- True if all values are the same, False otherwise.
    """
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()
  
def fix_to_bed(filepath_RF_scored, args):
    """Convert the scored file to a BED file format.
    
    Arguments:
    filepath_RF_scored -- the path to the file that has been scored
    args -- the command line arguments
    
    Returns:
    None
    """
    tt = pd.read_csv(filepath_RF_scored, sep = '\t', header=None)
    tt.insert(loc=0, column='B', value="ignore")
    tt.insert(loc=0, column='A', value="ignore")
    tt.columns  = ['ignore', 'gene', 'full_name', "score"]
    tt.rename({0: 'ignore', 1: 'gene', 2: 'full_name', 3:"score"}, axis=1, inplace=True)
    tt= pd.concat([tt[["gene","full_name", "score"]],tt["full_name"].str.split('\(|\)|_|chr', expand=True)], axis=1)
    tt.columns = ["gene", "full_name", "score", "ignore", "gene_coor", "strand", "trash", "lxn"]
    tt= pd.concat([tt[[ "full_name", "score", "strand", "lxn"]],tt["gene_coor"].str.split(':|-', expand=True)], axis=1)
    tt = tt[pd.to_numeric(tt['lxn'], errors='coerce').notnull()]
    assert is_unique(tt["strand"]), "Different strands detected"
    tt.rename({0: 'chr', 1: 'start', 2: 'end'}, axis=1, inplace=True)
    tt["score_coordinate"] = 0
    tt[["start", "end", "lxn"]] = tt[["start", "end", "lxn"]].apply(pd.to_numeric)
    tt.loc[(tt['strand'] == '+'), "score_coordinate"] =tt["start"]+tt["lxn"]
    tt.loc[(tt['strand'] == '-'), "score_coordinate"] = tt["end"]-tt["lxn"]
    tt = tt[["chr", "score_coordinate", "score_coordinate", "full_name", "lxn", "score"]]
    tt["chr"] = "chr" +tt['chr']
    tt.iloc[:, 1] = tt.iloc[:, 1]-1
    tt.iloc[:, 4] = "."
    outname = os.path.basename(filepath_RF_scored)
    base=outname
    print(base)
    
    
    
with open(args.infile) as file:
    for line in file:
      print(line)
      fix_to_bed(line.rstrip())



