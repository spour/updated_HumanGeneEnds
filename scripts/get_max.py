#Purpose: in adjacent regions in collected_global_cryptic_sites.merged, select the window with the highest score
import pandas as pd
import numpy as np
import sys

def get_argmax(mx):
    """
    Return the index of the maximum value in the list mx.
    """
    return np.argmax(mx)

def return_idx(listx, x):
    """
    Return the value in listx at index x.
    """
    return listx[x]

def process_input(input_file, output_dir):
    """
    Read a tab-separated input file, extract information and write a new file.

    Parameters:
        input_file (str): the path to the input file.
        output_dir (str): the directory where the output file will be written.

    Returns:
        None
    """
    x = pd.read_csv(input_file, sep='\t', header=None)
    x[8] = x[3].apply(lambda x: [float(y) for y in x.split(',')])  #scores
    x[9] = x[4].apply(lambda x: [int(y) for y in x.split(',')]) #start
    x[10] = x[5].apply(lambda x: [int(y) for y in x.split(',')]) #end
    x[11] = x[7].apply(lambda x: [str(y) for y in x.split(',')]) #names
    x[12] = x[8].apply(get_argmax)
    x[13] = x.apply(lambda x: return_idx(x[8],x[12]), axis=1) #max score
    x[14] = x.apply(lambda x: return_idx(x[9],x[12]), axis=1) #start
    x[15] = x.apply(lambda x: return_idx(x[10],x[12]), axis=1) #end
    x[16] = x.apply(lambda x: return_idx(x[11],x[12]), axis=1) #name
    full = x[[0, 14,15,16,13,6]]
    out = output_dir
    full.to_csv(out+"/"+"cryptic_regions", sep='\t', header=False, index=False)
