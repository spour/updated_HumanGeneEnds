print("running")
from Bio import motifs
import numpy as np
from scipy.signal import convolve, correlate
from skimage.util.shape import view_as_windows
import regex as re
from numba import jit, njit
import random
import os
import random
import pandas as pd
from scipy.signal import convolve, correlate
import numpy as np
import time
import argparse
from Bio import SeqIO
import argparse 
from pathlib import Path
import joblib
from skimage.util.shape import view_as_windows
from numpy.lib.recfunctions import unstructured_to_structured
pd.options.display.precision = 16
pd.options.display.max_columns = None

print("imports successful")




#####FUNCTIONS################
def ematrix(m):
    """
    Energy matrix according to https://static-content.springer.com/esm/art%3A10.10
    38%2Fnbt.1893/MediaObjects/41587_2011_BFnbt1893_MOESM84_ESM.pdf page 4 and
    https://static-content.springer.com/esm/art%3A10.1038%2Fnbt.2486/MediaObjects/
    41587_2013_BFnbt2486_MOESM15_ESM.pdf page 40. Takes in np array and outputs array
    """
    # neglog= -1*np.log(m)
    m = -1 * np.log(m + 0.0001)
    x = np.amin(m, axis=0)
    ematric = np.subtract(m, x[None, ...])
    return ematric

def one_hot_encode_along_channel_axis(sequence):
    "returns one hot encoding of given sequence, useful for scoring with pwms because then you can just take the inner product"
    to_return = np.zeros((len(sequence),4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                 sequence=sequence, one_hot_axis=1)
    return to_return

def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis==0 or one_hot_axis==1
    if (one_hot_axis==0):
        assert zeros_array.shape[1] == len(sequence)
    elif (one_hot_axis==1):
        assert zeros_array.shape[0] == len(sequence)
    #will mutate zeros_array
    for (i,char) in enumerate(sequence):
        if (char=="A" or char=="a"):
            char_idx = 0
        elif (char=="C" or char=="c"):
            char_idx = 1
        elif (char=="G" or char=="g"):
            char_idx = 2
        elif (char=="T" or char=="t"):
            char_idx = 3
        elif (char=="N" or char=="n"):
            continue #leave that pos as all 0's
        else:
            raise RuntimeError("Unsupported character: "+str(char))
        if (one_hot_axis==0):
            zeros_array[char_idx,i] = 1
        elif (one_hot_axis==1):
            zeros_array[i,char_idx] = 1


def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))


def string_to_strided_app(string, L, s):
    """
    Converts a string into an array of arrays split by a strided window.

    Parameters:
        string (str): The input string to be processed.
        L (int): The size of the window to be considered.
        s (int): The step size for the strided window.

    Returns:
        tuple: A tuple of two arrays. The first is an array of one-hot encoded arrays of split windows.
               The second is an array of strings of the split windows.
    """
    # Convert string to uppercase
    string = string.upper()

    # Convert string to a numpy array
    string_arr = np.array(list(string))

    # Split the string using the strided window
    split_arr = strided_app(string_arr, L, s)

    # Convert the split arrays back to strings
    no_ohe = [''.join(row) for row in split_arr.tolist()]

    # One-hot encode the split arrays
    ohe = np.apply_along_axis(one_hot_encode_along_channel_axis, -1, split_arr)

    return ohe, no_ohe


def others(dna_seqs):
    """
    Counts specific features in each DNA sequence and returns a feature vector.
    
    Parameters:
    dna_seqs (List[str]): A list of DNA sequences, for example: ["AGCGCATCG", "CGACTAGCA", "CAGTCTAGCA"]
    
    Returns:
    numpy.ndarray: A feature vector of size (nbins * nfeatures[5]), where nbins is the number of DNA sequences
                   and nfeatures is the number of features counted. The features are joined in order, for example:
                   "ca|ta [bin1] ca|ta [bin2] ca|ta [bin3]..."
    
    """
    counts = np.array([(len(re.findall("CA|TA", dna_seq, overlapped=True)),
                       len(re.findall("GT|TG|TT|GG|T|G|TTT|TGG|TGT|GTG", dna_seq, overlapped=True)),
                       len(re.findall("TATTTT|TGTTTT|TTTTTT", dna_seq, overlapped=True)),
                       len(re.findall("TTTT", dna_seq)),
                       len(re.findall("TGTA", dna_seq, overlapped=True))) for dna_seq in dna_seqs])
    counts = counts.swapaxes(0, 1).reshape(1, -1)
    kmers = ["CA_UA", "DSEs", "poly_U_kmers", "poly_U_kmer", "UGUA_kmers"]
    names = [j + "_" + str(i) for j in kmers for i in range(1, int(counts.shape[1]/5)+1)]
    bin = ["int64"] * (int(counts.shape[1]))
    dtypes = list(zip(names, bin))
    counts.dtype = dtypes
    return counts


def string_to_strided_app_alt(string, L, s):
    """
    Converts a string into an array of arrays split by a sliding window of size L with a step size of s.

    Parameters:
    string (str): The input string.
    L (int): The size of the sliding window.
    s (int): The step size of the sliding window.

    Returns:
    tuple: A tuple containing:
        - ohe (np.array): An array of one-hot-encoded arrays of the split window.
        - no_ohe (list): A list of strings formed by joining the elements of each row in the split array.
        - split_arr (np.array): The original split array of the string.
    """
    string = string.upper()
    string_arr = np.array(list(string))
    split_arr = strided_app(string_arr, L, s)
    no_ohe = [''.join(row) for row in split_arr.tolist()]
    ohe = np.apply_along_axis(one_hot_encode_along_channel_axis, -1, split_arr)
    return ohe, no_ohe, split_arr



def ohe_to_seq(ohe):
  """
  Converts One Hot Encoded sequence to characters
  Parameters:
  ohe (list): One Hot Encoded sequence in the form of [0,1,0,0]

  Returns:
  str: Decoded sequence in the form of 'C'

  Example:
  ohe_to_seq([0,1,0,0]) -> 'C'
  """
  encoded_sequences = ohe
  sequence_characters = np.chararray(ohe.shape)
  sequence_characters[:] = 'N'
  for i, letter in enumerate(['A', 'C', 'G', 'T']):
      try:
          letter_indxs = encoded_sequences[:, i] == 1
          sequence_characters[letter_indxs] = letter
      except:
          letter_indxs = (encoded_sequences[:, i] == 1).squeeze()
          sequence_characters[letter_indxs] = letter
  return ''.join([seq.decode('utf-8') for seq in sequence_characters[:, 1]])



def break_and_score_indiv_pwm(seq, macrowindow_size, macrowindow_stride,microwindow_size, microwindow_stride, pwm, name):
  """
  Divides a sequence into windows and scores them based on a PWM.

  Parameters:
  seq (str): The sequence to be divided and scored.
  macrowindow_size (int): The size of the larger windows.
  macrowindow_stride (int): The stride for dividing the sequence into larger windows.
  microwindow_size (int): The size of the smaller windows.
  microwindow_stride (int): The stride for dividing the larger windows into smaller windows.
  pwm (np.ndarray): The position weight matrix used for scoring.
  name (str): A name to be used for naming the resulting data structure.

  Returns:
  np.ndarray: A structured numpy array of the scores.

  Example:
  The function takes a gene and divides it into sliding windows of size 500 with stride 1. 
  Then, each 500 window is subdivided into windows of size 30 with stride 10. Finally, 
  each 30 window is scored using the pwm and the highest affinity score is taken.
  """
  ohee = one_hot_encode_along_channel_axis(seq)
  ohee_500 = np.squeeze(view_as_windows(ohee, (macrowindow_size, 4), step=(macrowindow_stride,4)), 1)
  l1 = np.squeeze(view_as_windows(ohee_500, (1,microwindow_size,4), step=(1,microwindow_stride,4)),( 2,3))
  jj=np.squeeze(view_as_windows(l1, (1,1, pwm.shape[-1], 4), step = (1,1,1,4)), (3,4,5))
  jjh = np.einsum('...ij, ...ij -> ...', jj, np.transpose(pwm))
  jjhi = 1/(1+np.exp(jjh))
  maximm = np.max(jjhi, axis = 2)
  names =[name+"_"+str(i) for i in range(1, maximm.shape[1]+1)]
  bin_ = ["float64"] * maximm.shape[1]
  dtypes = list(zip(names, bin_))
  maximm = unstructured_to_structured(maximm, dtype=np.dtype(dtypes))
  return maximm



def maxentscan(dna_seqs):
  """
  Calculates the maxentscan scores for a one-hot encoded DNA sequence.
  Parameters:
      dna_seqs: a one-hot encoded DNA sequence in the form of a numpy array of shape (500, 4)

  Returns:
      df_struct: a structured numpy array of the maximum maxentscan scores, with dtype "float"

  The input sequence is split into 30 20bp windows with a 10bp stride and each window is further split into 9bp segments for maxentscan scoring. The function returns the maximum score among all the windows.
  """
  from maxentpy import maxent
  from maxentpy.maxent import load_matrix5

  matrix5 = load_matrix5()
  df = np.array([max([2**maxent.score5(y, matrix=matrix5)/(2**maxent.score5(y, matrix=matrix5)+1) for y in string_to_strided_app_alt(y, 9, 1)[1]]) for y in string_to_strided_app_alt(dna_seqs, 20, 10)[1]])
  df = np.expand_dims(df, 0)
  kmers = ["MES"]
  names = [j+"_"+str(i)  for j in kmers for i in range(1, int(df.shape[1])+1)]
  bin_ = ["float"] *  (int(df.shape[1]))
  dtypes = list(zip(names, bin_))
  df_struct = unstructured_to_structured(df, dtype=np.dtype(dtypes))
  return df_struct



def mod_rhyb(dna_seqs, rnahyb_lib):
    """
    This function calculates the RNAhyb score for a given DNA sequence.
    
    Parameters:
    dna_seqs (str): The input DNA sequence
    rnahyb_lib (np lib):library of 7mer scores from RNAhyb
    
    Returns:
    gg (np.ndarray): The RNAhyb score in a structured array format
    
    """
    read_dictionary = np.load(rnahyb_lib, allow_pickle='TRUE').item()
    lis = string_to_strided_app_alt(dna_seqs, 20, 10)[1]
    max_list = []
    for subsubseq in lis:
        full_to_140_to_20_to_12 = min([read_dictionary[y] for y in string_to_strided_app_alt(subsubseq, 7, 1)[1]])
        max_list.append(full_to_140_to_20_to_12)
    df = pd.DataFrame(max_list)
    kmers = ["RNAhyb"]
    names = [j + "_" + str(i) for j in kmers for i in range(1, int(df.shape[0]) + 1)]
    bin = ["float"] * (int(df.shape[0]))
    dtypes = list(zip(names, bin))
    gg = unstructured_to_structured(np.expand_dims(np.array(df[0]), 0), dtype=np.dtype(dtypes))
    return gg


#READ IN DATA 
# with open(args.pwms) as pwm_lists:
#   pwms={}
#   for line in pwm_lists:
#     line = line.rstrip()
#     with open(line) as handle:
#       name =os.path.basename(line)
#       motif = motifs.read(handle, "pfm-four-columns")
#       if int(motif.length) >20:
#         continue
#       else:
#         motif_array = np.array(list(motif.pwm[x] for x in sorted(motif.pwm.keys())))
#         pwms[name] = ematrix(motif_array)

def read_pwms(pwm_file):
    """
    This function reads in the PWM files and stores them in a dictionary.
    
    Parameters:
    pwm_file (str): The file path for the list of PWM files
    
    Returns:
    pwms (dict): The dictionary containing the PWMs as numpy arrays
    
    """
    pwms = {}
    with open(pwm_file) as pwm_lists:
        for line in pwm_lists:
            line = line.rstrip()
            with open(line) as handle:
                name = os.path.basename(line)
                motif = motifs.read(handle, "pfm-four-columns")
                if int(motif.length) > 20:
                    continue
                else:
                    motif_array = np.array(list(motif.pwm[x] for x in sorted(motif.pwm.keys())))
                    pwms[name] = ematrix(motif_array)
    print("pwms read in")
    return pwms



# names_id = []
# from pathlib import Path
# for i,record in enumerate(SeqIO.parse(args.fasta, "fasta")):
#   id = record.id
#   seq = str(record.seq).upper()
#   if len(seq)<140:
#     continue
#   scores = [pd.DataFrame(break_and_score_indiv_pwm(seq,140, 1, 20, 10, pwms[name], name))  for name in list(pwms) if pwms[name].shape[1]<=20]
#   scores1 = pd.concat(scores, axis = 1)
#   ohee = one_hot_encode_along_channel_axis(seq)
#   ohee_500 = np.squeeze(view_as_windows(ohee, (20, 4), step=(10,4)), 1)
#   mes_list = []
#   rhyb_list = []
#   for subseq in string_to_strided_app_alt(seq, 140, 1)[1]:
#     subseq = ''.join(i if i != 'N'  else random.choice(["A", "C", "T", "G"]) for i in subseq)
#     mes_list.append(pd.DataFrame(maxentscan(subseq)))
#     rhyb_list.append(pd.DataFrame(mod_rhyb(subseq)))
#   mes_scores = pd.concat(mes_list, axis =0)
#   rnahyb_scores = pd.concat(rhyb_list, axis = 0)
#   mes_scores.reset_index(inplace=True, drop=True)
#   rnahyb_scores.reset_index(inplace=True, drop=True)
#   scores1.reset_index(inplace=True, drop=True)
#   total = pd.concat([mes_scores, rnahyb_scores, scores1], axis =1 )
#   print(total.shape)
#   names_id =[id+"_"+str(i) for i in range(total.shape[0])]
#   dictionary = dict(zip(list(np.arange(total.shape[0])), names_id))
#   total.rename(index=dictionary, inplace=True)
#   out_id = re.sub('[^a-zA-Z0-9 \n\.]', '_', id).strip('_')
#   total.to_csv(args.outfile+".featvect", sep = "\t", header = True,  mode='a')
#   print(id+": done")
# print("done")
def process_fasta(args):
    pwms = args.pwms
    names_id = []
    model = joblib.load(args.model)
    for i,record in enumerate(SeqIO.parse(args.fasta, "fasta")):
        id = record.id
        seq = str(record.seq).upper()
        if args.function == "cryptic":    
            if len(seq) < 140:
                continue
            else:
                scores = [pd.DataFrame(break_and_score_indiv_pwm(seq, 140, 1, 20, 10, pwms[name], name)) 
                          for name in list(pwms) if pwms[name].shape[1] <= 20]
                scores1 = pd.concat(scores, axis = 1)
                ohee = one_hot_encode_along_channel_axis(seq)
                ohee_500 = np.squeeze(view_as_windows(ohee, (20, 4), step=(10, 4)), 1)
                mes_list = []
                rhyb_list = []
                for subseq in string_to_strided_app_alt(seq, 140, 1)[1]:
                    subseq = ''.join(i if i != 'N'  else random.choice(["A", "C", "T", "G"]) for i in subseq)
                    mes_list.append(pd.DataFrame(maxentscan(subseq)))
                    rhyb_list.append(pd.DataFrame(mod_rhyb(subseq)))
                mes_scores = pd.concat(mes_list, axis = 0)
                rnahyb_scores = pd.concat(rhyb_list, axis = 0)
                mes_scores.reset_index(inplace=True, drop=True)
                rnahyb_scores.reset_index(inplace=True, drop=True)
                scores1.reset_index(inplace=True, drop=True)
                total = pd.concat([mes_scores, rnahyb_scores, scores1], axis = 1)
                names_id = [f"{id}_{i}" for i in range(total.shape[0])]
                dictionary = dict(zip(list(np.arange(total.shape[0])), names_id))
                total.rename(index=dictionary, inplace=True)
                out_id = re.sub('[^a-zA-Z0-9 \n\.]', '_', id).strip('_')
                #total.to_csv(args.outfile+".featvect", sep = "\t", header = True,  mode='a')
                f=total
                output = np.column_stack([f.index, model.predict_proba(f.values)[:,1]])
                DF=pd.DataFrame(output)
                DF.to_csv(f"{args.outfile}/{id}.crypt.scored", sep="\t", header=True, index=False)
                print(f"{id}: done")
       if args.function == "baseline":   
            if len(seq) < 500:
                continue
            else:
                scores = [pd.DataFrame(break_and_score_indiv_pwm(seq, 500, 1, 30, 10, pwms[name], name))  for name in list(pwms)]
                ohee = one_hot_encode_along_channel_axis(seq)
                ohee_500 = np.squeeze(view_as_windows(ohee, (500, 4), step=(1,4)), 1)
                t = [others(string_to_strided_app_alt(ohe_to_seq(ohee_500[x]), 30, 10)[1]) for x in range(ohee_500.shape[0])]
                counts = pd.DataFrame(np.array(t).squeeze((1,2)))
                assert counts.shape[0] == scores[0].shape[0]
                scores.append(counts)
                total = pd.concat(scores, axis =1 )
                names_id = [f"{id}_{i}" for i in range(total.shape[0])]
                dictionary = dict(zip(list(np.arange(total.shape[0])), names_id))
                total.rename(index=dictionary, inplace=True)
                out_fasta = Path(args.fasta).stem
                out_id = re.sub('[^a-zA-Z0-9 \n\.]', '_', id).strip('_')
                f=total
                output = np.column_stack([f.index, model.predict_proba(f.values)[:,1]])
                DF=pd.DataFrame(output)
                DF.to_csv(f"{args.outfile}/{id}.crypt.scored", sep="\t", header=True, index=False)
                print(f"{id}: done")
    print("done")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--fasta", type=str, help="sequences to score", required=True)
  parser.add_argument("-pwms", "--pwms", type=str, help="file containing list of paths of the PWMs you want", required=True)
  parser.add_argument("-out", "--outfile", type = str, help="gz out file", required=True)
  parser.add_argument("-m", "--model", type = str, help="model to score with", required=True)
  parser.add_argument('-ft', '--function', type=str, choices=['baseline', 'cryptic'], required=True, help='Choose between baseline (500nt) or cryptic (140nt)')
  #parser.add_argument("-t", "--thresh", type=float, help="threshold for accuracy prediction", default=0)
  args = parser.parse_args()

  process_fasta(args)








#create_baseline_features_23112021_140_rhybshort.py
