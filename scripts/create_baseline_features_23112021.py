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
#import pyarrow as pa
#import pyarrow.parquet as pq
import time
import argparse
from Bio import SeqIO
import argparse 
print("imports successful")
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fasta", type=str, help="sequences to score", required=True)
parser.add_argument("-pwms", "--pwms", type=str, help="file containing list of paths of the PWMs you want", required=True)
parser.add_argument("-out", "--outfile", type = str, help="gz out file", required=True)
#parser.add_argument("-t", "--thresh", type=float, help="threshold for accuracy prediction", default=0)
args = parser.parse_args()


#####FUNCTIONS################
def ematrix(m):
  """
  Energy matrix according to https://static-content.springer.com/esm/art%3A10.10
  38%2Fnbt.1893/MediaObjects/41587_2011_BFnbt1893_MOESM84_ESM.pdf page 4 and
  https://static-content.springer.com/esm/art%3A10.1038%2Fnbt.2486/MediaObjects/
  41587_2013_BFnbt2486_MOESM15_ESM.pdf page 40. Takes in np array and outputs array
  """
  # neglog= -1*np.log(m)
  m = -1*np.log(m+0.0001)
  # print(m)
  x= np.amin(m, axis=0)
  # print(x.shape)
  # print(m.shape)
  ematric = np.subtract(m, x[None, ...])
  #ematric = np.subtract(matric1, x[...,None])
  return ematric

#this is set up for 1d convolutions where examples
#have dimensions (len, num_channels)
#the channel axis is the axis for one-hot encoding.
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

from scipy.signal import convolve, correlate
import numpy as np
def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

from skimage.util.shape import view_as_windows

def string_to_strided_app(string, L, s):
  """
  string is the string of choice; L is the size of the window you're considering, s is the step size,
  returns array of arrays of the split window
  """
  string = string.upper()
  string_arr = np.array(list(string))
  split_arr = strided_app(string_arr, L, s)
  # print(split_arr)
  no_ohe = [''.join(row) for row in split_arr.tolist()]
  ohe = np.apply_along_axis(one_hot_encode_along_channel_axis, -1, split_arr)
  return ohe, no_ohe

def others(dna_seqs):
  """
  takes array like ["AGCGCATCG", "CGACTAGCA", "CAGTCTAGCA"] and for each bin counts the features.
  then it joins them to make a feature vector of size (nbins*nfeatures[5]) and it is in order like
  "ca|ta [bin1] ca|ta [bin2] ca|ta [bin3]...
  """
  import regex as re
  counts = np.array([(len(re.findall("CA|TA",dna_seq, overlapped = True)),
             len(re.findall("GT|TG|TT|GG|T|G|TTT|TGG|TGT|GTG",dna_seq, overlapped = True)),
             len(re.findall("TATTTT|TGTTTT|TTTTTT",dna_seq, overlapped = True)),
             len(re.findall("TTTT",dna_seq, )),
             len(re.findall("TGTA",dna_seq, overlapped = True))) for dna_seq in dna_seqs])
  counts = counts.swapaxes(0,1).reshape(1, -1)
  kmers = ["CA_UA", "DSEs", "poly_U_kmers", "poly_U_kmer", "UGUA_kmers"]
  names = [j+"_"+str(i)  for j in kmers for i in range(1, int(counts.shape[1]/5)+1)]
  bin = ["int64"] *  (int(counts.shape[1]))
  dtypes = list(zip(names, bin))
  counts.dtype = dtypes
  return counts

def string_to_strided_app_alt(string, L, s):
  """
  string is the string of choice; L is the size of the window you're considering, s is the step size,
  returns array of arrays of the split window
  """
  string = string.upper()
  string_arr = np.array(list(string))
  split_arr = strided_app(string_arr, L, s)
  # print(split_arr)
  no_ohe = [''.join(row) for row in split_arr.tolist()]
  ohe = np.apply_along_axis(one_hot_encode_along_channel_axis, -1, split_arr)
  return ohe, no_ohe, split_arr


def ohe_to_seq(ohe):
  """
  takes sequence like [0,1,0,0] etc and turns it to "C"
  """
  encoded_sequences =ohe
  sequence_characters = np.chararray(ohe.shape)
  sequence_characters[:] = 'N'
  for i, letter in enumerate(['A', 'C', 'G', 'T']):
    try:
              letter_indxs = encoded_sequences[ :,i] == 1
              sequence_characters[letter_indxs] = letter
    except:
              letter_indxs = (encoded_sequences[ :,i] == 1).squeeze()
              sequence_characters[letter_indxs] = letter
  return ''.join([seq.decode('utf-8') for seq in sequence_characters[:,1]])


from numpy.lib.recfunctions import unstructured_to_structured
def break_and_score_indiv_pwm(seq, macrowindow_size, macrowindow_stride,
                              microwindow_size, microwindow_stride, pwm, name):
  """
  this takes in a sequence of arbitrary length (seq) and divides it into windows of size
  macrowindow_size with macrowindow_stride. it then takes each smaller macrowindow and
  subdivides into microwindow_size with microwindow_stride. then it takes the pwm
  and scores and produces the highest affinity per position so your output is
  (len(seq)-macrowindow_size)/macrowindow_stride+1*(macrowindow_size-microwindow_size)/microwindow_stride+1.
  E.g. for aleksei's you take your whole gene and divide into sliding windows of 500 with 1 step, then for each
  500 window you subdivide into window of 30 with step 10 and and for each window you score with the pwm and take
  the highest affinity score after transforming.
  """
  ohee = one_hot_encode_along_channel_axis(seq)
  ohee_500 = np.squeeze(view_as_windows(ohee, (macrowindow_size, 4), step=(macrowindow_stride,4)), 1)
  l1 = np.squeeze(view_as_windows(ohee_500, (1,microwindow_size,4), step=(1,microwindow_stride,4)),( 2,3))
  jj=np.squeeze(view_as_windows(l1, (1,1, pwm.shape[-1], 4), step = (1,1,1,4)), (3,4,5))
  jjh = np.einsum('...ij, ...ij -> ...', jj, np.transpose(pwm))
  jjhi = 1/(1+np.exp(jjh))
  maximm = np.max(jjhi, axis = 2)
  names =[name+"_"+str(i) for i in range(1, maximm.shape[1]+1)]
  bin = ["float64"] * maximm.shape[1]
  dtypes = list(zip(names, bin))
  maximm = unstructured_to_structured(maximm, dtype=np.dtype(dtypes))
  # maximm.dtype  = dtypes
  return maximm

pd.options.display.precision = 16
pd.options.display.max_columns = None


#READ IN DATA 
pwms = {}
with open(args.pwms) as pwm_lists: # takes the list of pwm paths and reads each line into a dictionary of pwm ematrix arrays
  for line in pwm_lists:
    line = line.rstrip()
    with open(line) as handle:
      name =os.path.basename(line)
      motif = motifs.read(handle, "pfm-four-columns")
      motif_array = np.array(list(motif.pwm[x] for x in sorted(motif.pwm.keys())))
      pwms[name] = ematrix(motif_array)
print("pwms read in")
names_id = []
from pathlib import Path
for i,record in enumerate(SeqIO.parse(args.fasta, "fasta")):
  id = record.id
  seq = str(record.seq).upper()
  if len(seq)<500:
    continue
  scores = [pd.DataFrame(break_and_score_indiv_pwm(seq, 500, 1, 30, 10, pwms[name], name))  for name in list(pwms)]
  # total_scores = pd.concat(scores)
  ohee = one_hot_encode_along_channel_axis(seq)
  # macrowindow_size, macrowindow_stride =500, 1
  ohee_500 = np.squeeze(view_as_windows(ohee, (500, 4), step=(1,4)), 1)
  t = [others(string_to_strided_app_alt(ohe_to_seq(ohee_500[x]), 30, 10)[1]) for x in range(ohee_500.shape[0])]
  counts = pd.DataFrame(np.array(t).squeeze((1,2)))
  assert counts.shape[0] == scores[0].shape[0]
  scores.append(counts)
  total = pd.concat(scores, axis =1 )
  names_id =[id+"_"+str(i) for i in range(total.shape[0])]
  dictionary = dict(zip(list(np.arange(total.shape[0])), names_id))
  total.rename(index=dictionary, inplace=True)
  out_fasta = Path(args.fasta).stem
  out_id = re.sub('[^a-zA-Z0-9 \n\.]', '_', id).strip('_')
  #total.to_csv("17022022_testing_negatives.csv", sep = "\t", header = False,  mode='a')
  total.to_csv(args.outfile, sep = "\t", header = False,  mode='a')
  print(id+": done")
print("done")
