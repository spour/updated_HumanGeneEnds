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

def maxentscan(dna_seqs):
  """
  takes array like array([[0, 0, 1, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       ...,
       [0, 0, 1, 0],
       [1, 0, 0, 0],
       [1, 0, 0, 0]], dtype=int8) one hot encoded sequence (500, 4) and then calcualtes the maxentscan scores in probability 0-1 domain,
       splits the 500bp sequence into 30 with 10 stride and those windows into 9bp windows for maxentscan and then returns the maximum value.
  """
  from maxentpy import maxent
  from maxentpy.maxent import load_matrix5, load_matrix3
  matrix5 = load_matrix5()
  import regex as re
  from numpy.lib.recfunctions import unstructured_to_structured

  df = np.array([max([2**maxent.score5(y, matrix=matrix5)/(2**maxent.score5(y, matrix=matrix5)+1)  for y in string_to_strided_app_alt(y, 9, 1)[1]]) for y in string_to_strided_app_alt(dna_seqs, 20, 10)[1]])
  df = np.expand_dims(df, 0)
  kmers = ["MES"]
  names = [j+"_"+str(i)  for j in kmers for i in range(1, int(df.shape[1])+1)]
  bin = ["float"] *  (int(df.shape[1]))
  dtypes = list(zip(names, bin))
  df_struct = unstructured_to_structured(df, dtype=np.dtype(dtypes))
  return df_struct


#def rnahyb(dna_seqs):
#  """
#  takes array like array([[0, 0, 1, 0],
#       [0, 1, 0, 0],
#       [0, 0, 0, 1],
#       ...,
#       [0, 0, 1, 0],
#       [1, 0, 0, 0],
#       [1, 0, 0, 0]], dtype=int8) one hot encoded sequence (500, 4) and then calcualtes the rnayhyb scores in -e**deltaG,
#       splits the 500bp sequence into 30 with 10 stride  and then returns the  max value.
#  """
#  import subprocess
#  lis = [y for y in string_to_strided_app_alt(dna_seqs, 20, 10)[1]]
#  print("shape rnahyb", len(lis))
#  import tempfile
#  from numpy.lib.recfunctions import unstructured_to_structured
#  with tempfile.NamedTemporaryFile(mode = "w") as tmp:
#    print(tmp.name)
#    tmp.write(''.join('>number:{}\n{}\n'.format(i[0], i[1]) for i in enumerate(lis)))
#    print(tmp)
#    tmp.flush()
#    bashCommand = "RNAhybrid -t {} ACUUACCUG -s 3utr_human -b 1".format(str(tmp.name))
#    process = subprocess.run(bashCommand.split(), text=True, check=True, shell=False, capture_output=True)
#
#  df = pd.DataFrame({'col': list(filter(None, process.stdout.splitlines())) })
#  df = df[df['col'].str.contains("mfe")]
#  df['kd'] = np.array(df['col'].map(lambda x: np.exp(float(re.sub(r'[A-Za-z/:\s]*','', str(x))))))
#  kmers = ["RNAhyb"]
#  names = [j+"_"+str(i)  for j in kmers for i in range(1, int(df.shape[0])+1)]
#  bin = ["float"] *  (int(df.shape[0]))
#  dtypes = list(zip(names, bin))
#  gg = unstructured_to_structured(np.expand_dims(np.array(df['kd']), 0), dtype=np.dtype(dtypes))
#  return gg
def mod_rhyb(dna_seqs):
  read_dictionary = np.load('/scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/rbp_u1_models/ensemble/7mer_RHYB.npy',allow_pickle='TRUE').item()
  lis = string_to_strided_app_alt(dna_seqs, 20, 10)[1]
  #print(lis)
  max_list = []
  for subsubseq in lis:
    full_to_140_to_20_to_12 = min([read_dictionary[y] for y in string_to_strided_app_alt(subsubseq, 7, 1)[1]])
    max_list.append(full_to_140_to_20_to_12)
  df = pd.DataFrame(max_list)
  kmers = ["RNAhyb"]
  names = [j+"_"+str(i)  for j in kmers for i in range(1, int(df.shape[0])+1)]
  bin = ["float"] *  (int(df.shape[0]))
  dtypes = list(zip(names, bin))
  gg = unstructured_to_structured(np.expand_dims(np.array(df[0]), 0), dtype=np.dtype(dtypes))
  return gg

#READ IN DATA 
#pwms = {}
#with open(args.pwms) as pwm_lists: # takes the list of pwm paths and reads each line into a dictionary of pwm ematrix arrays
#  for line in pwm_lists:
#    line = line.rstrip()
#    with open(line) as handle:
#      name =os.path.basename(line)
#      motif = motifs.read(handle, "pfm-four-columns")
#      motif_array = np.array(list(motif.pwm[x] for x in sorted(motif.pwm.keys())))
#      pwms[name] = ematrix(motif_array)
with open(args.pwms) as pwm_lists:
  pwms={}
  for line in pwm_lists:
    line = line.rstrip()
    with open(line) as handle:
      name =os.path.basename(line)
      motif = motifs.read(handle, "pfm-four-columns")
      if int(motif.length) >20:
        continue
      else:
        motif_array = np.array(list(motif.pwm[x] for x in sorted(motif.pwm.keys())))
        pwms[name] = ematrix(motif_array)

print("pwms read in")
names_id = []
from pathlib import Path
for i,record in enumerate(SeqIO.parse(args.fasta, "fasta")):
  id = record.id
  seq = str(record.seq).upper()
  if len(seq)<140:
    continue
  scores = [pd.DataFrame(break_and_score_indiv_pwm(seq,140, 1, 20, 10, pwms[name], name))  for name in list(pwms) if pwms[name].shape[1]<=20]
  # total_scores = pd.concat(scores)
  #ohee = one_hot_encode_along_channel_axis(seq)
  # macrowindow_size, macrowindow_stride =500, 1
  #ohee_500 = np.squeeze(view_as_windows(ohee, (140, 4), step=(1,4)), 1)
  #t = [others(string_to_strided_app_alt(ohe_to_seq(ohee_500[x]), 20, 10)[1]) for x in range(ohee_500.shape[0])]
  #counts = pd.DataFrame(np.array(t).squeeze((1,2)))
  #assert counts.shape[0] == scores[0].shape[0]
  #scores.append(counts)
#  total = pd.concat(scores, axis =1 )
  scores1 = pd.concat(scores, axis = 1)
  #print(scores)
  ohee = one_hot_encode_along_channel_axis(seq)
  #break sequence of 140 apart into 20 with stride 10
  ohee_500 = np.squeeze(view_as_windows(ohee, (20, 4), step=(10,4)), 1)
  # print("hh", ohee_500.shape)
  print(len(string_to_strided_app_alt(seq, 140, 1)[1]))
  mes_list = []
  rhyb_list = []
  for subseq in string_to_strided_app_alt(seq, 140, 1)[1]:
    subseq = ''.join(i if i != 'N'  else random.choice(["A", "C", "T", "G"]) for i in subseq)
    #print(pd.DataFrame(maxentscan(subseq)))
    mes_list.append(pd.DataFrame(maxentscan(subseq)))
    rhyb_list.append(pd.DataFrame(mod_rhyb(subseq)))
  mes_scores = pd.concat(mes_list, axis =0)
  rnahyb_scores = pd.concat(rhyb_list, axis = 0)
  mes_scores.reset_index(inplace=True, drop=True)
  rnahyb_scores.reset_index(inplace=True, drop=True)
  scores1.reset_index(inplace=True, drop=True)
  print(mes_scores.shape, rnahyb_scores.shape)
#  mes_scores, rnahyb_scores = pd.DataFrame(maxentscan(seq)), pd.DataFrame(rnahyb(seq))
  # t = [others(string_to_strided_app_alt(ohe_to_seq(ohee_500[x]), 30, 10)[1]) for x in range(ohee_500.shape[0])]
  # counts = pd.DataFrame(np.array(t).squeeze((1,2)))
  # assert counts.shape[0] == scores[0].shape[0]
  total = pd.concat([mes_scores, rnahyb_scores, scores1], axis =1 )
  print(total.shape)
#  total = pd.concat(scores, axis =1 )
  names_id =[id+"_"+str(i) for i in range(total.shape[0])]
  dictionary = dict(zip(list(np.arange(total.shape[0])), names_id))
  total.rename(index=dictionary, inplace=True)
#  out_fasta = Path(args.fasta).stem
  out_id = re.sub('[^a-zA-Z0-9 \n\.]', '_', id).strip('_')
  total.to_csv(args.outfile+".featvect", sep = "\t", header = True,  mode='a')
  #total.to_csv(args.outfile+"/"+id+".scored", sep = "\t", header = False)
  print(id+": done")
print("done")
