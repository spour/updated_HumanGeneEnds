import numpy as np
import pandas as pd
import tensorflow as tf
from Bio import SeqIO
import regex as re
from skimage.util.shape import view_as_windows
pd.set_option("display.precision", 8)
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fasta", type=str, help="fa files", required=True)
parser.add_argument("-m", "--model", type=str, help="tf_model", required=True)
#parser.add_argument("-out", "--outfile", type=str, help="output_prefix", required = True)
args = parser.parse_args()
print(args.fasta, args.model)
physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:
  config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
  print("NO GPU")
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
print("configured")
model = tf.keras.models.load_model(args.model, compile=False)
print("loaded")
def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))


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

def one_hot_encode_along_channel_axis(sequence):
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
from Bio import SeqIO
#seq_scores={}
import os
base=os.path.basename(args.fasta)
fasta_base_str = os.path.splitext(base)[0]
model_base = os.path.basename(args.model)
model_base_str = os.path.splitext(model_base)[0]

for i,record in enumerate(SeqIO.parse(args.fasta, "fasta")):
# for i,record in enumerate(SeqIO.parse("./myfile.txt", "fasta")):
  id = record.id
  seq = str(record.seq).upper()
  scores = []
  if len(seq)<500:
    continue
  seq_scores={}
  for j, x in enumerate(string_to_strided_app_alt(seq, 500, 1)[0]):
    score = model.predict(x[None, :])
    seq_scores[id+"_"+str(j)]= [float(score[0])]
  df = pd.DataFrame.from_dict(seq_scores, orient="index")
  df.to_csv("/scratch/spour98/scoring_aleksei_15112021/redo_04012022/training_chrsplit_baseline_16022022/CNN_baseline/testing_v_training/testing"+id+"_"+model_base_str+"_scored", sep = '\t')
  print("DONE: ", id)

#ALL = pd.DataFrame.from_dict(seq_scores, orient="index")
#out = args.model+"_"+args.fasta+"CNN_predictions.csv"
#ALL.to_csv(out, sep = "\t")

