import tensorflow as tf
#from dragonn.simulations import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Activation, Flatten, Dense, Dropout, MaxPool1D
import tensorflow.keras as keras
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
from tensorflow.keras.callbacks import EarlyStopping, History
import simdna.synthetic as synthetic
from tensorflow.keras.models import model_from_json
from sklearn.model_selection import train_test_split
import tensorflow.keras.initializers as initializers
from tensorflow.keras.layers import AveragePooling1D
#import shap
from vizsequence import viz_sequence
#import gzip
from tensorflow.keras.layers import GlobalMaxPooling1D
import argparse
import pandas as pd
import sys
from Bio import SeqIO
parser = argparse.ArgumentParser()
# parser.add_argument("-p", "--pos", type=str, help="positive test examples", required=True)
# parser.add_argument("-n", "--neg", type=str, help="negative test examples", required=True)
parser.add_argument("-pte", "--pos_testing", type=str, help="positive test examples", required=True)
parser.add_argument("-nte", "--neg_testing", type=str, help="negative test examples", required=True)
parser.add_argument("-ptr", "--pos_training", type=str, help="positive training examples", required=True)
parser.add_argument("-ntr", "--neg_training", type=str, help="negative training examples", required=True)
parser.add_argument("-outdir", "--outdir", type=str, help="output_prefix", required = True)
parser.add_argument("-outname", "--outname", type=str, help="output_name", required = True)
args = parser.parse_args()
pos_testing_identifiers, pos_testing_sequences = [], []
neg_testing_identifiers, neg_testing_sequences = [], []
pos_training_identifiers, pos_training_sequences = [], []
neg_training_identifiers, neg_training_sequences = [], []

with open(args.pos_testing) as fasta_file:  # Will close handle cleanly
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
        pos_testing_identifiers.append(seq_record.id)
        pos_testing_sequences.append(str(seq_record.seq).upper())

with open(args.pos_training) as fasta_file:  # Will close handle cleanly
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
        pos_training_identifiers.append(seq_record.id)
        pos_training_sequences.append(str(seq_record.seq).upper())

with open(args.neg_testing) as fasta_file:  # Will close handle cleanly
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
        neg_testing_identifiers.append(seq_record.id)
        neg_testing_sequences.append(str(seq_record.seq).upper()) 
               
with open(args.neg_training) as fasta_file:  # Will close handle cleanly
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
        neg_training_identifiers.append(seq_record.id)
        neg_training_sequences.append(str(seq_record.seq).upper()) 
#pos_seqs = np.array([one_hot_encode_along_channel_axis(seq) for seq in pos_sequences])
#neg_seqs = np.array([one_hot_encode_along_channel_axis(seq) for seq in neg_sequences])
# pos = pd.DataFrame({'sequence' : pos_sequences, 'id': pos_identifiers})
# neg = pd.DataFrame({'sequence' : neg_sequences, 'id': neg_identifiers})
# pos['labels'] = 1
# neg['labels']=0
# pos_neg_cat = pd.concat([pos, neg])


# X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(pos_neg_cat['sequence'], pos_neg_cat['labels'], test_size=0.15, random_state=42, stratify = pos_neg_cat['labels'])

#pos = pd.read_csv(args.pos, header = 0, sep = "\t")
#neg = pd.read_csv(args.neg, header = 0, sep = "\t")

#print(pos.shape)
#print(neg.shape)

import numpy as np
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.compat.v1.keras.backend as K1


def make_directory(path, foldername, verbose=1):
    """make a directory"""

    if not os.path.isdir(path):
        os.mkdir(path)
        print("making directory: " + path)

    outdir = os.path.join(path, foldername)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print("making directory: " + outdir)
    return outdir


def run_function_batch(sess, signed_grad, model, placeholders, inputs, batch_size=128):

    def feed_dict_batch(placeholders, inputs, index):
        feed_dict = {}
        for i in range(len(placeholders)):
            feed_dict[placeholders[i]] = inputs[i][index]
        return feed_dict

    N = len(inputs[0])
    num_batches = int(np.floor(N/batch_size))

    values = []
    for i in range(num_batches):
        index = range(i*batch_size, (i+1)*batch_size)
        values.append(sess.run(signed_grad, feed_dict_batch(placeholders, inputs, index)))
    if num_batches*batch_size < N:
        index = range(num_batches*batch_size, N)
        values.append(sess.run(signed_grad, feed_dict_batch(placeholders, inputs, index)))
    values = np.concatenate(values, axis=0)

    return values


def calculate_class_weight(y_train):
    # calculate class weights
    count = np.sum(y_train, axis=0)
    weight = np.sqrt(np.max(count)/count)
    class_weight = {}
    for i in range(y_train.shape[1]):
        class_weight[i] = weight[i]
    return class_weight


def compile_regression_model(model, learning_rate=0.001, mask_val=None, **kwargs):

    optimizer = optimizers(optimizer=optimizer, learning_rate=learning_rate, **kwargs)

    if mask:
        def masked_loss_function(y_true, y_pred):
            mask = tf.cast(tf.math.logical_not(tf.math.equal(y_true, mask_val)), dtype=tf.float32)
            return keras.losses.mean_squared_error(y_true*mask, y_pred*mask)
        loss = masked_loss_function
    else:
        loss = keras.losses.mean_squared_error

    model.compile(optimizer=optimizer, loss=loss)



def compile_classification_model(model, loss_type='binary', optimizer='adam',
                                 learning_rate=0.001, monitor=['acc', 'auroc', 'aupr'],
                                 label_smoothing=0.0, from_logits=False, **kwargs):

    optimizer = optimizers(optimizer=optimizer, learning_rate=learning_rate, **kwargs)

    metrics = []
    if 'acc' in monitor:
        metrics.append('accuracy')
    if 'auroc' in monitor:
        metrics.append(keras.metrics.AUC(curve='ROC', name='auroc'))
    if 'auroc' in monitor:
        metrics.append(keras.metrics.AUC(curve='PR', name='aupr'))

    if loss_type == 'binary':
        loss = keras.losses.BinaryCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing)
    elif loss_type == 'categorical':
        loss = keras.losses.CategoricalCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)



def optimizers(optimizer='adam', learning_rate=0.001, **kwargs):

    if optimizer == 'adam':
        if 'beta_1' in kwargs.keys():
            beta_1 = kwargs['beta_1']
        else:
            beta_1 = 0.9
        if 'beta_2' in kwargs.keys():
            beta_2 = kwargs['beta_2']
        else:
            beta_2 = 0.999
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

    elif optimizer == 'sgd':
        if 'momentum' in kwargs.keys():
            momentum = kwargs['momentum']
        else:
            momentum = 0.0
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

    return optimizer



def clip_filters(W, threshold=0.5, pad=3):

    W_clipped = []
    for w in W:
        L,A = w.shape
        entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=1)
        index = np.where(entropy > threshold)[0]
        if index.any():
            start = np.maximum(np.min(index)-pad, 0)
            end = np.minimum(np.max(index)+pad+1, L)
            W_clipped.append(w[start:end,:])
        else:
            W_clipped.append(w)

    return W_clipped



def meme_generate(W, output_file='meme.txt', prefix='filter'):

    # background frequency
    nt_freqs = [1./4 for i in range(4)]

    # open file for writing
    f = open(output_file, 'w')

    # print intro material
    f.write('MEME version 4\n')
    f.write('\n')
    f.write('ALPHABET= ACGT\n')
    f.write('\n')
    f.write('Background letter frequencies:\n')
    f.write('A %.4f C %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
    f.write('\n')

    for j, pwm in enumerate(W):
        L, A = pwm.shape
        f.write('MOTIF %s%d \n' % (prefix, j))
        f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (L, L))
        for i in range(L):
            f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[i,:]))
        f.write('\n')

    f.close()



def match_hits_to_ground_truth(file_path, motifs, size=30):

    # get dataframe for tomtom results
    df = pd.read_csv(file_path, delimiter='\t')

    # loop through filters
    best_qvalues = np.ones(size)
    best_match = np.zeros(size)
    for name in np.unique(df['Query_ID'].as_matrix()):

        if name[:6] == 'filter':
            filter_index = int(name.split('r')[1])

            # get tomtom hits for filter
            subdf = df.loc[df['Query_ID'] == name]
            targets = subdf['Target_ID'].as_matrix()

            # loop through ground truth motifs
            for k, motif in enumerate(motifs):

                # loop through variations of ground truth motif
                for motifid in motif:

                    # check if there is a match
                    index = np.where((targets == motifid) ==  True)[0]
                    if len(index) > 0:
                        qvalue = subdf['q-value'].as_matrix()[index]

                        # check to see if better motif hit, if so, update
                        if best_qvalues[filter_index] > qvalue:
                            best_qvalues[filter_index] = qvalue
                            best_match[filter_index] = k

    # get the minimum q-value for each motif
    min_qvalue = np.zeros(13)
    for i in range(13):
        index = np.where(best_match == i)[0]
        if len(index) > 0:
            min_qvalue[i] = np.min(best_qvalues[index])

    match_index = np.where(best_qvalues != 1)[0]
    match_fraction = len(match_index)/float(size)

    return best_qvalues, best_match, min_qvalue, match_fraction



def activation_fn(activation):

    if activation == 'exp_relu':
        return exp_relu
    elif activation == 'shift_scale_tanh':
        return shift_scale_tanh
    elif activation == 'shift_scale_relu':
        return shift_scale_relu
    elif activation == 'shift_scale_sigmoid':
        return shift_scale_sigmoid
    elif activation == 'shift_relu':
        return shift_relu
    elif activation == 'shift_sigmoid':
        return shift_sigmoid
    elif activation == 'shift_tanh':
        return shift_tanh
    elif activation == 'scale_relu':
        return scale_relu
    elif activation == 'scale_sigmoid':
        return scale_sigmoid
    elif activation == 'scale_tanh':
        return scale_tanh
    elif activation == 'log_relu':
        return log_relu
    elif activation == 'log':
        return log
    elif activation == 'exp':
        return 'exponential'
    else:
        return activation

def exp_relu(x, beta=0.001):
    return K.relu(K.exp(.1*x)-1)

def log(x):
    return K.log(K.abs(x) + 1e-10)

def log_relu(x):
    return K.relu(K.log(K.abs(x) + 1e-10))

def shift_scale_tanh(x):
    return K.tanh(x-6.0)*500 + 500

def shift_scale_sigmoid(x):
    return K.sigmoid(x-8.0)*4000

def shift_scale_relu(x):
    return K.relu(K.pow(x-0.2, 3))

def shift_tanh(x):
    return K.tanh(x-6.0)

def shift_sigmoid(x):
    return K.sigmoid(x-8.0)

def shift_relu(x):
    return K.relu(x-0.2)

def scale_tanh(x):
    return K.tanh(x)*500 + 500

def scale_sigmoid(x):
    return K.sigmoid(x)*4000

def scale_relu(x):
    return K.relu((x)**3)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.layers import Layer


def dense_layer(input_layer, num_units, activation, dropout=0.5, l2=None, bn=True, kernel_initializer=None):
    if l2:
        l2 = keras.regularizers.l2(l2)
    else:
        l2 = None

    nn = keras.layers.Dense(num_units,
                            activation=None,
                            use_bias=False,
                            kernel_initializer=kernel_initializer,
                            bias_initializer='zeros',
                            kernel_regularizer=l2,
                            bias_regularizer=None,
                            activity_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=None)(input_layer)
    if bn:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(activation)(nn)
    if dropout:
        nn = keras.layers.Dropout(dropout)(nn)

    return nn


def conv_layer(inputs, num_filters, kernel_size, padding='same', activation='relu', dropout=0.2, l2=None, bn=True, kernel_initializer=None):
    if l2:
        l2 = keras.regularizers.l2(l2)
    else:
        l2 = None

    nn = keras.layers.Conv1D(filters=num_filters,
                             kernel_size=kernel_size,
                             strides=1,
                             activation=None,
                             use_bias=False,
                             padding=padding,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=l2,
                             bias_regularizer=None,
                             activity_regularizer=None,
                             kernel_constraint=None,
                             bias_constraint=None,
                             )(inputs)
    if bn:
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(activation)(nn)
    if dropout:
        nn = keras.layers.Dropout(dropout)(nn)
    return nn



def residual_block(input_layer, filter_size, activation='relu', l2=None):
    if l2:
        l2 = keras.regularizers.l2(l2)
    else:
        l2 = None

    num_filters = input_layer.shape.as_list()[-1]

    nn = keras.layers.Conv1D(filters=num_filters,
                             kernel_size=filter_size,
                             strides=1,
                             activation='relu',
                             use_bias=False,
                             padding='same',
                             dilation_rate=1,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2
                             )(input_layer)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.Conv1D(filters=num_filters,
                             kernel_size=filter_size,
                             strides=1,
                             activation='relu',
                             use_bias=False,
                             padding='same',
                             dilation_rate=1,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2
                             )(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.add([input_layer, nn])
    return keras.layers.Activation(activation)(nn)


#this is set up for 1d convolutions where examples
#have dimensions (len, num_channels)
#the channel axis is the axis for one-hot encoding.
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


#shuffled background
from collections import defaultdict
from random import shuffle


# compile the dinucleotide edges
def prepare_edges(s):
    edges = defaultdict(list)
    for i in range(len(s) - 1):
        edges[tuple(s[i])].append(s[i + 1])
    return edges


def shuffle_edges(edges):
    # for each character, remove the last edge, shuffle, add edge back
    for char in edges:
        last_edge = edges[char][-1]
        edges[char] = edges[char][:-1]
        the_list = edges[char]
        shuffle(the_list)
        edges[char].append(last_edge)
    return edges


def traverse_edges(s, edges):
    generated = [s[0]]
    edges_queue_pointers = defaultdict(lambda: 0)
    for i in range(len(s) - 1):
        last_char = generated[-1]
        generated.append(edges[tuple(last_char)][edges_queue_pointers[tuple(last_char)]])
        edges_queue_pointers[tuple(last_char)] += 1
    if isinstance(generated[0], str):
        return "".join(generated)
    else:
        import numpy as np
        return np.asarray(generated)


def dinuc_shuffle(s):
    if isinstance(s, str):
        s = s.upper()
    return traverse_edges(s, shuffle_edges(prepare_edges(s)))


def onehot_dinuc_shuffle(s):
    s = np.squeeze(s)
    argmax_vals = "".join([str(x) for x in np.argmax(s, axis=-1)])
    shuffled_argmax_vals = [int(x) for x in traverse_edges(argmax_vals,
                            shuffle_edges(prepare_edges(argmax_vals)))]
    to_return = np.zeros_like(s)
    to_return[list(range(len(s))), shuffled_argmax_vals] = 1
    return to_return

shuffle_several_times = lambda s: np.array([onehot_dinuc_shuffle(s) for i in range(10)])

#ohd_pos = np.array([one_hot_encode_along_channel_axis(seq) for seq in pos.sequence])
#ohd_neg = np.array([one_hot_encode_along_channel_axis(seq) for seq in neg.sequence])

#total = np.concatenate((ohd_pos, ohd_neg))
#labels = np.expand_dims(np.array([1]*len(pos.sequence) + [0]*len(neg.sequence)), axis = 1)
#neg_training_sequences_ohe = np.array([one_hot_encode_along_channel_axis(seq) for seq in neg_training_sequences])
#neg_testing_sequences_ohe = np.array([one_hot_encode_along_channel_axis(seq) for seq in neg_testing_sequences])
#pos_training_sequences_ohe = np.array([one_hot_encode_along_channel_axis(seq) for seq in pos_training_sequences])
#pos_testing_sequences_ohe = np.array([one_hot_encode_along_channel_axis(seq) for seq in pos_testing_sequences])
#X_train_new = np.concatenate((neg_training_sequences_ohe, pos_training_sequences_ohe), axis = 0)
#X_test_new = np.concatenate((neg_testing_sequences_ohe, pos_testing_sequences_ohe), axis = 0)
#y_train_new = np.concatenate(([0]*len(neg_training_sequences), [1]*len(pos_training_sequences)))
#y_test_new = np.concatenate(([0]*len(neg_testing_sequences), [1]*len(pos_testing_sequences)))
import pandas as pd
neg_training_df = pd.DataFrame(neg_training_sequences, columns=['sequences'])
pos_training_df = pd.DataFrame(pos_training_sequences, columns=['sequences'])
pos_testing_df = pd.DataFrame(pos_testing_sequences, columns=['sequences'])
neg_testing_df = pd.DataFrame(neg_testing_sequences, columns=['sequences'])
neg_training_df["label"] = 0
neg_testing_df["label"] = 0
pos_testing_df["label"] = 1
pos_training_df["label"] = 1
training = pd.concat([pos_training_df, neg_training_df])
testing = pd.concat([pos_testing_df, neg_testing_df])
from sklearn.model_selection import train_test_split
X_train, _, y_train, _= train_test_split(training['sequences'], training['label'], test_size=0.01, random_state=42)
_, X_test, _, y_test = train_test_split(testing['sequences'], testing['label'], test_size=1-0.01, random_state=42)
X_train_met1_ohe = np.array([one_hot_encode_along_channel_axis(seq) for seq in X_train])
X_test_met1_ohe = np.array([one_hot_encode_along_channel_axis(seq) for seq in X_test])
y_train_new = y_train
y_test_new = y_test
X_test_new = X_test_met1_ohe
X_train_new = X_test_met1_ohe

#basset
from tensorflow.keras import Input, Model

def model(activation='relu'):

    # input layer
    inputs = Input(shape=X_train_new.shape[1::])

    activation = activation_fn(activation)


    # layer 1
    nn = conv_layer(inputs,
                           num_filters=70,
                           kernel_size=19,   # 192
                           padding='same',
                           activation=activation,
                           dropout=0.2,
                           l2=1e-6,
                           bn=True)
    nn = MaxPool1D(pool_size=3)(nn)

    # layer 2
    nn = conv_layer(nn,
                           num_filters=50,
                           kernel_size=11,  # 56
                           padding='valid',
                           activation='relu',
                           dropout=0.2,
                           l2=1e-6,
                           bn=True)
    nn = MaxPool1D(pool_size=4)(nn)

    # layer 3
    nn = conv_layer(nn,
                           num_filters=50,
                           kernel_size=7,  # 56
                           padding='valid',
                           activation='relu',
                           dropout=0.2,
                           l2=1e-6,
                           bn=True)
    nn = MaxPool1D(pool_size=4)(nn)

    # layer 4
    nn = Flatten()(nn)
    nn = dense_layer(nn, num_units=1000, activation='relu', dropout=0.5, l2=1e-6, bn=True)

    # layer 5
    nn = dense_layer(nn, num_units=1000, activation='relu', dropout=0.5, l2=1e-6, bn=True)

    # Output layer
    logits = Dense(1, activation='linear', use_bias=True)(nn)
    outputs = Activation('sigmoid')(logits)

    model = Model(inputs=inputs, outputs=outputs)

    return model


ex = model()
ex.compile(optimizer='adam',
                               loss='binary_crossentropy',
                               metrics=[tf.keras.metrics.BinaryCrossentropy(),
                                        tf.keras.metrics.AUC(curve='ROC'),
                                        tf.keras.metrics.AUC(curve='PR')])


hist = ex.fit(x=X_train_met1_ohe,y=y_train,batch_size=16,epochs=50,  validation_split = 0.1)
np.set_printoptions(threshold=sys.maxsize)
from sklearn.metrics import roc_curve, auc, plot_roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
# skplt.metrics.plot_roc_curve(y_true, y_probas)
# skplt.metrics.plot_roc_curve(y_test_new, model1.predict(X_test_new))
auc = roc_auc_score(y_test, ex.predict(X_test_met1_ohe))
#y_train_new = y_train
#y_test_new = y_test
#X_test_new = X_test_met1_ohe
#X_train_new = X_test_met1_ohe
new = pd.concat([pd.DataFrame(ex.predict(X_test_met1_ohe)), pd.DataFrame(ex.predict(X_train_met1_ohe))], axis=1) 
new.columns = ["Test_scores", "Train_scores"]
new.to_csv("TF_model_bassett_baseline_CPA_scores_dist", sep = "\t", index = False)
auprc = average_precision_score(y_test_new, ex.predict(X_test_new))
fpr, tpr, _ = roc_curve(y_test_new, ex.predict(X_test_new))
precision, recall, thresholds = precision_recall_curve(y_test_new, ex.predict(X_test_new))
fscore = (2 * precision * recall) / (precision + recall)
ix = np.argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

x=pd.DataFrame({k:[np.array(v)] for k, v in hist.history.items()})
x['auc_test'] = auc
#x['val_auc'] = auprc
x['auprc_test'] = auprc
#x['val_auc'] = val_auc_1
#x['loss'] = loss
#x['val_loss'] = val_loss
x['fpr_test'] = [fpr]
x['tpr_test'] = [tpr]
x['precision_test'] = [precision]
x['recall_test'] = [recall]
x = x.T
import datetime as dt
timeout = dt.datetime.now().strftime('%Y%m%d%H%M%S')
#outputfilename = 'bassett_{}.csv'.format( timeout)
#outputfilename = 'bassett_u1_{}.csv'.format( timeout)
#ex.save('model_bassett_tf_{}'.format( timeout ))
ex.save(args.outdir+"/" +args.outname+'_bassett_model_{}.tf'.format( timeout ))
x.to_csv(args.outdir+"/" +args.outname+'_bassett_performance_{}.csv'.format( timeout ),  sep= '\t')
#np.savetxt(outputfilename, x, delimiter=',')


perf = ex.evaluate(X_test_new, y_test_new)
print(perf)

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 250
mpl.rcParams['figure.figsize'] = [8, 4]

plt.plot(hist.history['auc'])
plt.plot(hist.history['val_auc'])
plt.title('Model Training AUROC', weight = 'bold')
plt.ylabel('AUROC', weight = 'bold')
plt.xlabel('Epoch', weight = 'bold')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()
#plt.savefig('model_training_auroc_bassett_{}.png'.format( timeout ), dpi = 300)
plt.savefig(args.outdir+"/" +args.outname+'_model_training_auroc_bassett_{}.png'.format( timeout ), dpi = 300)
plt.clf()

plt.plot(hist.history['auc_1'])
plt.plot(hist.history['val_auc_1'])
plt.title('Model Training AUPRC', weight = 'bold')
plt.ylabel('AUPRC', weight = 'bold')
plt.xlabel('Epoch', weight = 'bold')
# plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()
#plt.savefig('model_training_auprc_bassett_{}.png'.format( timeout ), dpi = 300)
plt.savefig(args.outdir+"/" +args.outname+'_model_training_auprc_bassett_{}.png'.format( timeout ), dpi = 300)
plt.clf()


# mpl.rcParams['figure.dpi']= 100
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss', weight = 'bold')
plt.ylabel('Loss', weight = 'bold')
plt.xlabel('Epoch', weight = 'bold')
# plt.legend(['Training', 'Validation'], loc='lower left')
plt.show()
#plt.savefig('model_training_loss_bassett_{}.png'.format( timeout ), dpi = 300)
plt.savefig(args.outdir+"/" +args.outname+'_model_training_loss_bassett_{}.png'.format( timeout ), dpi = 300)
plt.clf()

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 250
mpl.rcParams['figure.figsize'] = [8, 4]
from sklearn.metrics import roc_curve, auc, plot_roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
# skplt.metrics.plot_roc_curve(y_true, y_probas)
# skplt.metrics.plot_roc_curve(y_test_new, model1.predict(X_test_new))
auc = roc_auc_score(y_test_new, ex.predict(X_test_new))
auprc = average_precision_score(y_test_new, ex.predict(X_test_new))
fpr, tpr, _ = roc_curve(y_test_new, ex.predict(X_test_new))
precision, recall, _ = precision_recall_curve(y_test_new, ex.predict(X_test_new))
fig, axs = plt.subplots(1,2)
axs[0].plot(fpr,tpr,label="Test set, AUC="+str(round(auc, 3)))
axs[1].plot(recall, precision,label="Test set, AUPRC="+str(round(auprc, 3)))
# axs[0].title('Model Performance on Held-Out Test Data')
axs[0].set_title("AUROC of Model on Test Set", weight = 'bold')
axs[1].set_title("AUPRC of Model on Test Set", weight = 'bold')
axs[0].set_ylabel('True Positive Rate', weight = "bold")
axs[1].set_ylabel('Precision', weight = "bold")
axs[0].set_xlabel('False Positive Rate', weight = "bold")
axs[1].set_xlabel('Recall', weight = "bold")
axs[0].legend(loc="lower center")
axs[1].legend(loc="lower center")
# plt.legend(['AUROC', 'AUPRC'], loc='lower left')
plt.show()
#plt.savefig('model_testing_bassett_{}.png'.format( timeout ), dpi = 300)
plt.savefig(args.outdir+"/" + args.outname+'_model_testing_auroc_auprc_bassett_{}.png'.format( timeout ), dpi = 300)
plt.clf()
