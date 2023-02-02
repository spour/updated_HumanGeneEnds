#read multiple pq files at once:
import pandas as pd
from pathlib import Path
import argparse
# import pyarrow as pa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument("-ptr", "--positive_training_csv", type=str, help="csv of positive training file", required=True)
parser.add_argument("-ntr", "--negative_training_csv", type=str, help="csv of negative training file", required=True)
parser.add_argument("-pte", "--positive_testing_csv", type=str, help="csv of positive testing file", required=True)
parser.add_argument("-nte", "--negative_testing_csv", type=str, help="csv of negative testing file", required=True)
parser.add_argument("-out", "--outdir", type=str, help="outdir", required=True)
parser.add_argument("-outp", "--outprefix", type=str, help="outprefix", required=True)
args = parser.parse_args()

# data_dir = Path(args.directory)
import glob
pos_train  = pd.read_csv(args.positive_training_csv, sep = "\t", header = 0, index_col=0).to_numpy()
neg_train = pd.read_csv(args.negative_training_csv, sep = "\t", header = 0, index_col=0).to_numpy()
pos_test = pd.read_csv(args.positive_testing_csv, sep = "\t", header = 0, index_col=0).to_numpy()
neg_test = pd.read_csv(args.negative_testing_csv, sep = "\t", header = 0, index_col=0).to_numpy()

pos_train_all  = pos_train
neg_train_all = neg_train
pos_test_all = pos_test
neg_test_all = neg_test
print("read in")

print("read in")

print("pos train shape: ", pos_train_all.shape, "neg train shape: ",  neg_train_all.shape,
      "pos test shape: ", pos_test_all.shape, "neg test shape:", neg_test_all.shape)
#import numpy as np
#negative_class = np.array([0] * NEGATIVE_DF.shape[0])
pos_train_all_class = np.array([1] * pos_train_all.shape[0])
neg_train_all_class = np.array([0] * neg_train_all.shape[0])

pos_test_all_class = np.array([1] * pos_test_all.shape[0])
neg_test_all_class = np.array([0] * neg_test_all.shape[0])
#MODEL
#data = np.concatenate((np.array(NEGATIVE_DF), np.array(POSITIVE_DF)))
#classes = np.concatenate([negative_class, positive_class])
#classes_data_joined=np.column_stack((data, classes))
#np.savetxt("classes_data_training_from_pq_joined", classes_data_joined, delimiter="\t")
#balanced
num_to_use_train = pos_train_all.shape[0]
#shuffle neg
np.random.shuffle(neg_train_all)
subsample_neg_train = neg_train_all[:num_to_use_train,:]
#create labels for subsampled neg
subsample_neg_train_class = np.array([0] * subsample_neg_train.shape[0])
train_balanced_classes = np.concatenate([subsample_neg_train_class, pos_train_all_class ])
train_balanced = np.concatenate((subsample_neg_train, pos_train_all))
# X_train, X_test, y_train, y_test = train_test_split(    data, classes, test_size=0.2, random_state=42, stratify=classes)

num_to_use_test = pos_test_all.shape[0]
#shuffle neg
np.random.shuffle(neg_test_all)
subsample_neg_test = neg_test_all[:num_to_use_test,:]
#create labels for subsampled neg
subsample_neg_test_class = np.array([0] * subsample_neg_test.shape[0])

print("17022022: pos_train_balanced_shape: ", pos_train_all_class.shape, "pos_test_balanced_shape: ",pos_test_all_class.shape, "neg_train_balanced_shape:",subsample_neg_train.shape, "neg_test_balanced.shape: ",subsample_neg_test.shape)
#clf_balanced  = RandomForestClassifier(n_jobs=-1,  n_estimators=30000, min_samples_split=5, class_weight="balanced") #n_estimators=30000, min_samples_split=5,
clf_balanced  =LogisticRegression(n_jobs=-1, C=0.0018, penalty = 'l1', tol=0.01, solver = 'saga', class_weight='balanced')
X_train = np.concatenate((subsample_neg_train, pos_train_all))
y_train = np.concatenate((subsample_neg_train_class, pos_train_all_class))

#X_test = np.concatenate((subsample_neg_test, pos_test_all))
#y_test = np.concatenate((subsample_neg_test_class, pos_test_all_class))
X_test = np.concatenate((neg_test_all, pos_test_all))
y_test = np.concatenate((neg_test_all_class, pos_test_all_class))
print("X_train_balanced_shape: ",X_train.shape, "X_test_balanced_shape: ",X_test.shape, "y_train_balanced_shape: ",y_train.shape, "y_test_balanced_shape: ",y_test.shape)
clf_balanced.fit(X_train, y_train)
import joblib
joblib.dump(clf_balanced, args.outdir+"/"+args.outprefix+"LR_30032022_balanced_chr_split.joblib", compress=3)
print("balanced dumped")
preds = clf_balanced.predict_proba(X_test)[:, 1]
fig, axs = plt.subplots(2)
figure = plt.gcf() # get current figure
figure.set_size_inches(10, 10)
#AUROC
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
#AUROC
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
print(fpr, tpr, roc_auc)
import matplotlib.pyplot as plt
axs[0].set_title('Receiver Operating Characteristic')
axs[0].plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
axs[0].legend(loc = 'lower right')
axs[0].plot([0, 1], [0, 1],'r--')
axs[0].set_xlim([0, 1])
axs[0].set_ylim([0, 1])
axs[0].set_ylabel('True Positive Rate')
axs[0].set_xlabel('False Positive Rate')
# plt.show()

#AUPRC
precision, recall, thresholds  = precision_recall_curve(y_test, preds)
aurpc = average_precision_score(y_test, preds)
axs[1].set_title('AUPRC')
axs[1].plot(recall, precision, 'b', label = 'AUPRC = %0.2f' % aurpc)
axs[1].legend(loc = 'lower right')
# axs[1].plot([positive_class.shape[0]/classes.shape[0],1], [positive_class.shape[0]/classes.shape[0],1],'r--')
axs[1].axhline(y=pos_test_all_class.shape[0]/(pos_test_all_class.shape[0] + subsample_neg_test_class.shape[0]), color='r', linestyle='--')
axs[1].set_xlim([0, 1])
axs[1].set_ylim([0, 1])
axs[1].set_ylabel('Precision')
axs[1].set_xlabel('Recall')
plt.tight_layout()
plt.savefig(args.outdir+"/"+args.outprefix+'Balanced_LR_30032022_chrsplit.png', bbox_inches='tight', dpi = 300)
plt.clf()

#sAVE metrics
df_balanced = pd.concat([pd.DataFrame(fpr), pd.DataFrame(tpr), pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame([roc_auc]), pd.DataFrame([aurpc])], axis=1 )
df_balanced.columns = ['balanced_fpr', 'balanced_tpr', 'balanced_prec', 'balanced_rec', 'balanced_roc', 'balanced_auprc']



#unbalanced
pos_train_all_class = np.array([1] * pos_train_all.shape[0])
neg_train_all_class = np.array([0] * neg_train_all.shape[0])

pos_test_all_class = np.array([1] * pos_test_all.shape[0])
neg_test_all_class = np.array([0] * neg_test_all.shape[0])
#MODEL
# data = np.concatenate((neg_train_all, pos_train_all))
# classes = np.concatenate([neg_train_all_class, pos_train_all_class])
#classes_data_joined=np.column_stack((data, classes))
#np.savetxt("classes_data_training_from_pq_joined", classes_data_joined, delimiter="\t")
# X_train, X_test, y_train, y_test = train_test_split(    data, classes, test_size=0.2, random_state=42, stratify=classes)
#clf_unablanced  = RandomForestClassifier(n_jobs=-1, class_weight="balanced") #n_estimators=30000, min_samples_split=5
clf_unablanced  =LogisticRegression(n_jobs=-1, C=0.0018, penalty = 'l1', tol=0.01, solver = 'saga', class_weight='balanced')

X_train = np.concatenate((neg_train_all, pos_train_all))
y_train = np.concatenate((neg_train_all_class, pos_train_all_class))

X_test = np.concatenate((neg_test_all, pos_test_all))
y_test = np.concatenate((neg_test_all_class, pos_test_all_class))
print("X_train_unbalanced_shape: ",X_train.shape, "X_test_unbalanced_shape: ",X_test.shape, "y_train_unbalanced_shape: ",y_train.shape, "y_test_unbalanced_shape: ",y_test.shape)

clf_unablanced.fit(X_train, y_train)
import joblib
joblib.dump(clf_unablanced, args.outdir+"/"+args.outprefix+ "LR_30032022_unbalanced_chr_split.joblib", compress=3)
print("unbalanced dumped")
preds = clf_unablanced.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, preds))
print(average_precision_score(y_test, preds))
fig, axs = plt.subplots(2)
figure = plt.gcf() # get current figure
figure.set_size_inches(10, 10)
#AUROC
#AUROC
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
print(roc_auc)
axs[0].set_title('Receiver Operating Characteristic')
axs[0].plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
axs[0].legend(loc = 'lower right')
axs[0].plot([0, 1], [0, 1],'r--')
axs[0].set_xlim([0, 1])
axs[0].set_ylim([0, 1])
axs[0].set_ylabel('True Positive Rate')
axs[0].set_xlabel('False Positive Rate')

#AUPRC
precision, recall, thresholds  = precision_recall_curve(y_test, preds)
aurpc = average_precision_score(y_test, preds)
print("aurpc", aurpc)
axs[1].set_title('AUPRC')
axs[1].plot(recall, precision, 'b', label = 'AUPRC = %0.2f' % aurpc)
axs[1].legend(loc = 'lower right')
axs[1].plot([0, 1], [0, 1],'r--')
axs[1].set_xlim([0, 1])
axs[1].set_ylim([0, 1])
axs[1].set_ylabel('Precision')
axs[1].set_xlabel('Recall')
plt.tight_layout()
#SAVE
plt.savefig(args.outdir+"/"+args.outprefix+'Unalanced_LR_30032022_chrsplit.png', bbox_inches='tight', dpi = 300)
plt.clf()

df_unbalanced = pd.concat([pd.DataFrame(fpr), pd.DataFrame(tpr), pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame([roc_auc]), pd.DataFrame([aurpc])], axis=1 )
df_unbalanced.columns = ['unbalanced_fpr', 'unbalanced_tpr', 'unbalanced_prec', 'unbalanced_rec', 'unbalanced_roc', 'unbalanced_auprc']

perf = pd.concat([df_balanced, df_unbalanced], axis = 1)
perf.to_csv(args.outdir+"/"+args.outprefix+"LR_30032022_chrsplit.csv", sep = "\t")
print("done")
