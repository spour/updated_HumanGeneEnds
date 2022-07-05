import joblib
import numpy as np
import argparse
import shap
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser.add_argument("-rf", "--rf", type=str, help="random forest model to explain", required=True)
#parser.add_argument("-s", "--scores", type=str, help="infile of directory of pq matrices that you're scoring", required=True)
#parser.add_argument("-out", "--outfile", type = str, help="parquet out file", required=True)
#parser.add_argument("-t", "--thresh", type=float, help="threshold for accuracy prediction", default=0)
args = parser.parse_args()
import pandas as pd
names = ["DSE_1", "DSE_2", "DSE_3", "DSE_4", "NUDT", "PAS_Siepel", "PAS_Hu", "UGUA_Hu", "PAS_hexamers_PWM","PolyU_PWM", "CA_UA", "DSEs", "PolyU_kmers", "PolyU_kmer", "UGUA_kmers"]

A= joblib.load(args.rf)

print("loaded")
A=A.feature_importances_
import seaborn as sns
import datetime
now = datetime.datetime.now()
import os
model_name =os.path.basename(args.rf)
model_name = os.path.splitext(model_name)[0]
np.savetxt("RF_coefficients_{}_{}.csv".format(model_name, now), A, delimiter=",")
import matplotlib.pyplot as plt
A = np.reshape(A, (15, -1))
from scipy.stats import zscore
print("heatmapping")
#WRONG: ax = sns.heatmap(np.transpose(np.reshape(A, (-1, 15))), yticklabels=names,linewidth=0.5, cmap = "flare")
ax = sns.heatmap(zscore(A, axis = 1), linewidth=0.5, yticklabels = names, cmap = "flare")
plt.tight_layout()
#plt.savefig("heatmap_weights_unbalanced.png", dpi = 500)
#score_name = os.path.basename(args.scores)
plt.savefig("RF_coefficients_heatmap_{}_{}.png".format(model_name, now), dpi = 500)
plt.clf()
#import shap
#import glob
#data_dir = args.scores
#full_df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in glob.glob(args.scores+'/*.pq'))
#POSITIVE_DF = full_df[full_df.index.str.contains("positive")]
#NEGATIVE_DF = full_df[full_df.index.str.contains("negative")]
#negative_class = np.array([0] * NEGATIVE_DF.shape[0])
#positive_class = np.array([1] * POSITIVE_DF.shape[0])
#MODEL
#data = np.concatenate((np.array(NEGATIVE_DF), np.array(POSITIVE_DF)))
#classes = np.concatenate([negative_class, positive_class])
#X_train, X_test, y_train, y_test = train_test_split(    data, classes, test_size=0.2, random_state=42, stratify=classes)


#explainer = shap.TreeExplainer(joblib.load(args.rf))
#shap_values = np.array(explainer.shap_values(X_test))

#shap_values_ = shap_values.transpose((1,0,2))
#np.allclose(
#    clf_unablanced.predict_proba(X_test),
#    shap_values_.sum(2) + explainer.expected_value
#)

#shap_values = explainer.shap_values(X_test)
# style.use('seaborn-dark')
#import seaborn as sns
#sns.set_style("whitegrid")
#shap.summary_plot(shap_values[1], X_test, plot_type="dot", feature_names =full_df.columns, max_display=25, show=False)
#plt.tight_layout()
#plt.savefig("RF_coefficients_SHAP_{}_{}_{}.png".format(model_name, score_name,now), dpi = 300)
#print("done")



