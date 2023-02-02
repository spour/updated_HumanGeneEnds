import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import joblib


def read_csv_files(positive_training_csv, negative_training_csv, positive_testing_csv, negative_testing_csv):
    """
    Read the csv files of positive and negative training and testing examples and return them as numpy arrays.

    Parameters:
        positive_training_csv (str): path to the positive training examples file.
        negative_training_csv (str): path to the negative training examples file.
        positive_testing_csv (str): path to the positive testing examples file.
        negative_testing_csv (str): path to the negative testing examples file.

    Returns:
        Tuple of numpy arrays: positive training examples, negative training examples, positive testing examples, negative testing examples.
    """
    pos_train = pd.read_csv(positive_training_csv, sep = "\t", header = 0, index_col=0).to_numpy()
    neg_train = pd.read_csv(negative_training_csv, sep = "\t", header = 0, index_col=0).to_numpy()
    pos_test = pd.read_csv(positive_testing_csv, sep = "\t", header = 0, index_col=0).to_numpy()
    neg_test = pd.read_csv(negative_testing_csv, sep = "\t", header = 0, index_col=0).to_numpy()

    return pos_train, neg_train, pos_test, neg_test


def balance_classes(pos_data, neg_data, num_to_use):
    """
    Balance the number of positive and negative examples by subsampling negative examples.

    Parameters:
        pos_data (numpy array): positive examples
        neg_data (numpy array): negative examples
        num_to_use (int): number of negative examples to subsample

    Returns:
        Tuple of numpy arrays: balanced positive and negative examples, and corresponding positive and negative example labels
    """
    np.random.shuffle(neg_data)
    subsample_neg_data = neg_data[:num_to_use,:]
    subsample_neg_data_class = np.array([0] * subsample_neg_data.shape[0])
    balanced_classes = np.concatenate([subsample_neg_data_class, np.array([1] * pos_data.shape[0]) ])
    balanced_data = np.concatenate((subsample_neg_data, pos_data))

    return balanced_data, balanced_classes


def create_class_labels(data, class_label):
    """
    Create class labels for a given data array
    Args:
        data (np.array): data to be assigned labels to
        class_label (int): class label to be assigned to the data
    Returns:
        class_labels (np.array): class labels for the data
    """
    class_labels = np.array([class_label] * data.shape[0])
    return class_labels



def read_in(pos_train, pos_test, neg_train, neg_test):
	"""
	Read in positive and negative training and test data, create class labels, and build X/y train/test split.
	Parameters:
	pos_train (np.ndarray): Positive training data.
	pos_test (np.ndarray): Positive test data.
	neg_train (np.ndarray): Negative training data.
	neg_test (np.ndarray): Negative test data.

	Returns:
	X_train (np.ndarray): Input training data with shape (n_samples, n_features).
	y_train (np.ndarray): Input training target data with shape (n_samples,).
	X_test (np.ndarray): Input test data with shape (n_samples, n_features).
	y_test (np.ndarray): Input test target data with shape (n_samples,).
	"""
	pos_train_class = create_class_labels(pos_train, 1)
	pos_test_class = create_class_labels(pos_test, 1)
	neg_train_class = create_class_labels(neg_train, 0)
	neg_test_class = create_class_labels(neg_test, 0)

	#buid into X/y train/test split
	X_train = np.concatenate((neg_train, pos_train))
	y_train = np.concatenate((neg_train_class, pos_train_class))
	X_test = np.concatenate((neg_test, pos_test))
	y_test = np.concatenate((neg_test_class, pos_test_class))

	return X_train, y_train, X_test, y_test


#MODEL

def train_logistic_regression(X_train, y_train, out_path):
	"""
	This function trains a Logistic Regression classifier with unbalanced class weights on the given training data and saves the model.
	For cryptic model.
	Parameters:
	X_train (np.ndarray): Input training data with shape (n_samples, n_features).
	y_train (np.ndarray): Input training target data with shape (n_samples,).
	out_path (str): Path to save the trained model.

	Returns:
	clf_unablanced (LogisticRegression): Trained Logistic Regression model.
	"""
	clf_unablanced  =LogisticRegression(n_jobs=-1, C=0.0018, penalty = 'l1', tol=0.01, solver = 'saga', class_weight='balanced')
	clf_unablanced.fit(X_train, y_train)
	joblib.dump(clf_unablanced, out_path, compress=3)

	return clf_unablanced

def train_baseline_random_forest(X_train, y_train, out_path):
	"""
	This function trains a Logistic Regression classifier with unbalanced class weights on the given training data and saves the model.
	For baseline model.
	Parameters:
	X_train (np.ndarray): Input training data with shape (n_samples, n_features).
	y_train (np.ndarray): Input training target data with shape (n_samples,).
	out_path (str): Path to save the trained model.

	Returns:
	clf_unablanced (LogisticRegression): Trained Logistic Regression model.
	"""
	clf_unablanced  = RandomForestClassifier(n_jobs=-1, class_weight="balanced", n_estimators=30000, min_samples_split=5)
	clf_unablanced.fit(X_train, y_train)
	joblib.dump(clf_unablanced, out_path, compress=3)

	return clf_unablanced


def roc(y_test, X_test, clf, axs):
    """
    Plot ROC curve

    Parameters
    ----------
    y_test : array-like, shape (n_samples,)
        Ground truth (correct) target values.
    preds : array-like, shape (n_samples,)
        Estimated target values.
    axs : array-like, shape (2,)
        Array of matplotlib axis objects to plot on.
    
    Returns
    -------
    fpr, tpr
    """
    # AUROC
    preds = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    axs.set_title('Receiver Operating Characteristic')
    axs.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    axs.legend(loc='lower right')
    axs.plot([0, 1], [0, 1], 'r--')
    axs.set_xlim([0, 1])
    axs.set_ylim([0, 1])
    axs.set_ylabel('True Positive Rate')
    axs.set_xlabel('False Positive Rate')
    return fpr, tpr, roc_auc


def prc(y_test, preds, axs):
    """
    Plot AUPRC

    Parameters
    ----------
    y_test : array-like, shape (n_samples,)
        Ground truth (correct) target values.
    preds : array-like, shape (n_samples,)
        Estimated target values.
    axs : array-like, shape (2,)
        Array of matplotlib axis objects to plot on.
    
    Returns
    -------
    precision, recall, aurpc
    """
    # AUPRC
    precision, recall, thresholds = precision_recall_curve(y_test, preds)
    aurpc = average_precision_score(y_test, preds)
    print("aurpc", aurpc)
    axs.set_title('AUPRC')
    axs.plot(recall, precision, 'b', label='AUPRC = %0.2f' % aurpc)
    axs.legend(loc='lower right')
    axs.plot([0, 1], [0, 1], 'r--')
    axs.set_xlim([0, 1])
    axs.set_ylim([0, 1])
    axs.set_ylabel('Precision')
    axs.set_xlabel('Recall')
    plt.tight_layout()
    return precision, recall, aurpc


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-ptr", "--positive_training_csv", type=str, help="csv of positive training file with feature matrices", required=True)
	parser.add_argument("-ntr", "--negative_training_csv", type=str, help="csv of negative training file  with feature matrices", required=True)
	parser.add_argument("-pte", "--positive_testing_csv", type=str, help="csv of positive testing file  with feature matrices", required=True)
	parser.add_argument("-nte", "--negative_testing_csv", type=str, help="csv of negative testing file with feature matrices", required=True)
	parser.add_argument("-out", "--outdir", type=str, help="outdir", required=True)
	parser.add_argument("-outp", "--outprefix", type=str, help="outprefix", required=True)
	args = parser.parse_args()

	fig, axs = plt.subplots(2)
	figure = plt.gcf() # get current figure
	figure.set_size_inches(10, 10)

	pos_train, neg_train, pos_test, neg_test = read_csv_files(args.positive_training_csv, args.negative_training_csv, args.positive_testing_csv, args.negative_testing_csv)
	X_train, y_train, X_test, y_test = read_in(pos_train, pos_test, neg_train, neg_test)
	model = train_logistic_regression(X_train, y_train, args.outdir+"/" +args.outp) #or train_baseline_random_forest for the baseline RF
	fpr, tpr, roc_auc = roc(y_test, X_test, model, axs[0])
	precision, recall, aurpc = prc(y_test, X_test, model, axs[1])










#SAVE
# plt.savefig(args.outdir+"/"+args.outprefix+'Unalanced_LR_30032022_chrsplit.png', bbox_inches='tight', dpi = 300)
# plt.clf()

# df_unbalanced = pd.concat([pd.DataFrame(fpr), pd.DataFrame(tpr), pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame([roc_auc]), pd.DataFrame([aurpc])], axis=1 )
# df_unbalanced.columns = ['unbalanced_fpr', 'unbalanced_tpr', 'unbalanced_prec', 'unbalanced_rec', 'unbalanced_roc', 'unbalanced_auprc']

# perf = pd.concat([df_balanced, df_unbalanced], axis = 1)
# perf.to_csv(args.outdir+"/"+args.outprefix+"LR_30032022_chrsplit.csv", sep = "\t")
# print("done")
