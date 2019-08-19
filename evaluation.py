import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


EVAL_DICT = {"accuracy": accuracy_score, "f1": f1_score, \
             "recall": recall_score, "precision": precision_score,\
             "AUC_ROC": roc_auc_score}
eval_method_list = ["accuracy", "f1", "recall", "precision", "AUC_ROC"]

percentile_list = [0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]


def get_threshold_list(y_predp, percentile_list):
    '''
    This function is used to provide the threshold list using the precentile
    of y_predp
    Inputs:
        y_predp: predicted probability of the classifier
        precentile_list: list of target percentile
    Return: list of thresholds
    '''

    thres_list = pd.Series(y_predp[:,1]).sort_values(ascending=True).quantile(percentile_list)

    return list(thres_list)


def predp_distribution(y_predp):
    '''
    This function is used to depict the distribution of the
    predicted probability of the classifier
    Input: predicted probability of the classifier
    '''
    plt.xlabel("predicted probability")
    plt.ylabel("frequency in %")
    plt.hist(y_predp, bins=30, density=True)

    return "mean = {}, count = {}".format(y_predp.mean(), y_predp.shape[0])


def prob_to_class(threshold, y_predp):
    '''
    This function is used to transfer the predicted probability to 
    categorical calssification with a given threshold
    Inputs:
        threshold: threshold for the cutoff
        y_predp: predicted probability of the classifier
    Return: predicted classifier
    '''

    # using >= instead of > to accommodate the probability that the
    # scores might be the same and all of them should be included
    y_pos_loc = pd.Series(y_predp[:,1] >= threshold)
    y_pred = pd.Series([0] * len(y_predp))
    y_pred.loc[y_pos_loc] = 1
    
    return y_pred

def calculate_scores_r(y_test, y_predp, threshold, sample_weight=None):
    '''
    This function is used to create four critical values in the confusion metrix
    Inputs:
        y_test: actual classifier
        y_predp: the predicted probability fo the classifier from the training subset
        threshold_list: list of threshold to search
    Return: True Negative, False Positive, False Negative, True Positive
    '''
    y_pred = prob_to_class(threshold, y_predp)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, sample_weight=sample_weight).ravel()

    return tn, fp, fn, tp


def calculate_scores(y_test, y_predp, threshold, sample_weight=None):
    '''
    This function is used to claculate the scores with specific threshold 
    of every evaluation methods in the given dictionary
    Inputs:
        y_test: actual classifier
        y_predp: the predicted probability fo the classifier from the training subset
        threshold_list: list of threshold to search
    Returns: dictionary of scores
    '''
    score_dict = {}
    for name, eval_func in EVAL_DICT.items():
        if name =="AUC_ROC":
            score = eval_func(y_test, y_predp[:,1], \
                       sample_weight=sample_weight)
        else:
            y_pred = prob_to_class(threshold, y_predp)
            score = eval_func(y_test, y_pred, sample_weight=sample_weight)
        score_dict[name] = score

    return score_dict


def depict_ROC(y_test, y_predp):
    '''
    This function is used to depict ROC curve for the given classifications
    Inputs:
        y_test: actual classifier
        y_predp: the predicted probability fo the classifier from the training subset
    '''
    fpr, tpr, thresholds = roc_curve(y_test, y_predp[:,1])
    plt.xlabel("false positive %")
    plt.ylabel("true positive %")
    plt.plot(fpr, tpr)


####################################################################
# Source: https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
# Author: Rayid Ghani
# Date: 2017
####################################################################
# In this function y_prob is equivalent to y_predp in the above functions
def plot_precision_recall_n(y_true, y_prob, model_name):
    y_score = y_prob[:,1]
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population labelled 1 in prediction')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()


def single_threshold_evaluation(y_test, y_predp, model_name, threshold, eval_method_list, pr_curve=False, roc_curve=False, sample_weight=None):
    '''
    This function is used to previde a single evaluation anlysis with a given threshold
    Inputs:
        y_test: actual classifier
        y_predp: the predicted probability fo the classifier from the training subset
        eval_method: method used in the evaluation
        threshold: threshold to search
        model_name: model name used in plotting
    Returns: output dictionary of the evaluation
    '''
    output_dict = {"threshold": threshold}
    y_pred = prob_to_class(threshold, y_predp)
    for eval_method in eval_method_list:
        if eval_method == "AUC_ROC":
            cur_eval = EVAL_DICT[eval_method](y_test, y_predp[:,1], \
                                    sample_weight=sample_weight)
        else:
            cur_eval = EVAL_DICT[eval_method](y_test, y_pred, \
                                    sample_weight=sample_weight)

        output_dict[eval_method] = [cur_eval]

    if pr_curve:
        plot_precision_recall_n(y_test, y_predp, model_name)

    if roc_curve:
        depict_ROC(y_test, y_predp)

    return output_dict


def best_threshold(y_test, y_predp, eval_method, threshold_list, sample_weight=None):
    '''
    This function is used to find the best_threshold of given evaluation method
    Inputs:
        y_test: actual classifier
        y_predp: the predicted probability fo the classifier from the training subset
        eval_method: method used in the evaluation
        threshold_list: list of threshold to search
    Return: best thresthold, best score of the evaluation corresponding to it
    '''
    best_eval = 0
    threshold_list = get_threshold_list(y_predp, threshold_list)

    for threshold in threshold_list:
        y_pred = prob_to_class(threshold, y_predp)
        if eval_method == "AUC_ROC":
            cur_eval = EVAL_DICT[eval_method](y_test, y_predp[:,1], \
                                   sample_weight=sample_weight)
        else:
            cur_eval = EVAL_DICT[eval_method](y_test, y_pred, \
                                 sample_weight=sample_weight)

        print("operating best_threshold, eval_method: {}, threshold: {}, score: {}".format(eval_method, threshold, cur_eval))

        if cur_eval > best_eval:
            best_thres = threshold
            best_eval = cur_eval
    
    return best_thres, best_eval