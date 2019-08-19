import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
import feature_generation as fg
import imputation as imp
import train_test_split as split
import evaluation as eva

percentile_list = [0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
eval_method_list = ["accuracy", "f1", "recall", "precision", "AUC_ROC"]

#############################################################################
# Source: https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
# Author: Rayid Ghani
# Date: 2017
#############################################################################
clfs = {'decision_tree': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier(n_jobs=2),
        'logistics': LogisticRegression(penalty='l1', C=1e5),
        "linearSVC": LinearSVC(),
        'SVM': SVC(kernel='linear', probability=True, max_iter=100),
        'NB': GaussianNB(),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'gradientboost': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'adaboost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'bagging': BaggingClassifier(base_estimator=LogisticRegression(penalty='l1', C=1e5), n_jobs=2),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

large = { 
    'decision_tree': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'random_forest':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'logistics': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]}
           }
#'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
#'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
#'gradientboost': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
#'adaboost': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
#"bagging": {'n_estimators': [1,10,100,1000,10000]}

small = {
    'decision_tree': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'random_forest':{'n_estimators': [1, 10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'logistics': {'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'ET': { 'n_estimators': [1, 10, 100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs':[-1]},
    'gradientboost': {'n_estimators': [1,10], 'learning_rate' : [0.1,0.5],'subsample' : [0.5,1.0], 'max_depth': [5,50]},
    'adaboost': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100]},
    "bagging": {'n_estimators': [1, 10,100]},
           }

#'linearSVC' :{'C' :[0.001,0.01,0.1,1,10], 'penalty': ['l1', 'l2']},
#'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
#'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}

    
simple = { 
    'decision_tree': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'random_forest':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'logistics': { 'penalty': ['l1'], 'C': [0.01]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'NB' : {},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10], 'n_jobs': [-1]},
    'gradientboost': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'adaboost': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'bagging': {'n_estimators': [10]}, 
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }

# 'linearSVC' :{'C' :[0.01],'penalty':['l2']},
#'SVM' :{'C' :[0.01],'kernel':['linear']},

test_simple = {
    'decision_tree': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'random_forest':{'n_estimators': [1], 'max_depth': [1, 10, 100], 'max_features': ['sqrt'],'min_samples_split': [10]}
    }


def data_preprocessors(df, selected_features, classifier):
    '''
    This function is designed as a integrater of data preprocessing, which includes imputation,
    feature generation, train_test_split.
    Inputs: 
        df: dataframe
        selected_features: features selected
        classifiers: defined outcome of the test
    Returns:
    '''
    pass

def classifier_developer(method, clfs, parameter_dict):
    '''
    This function is used to generate possible combination of hyperparameters of
    given classifiers
    Inputs:
        method: specific classifiers
        clfs: dictionary of classifiers
        parameter_dict: parameters dictionary for the classifiers
    Returns: list of all possible combination of parameters
    '''
    parameters = parameter_dict[method]
    para_list = ParameterGrid(parameters)

    return para_list


def performance_developer(clfs, parameter_dict, X_train, y_train, X_test, y_test, percentile_list, eval_method_list, \
                          pr_curve=False, roc_curve=False, sample_weight=None):
    '''
    This function is used to generate performance matrix
    Inputs:
        clfs: dictionary of classifiers
        parameter_dict: parameters dictionary for the classifiers
        X_train, y_train, X_test, y_test: Traing and testing dataframes
        percentile_list: list of percentile used as cutoff points for the predicted probability
        eval_method_list: list of methods used in evaluation
        pr_curve, roc_curve: boolean used to plot curve
    Return: performance matrix
    '''
    performance_matrix = pd.DataFrame(columns= ["method"] + eval_method_list + ["threshold", "parameters"])
    for method, _ in clfs.items():
        if method in parameter_dict:
            print("operation of {} method begins".format(method))
            para_list = classifier_developer(method, clfs, parameter_dict)
            for para in para_list:
                clf = clfs[method].set_params(**para)
                model_name = method + " with parameters : " + str(clf.get_params())
                model = clf.fit(X_train, y_train)
                y_predp = model.predict_proba(X_test)
                threshold_list = eva.get_threshold_list(y_predp, percentile_list)
                pr_curve = True
                for i, threshold in enumerate(threshold_list):
                    output_dict = eva.single_threshold_evaluation(y_test, y_predp, model_name, threshold,\
                                                                  eval_method_list, pr_curve, roc_curve, \
                                                                  sample_weight)
                    output_dict["method"] = method
                    output_dict["threshold"] = percentile_list[i]
                    output_dict["parameters"] = model_name
                    sub_performance = pd.DataFrame(output_dict)
                    performance_matrix = pd.concat([performance_matrix, sub_performance], join="inner")
                    pr_curve = False

    return performance_matrix


def cross_validate_performance(clfs, parameter_dict, data_dict, percentile_list, eval_method_list, \
                               pr_curve=False, roc_curve=False, sample_weight=None):
    '''
    This function is used to perform cross validation of dataframes
    Inputs:
        clfs: dictionary of classifiers
        parameter_dict: parameters dictionary for the classifiers
        X_train, y_train, X_test, y_test: Traing and testing dataframes
        percentile_list: list of percentile used as cutoff points for the predicted probability
        eval_method_list: list of methods used in evaluation
        pr_curve, roc_curve: boolean used to plot curve
    Return: performance matrix
    '''
    cross_performance = pd.DataFrame(columns= ["method"] + eval_method_list + ["threshold", "parameters", "cross_validate_index"])
    for df_name, dfs in data_dict.items():
        print("temporal model: {} is operated".format(df_name))
        X_train, y_train, X_test, y_test = dfs[0], dfs[1], dfs[2], dfs[3]
        performance_matrix = performance_developer(clfs, parameter_dict, X_train, y_train, X_test, y_test, \
                                                   percentile_list, eval_method_list, \
                                                   pr_curve, roc_curve, sample_weight)
        performance_matrix["cross_validate_index"] = df_name
        cross_performance = pd.concat([cross_performance, performance_matrix], join='inner')

    return cross_performance