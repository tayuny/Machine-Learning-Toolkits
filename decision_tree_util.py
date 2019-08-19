import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import graphviz

def map_feature_importances(model, features):
    '''
    This function is used to provide feature importances of the
    festures used in the decision tree
    Inputs:
        tree: decision tree object
        features: features used in the decision tree
    Returns: feature importances dictionary
    '''
    feature_dict = {}
    importances = model.feature_importances_
    for i, val in enumerate(features):
        feature_dict[val] = importances[i]

    return feature_dict


#############################################################
# Source: Plotting Decision Trees via Graphviz, scikit-learn
# Author: scikit-learn developers
# Date: 2007 - 2018
#############################################################
def depict_decision_tree(decision_tree, features, classifier, output_filename):
    '''
    This function is used to generate the decision tree graph which
    contains the tree node and related information. The output will be
    saved as pdf file
    Inputs:
        dataframe
        features: list of feature names used in decision tree
        classifier: name of the classifier used in decision tree
        output_filename: the name of the pdf file
    '''
    dot_data = tree.export_graphviz(decision_tree, out_file="tree.dot", \
                                                   feature_names=features,\
                                                   class_names=classifier,\
                                                   filled=True, rounded=True,\
                                                   special_characters=True)
    file = open('tree.dot', 'r')
    text = file.read()
    graph = graphviz.Source(text)  
    graph.render(output_filename) 