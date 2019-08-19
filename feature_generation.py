import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Generate features/predictors
def get_percentile_bin(train_df, column, percentile_list):
    '''
    This function is used to create bins for transforming continuous variables
    to categorical variables
    Inputs: 
        train_df: training dataframe
        column: column used in transformation
        percentile_list: percentile of the column used in bin creation
    Returns: list of quantile for the column, labels for the column
    '''
    quantile_list = df[column].quantile(percentile_list)
    quantile_label = np.arange(0, len(quantile_list) - 1)

    return list(quantile_list), quantile_label


def continue_to_category(train_df, test_df, column, interval_list, category_list):
    '''
    This function is used to transform continuous variables to categorical 
    variables
    Inputs:
        df: dataframe
        column: given column to be transformed
        interval_list: list of the interval of the categories
        category_list: list of names of corresponding categories
    Returns: updated dataframe
    '''
    train_df[column + "_category"] = pd.cut(train_df[column], bins=interval_list, \
                                     labels=category_list, \
                                     include_lowest=False, right=True)
    test_df[column + "_category"] = pd.cut(test_df[column], bins=interval_list, \
                                     labels=category_list, \
                                     include_lowest=False, right=True)
    return df


def category_to_binary(train_df, test_df, column, k_categories=False):
    '''
    This function is used to transform each of the category in the categorical
    variable to a binary variable
        Inputs: dataframe
        column: given categorical variable
    Return: updated dataframe
    '''
    if k_categories:
        cat_indices = train_df.groupby(column).size().sort_values(ascending=False).index
        cat_list = cat_indices[:k_categories]
    else:
        cat_list = list(train_df[column].unique())
    
    for category in cat_list:
        train_df[column + "==" + str(category)] = 0
        train_df.loc[train_df[column] == category, (column + "==" + str(category))] = 1
        train_df[column + "==" + "others"] = 0
    train_df = train_df.drop([column], axis=1)

    for category in cat_list:
        test_df[column + "==" + str(category)] = 0
        test_df[column + "==" + "others"] = 0

        for category_test in list(test_df[column].unique()):
            if category_test in cat_list:
                test_df.loc[test_df[column] == category_test, column + "==" + str(category_test)] = 1
            else:
                test_df.loc[test_df[column] == category_test, column + "==" + "others"] = 1

    test_df = test_df.drop([column], axis=1)

    return train_df, test_df


def normalization_column(train_df, test_df, column):
    '''
    This function is used to normalize the features to make them comparable
    Inputs:
        df: dataframe
        column: the columns specified for the transformation
    Return: the transformed dataframe
    '''
    train_col_mean = train_df[column].mean()
    train_col_std = train_df[column].std()
    train_df[column] = (train_df[column] - train_col_mean) / train_col_std
    test_df[column] = (test_df[column] - train_col_mean) / train_col_std

    return train_df, test_df


def min_max_transformation(train_df, test_df, continuous_columns):
    '''
    This function is used to perform min-max scaling for all variables in the
    training data, and transform both training and testing dataframe
    Inputs:
        train_df: training dataframe
        test_df: testing dataframe
    Returns: updated dataframes
    '''
    scaler = MinMaxScaler()
    train_df_cont = scaler.fit_transform(train_df[continuous_columns])
    test_df_cont = scaler.transform(test_df[continuous_columns])

    train_df = train_df.drop(continuous_columns, axis=1).reset_index(drop=True)
    test_df = test_df.drop(continuous_columns, axis=1).reset_index(drop=True)
    train_df = train_df.join(pd.DataFrame(data=train_df_cont, columns=continuous_columns))
    test_df = test_df.join(pd.DataFrame(data=test_df_cont,columns=continuous_columns))

    return train_df, test_df


def combination_indexer(var_len, index_list):
    '''
    This function is used to provide binary index for the list of variables
    which will be used in get_var_combinations function
    Inputs:
        var_len: the length of the variable list
        index_list: list of the indices
    Returns: a list with all possible binary combinations (list of list)
    '''
    if len(index_list) == var_len:
        return [index_list]
    
    else:
        total_list = []
        for i in range(2):
            new_list = index_list + [i]
            total_list += combination_indexer(var_len, new_list)
    
    return total_list


def get_var_combinations(var_list):
    '''
    This function is used to provide all possible combinations of variables
    which will be used in the input of machine learning methods
    Inputs:
        var_list: a list of variables
    Returns: list of all possible combinations of variables in the given
             variable list (list of list)
    '''
    index_combination = combination_indexer(len(var_list), [])
    var_combinations = []
    
    for combination in index_combination:
        var_sublist = []
        for i, val in enumerate(combination):
            if val == 1:
                var_sublist.append(var_list[i])

        var_combinations.append(var_sublist)
    
    return var_combinations