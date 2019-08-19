# Spliting data to train and test
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

half_year_stamp = [(7,1), (1,1)]

# General
def train_test_general(df, features, classifier, test_size=0.1, seed=None):
    '''
    This function provide the most general method to split the data to
    training and testing subset.
    Inputs:
        df: dataframe
        features: features selected
        classifier: classifier used as the indicator of output
        test_size: the proportion of the testing subset
    Returns: X_train, y_train, X_test, y_test
    '''
    X_train, X_test, y_train, y_test = train_test_split(df[features], \
                                                        df[classifier], \
                                                        test_size=test_size,\
                                                        random_state=seed)
    return X_train, y_train, X_test, y_test


# Cross Sections
def train_test_KFold(df, features, classifier, split=4, shuffle=False, random_state=None):
    '''
    This function is used to provide KFold split to training and testing subsets
    Inputs:
        df: dataframe
        features: features selected
        classifier: classifier used as the indicator of output
        split: the number of ways to seperate training and testing subsets
    Returns: dictionary of [X_train, y_train, X_test, y_test]
    '''
    data_dict = {}
    i = 0
    dfs = KFold(n_splits=split, shuffle=shuffle, random_state=random_state)

    for train, test in dfs.split(df):
        X_train = df.iloc[train][features]
        X_test = df.iloc[test][features]
        y_train = df.iloc[train][classifier]
        y_test = df.iloc[test][classifier]
        data_dict[i] = [X_train, y_train, X_test, y_test]
        i += 1

    return data_dict


# Temporal
def time_thresholds_creater(start_date, end_date, time_gap):
    '''
    This function is used to create list of cutoff of the time
    Inputs:
        start_data: starting date of the data
        end_date: ending date of the data
        time_gap: the length of each time interval
    Returns: list of cutoff of the time
    '''
    time_cut_list = [start_date]
    while time_cut_list[-1] + time_gap <= end_date:
        time_cut_list.append(start_date + time_gap)

    return time_cut_list


def gen_time_cuts(start_date, end_date, date_thresholds):
    '''
    This function is used to generate the time cuts of the dataframe given specific
    cutoff date
    Inputs:
        start_data: starting date of the data
        end_date: ending date of the data
        date_thresholds: specific dates used as the cutoff
    Returns: list of cutoff of the time
    '''
    time_cut_list = []
    for month, day in date_thresholds:
        for year in range(start_date.year, end_date.year):
            time_cut = pd.Timestamp(year, month, day)
            if (start_date > time_cut) or (end_date < time_cut):
                pass
            else:
                time_cut_list.append(time_cut)
    time_cut_list = time_cut_list + [end_date]

    return time_cut_list


def define_time_variables(df, time_gap, waiting_time, start_time_col, end_time_col, start_date, end_date, time_thresholds):
    '''
    This function is used to create binary variables to indicate rolling window split
    Inputs:
        df: dataframe
        time_gap: the length of each time unit
        waiting_time: the time gap for the result to be observed
        start_time_col: column representing the start time
        end_time_col: column representing the end time
        start_data: starting date of the data
        end_date: ending date of the data
        time_thresholds: specific dates used as the cutoff
    '''
    # df["time_gap"] = df[end_time_col] - df[start_time_col]
    # df['label'] =  np.where(df['time_gap'] > pd.Timedelta('60 days'), 1, 0)

    time_cut_list = sorted(gen_time_cuts(start_date, end_date, time_thresholds))

    for i, time in enumerate(time_cut_list[1:-1]):
        tmp_label = "tmp_label" + str(i)
        df[tmp_label] = ""
        df.loc[((df[start_time_col] >= start_date) & (df[start_time_col] < time)), tmp_label] = "train"
        df.loc[((df[start_time_col] >= time + waiting_time) & (df[start_time_col] < time + waiting_time + time_gap)), tmp_label] = "test"

    return df


def rolling_window_split(df, time_split_indicators, features, classifier):
    '''
    This function is used to provide temportal split to training and testing subsets with
    rolling window strategy
    Inputs:
        df: dataframe with time_split_indicators
        time_index: the feature used as the time variable
        features: features selected
        classifier: classifier used as the indicator of output
    Returns: dictionary of [X_train, y_train, X_test, y_test]
    '''
    index_dict = {}
    for time_ind in time_split_indicators:
        test = df[df[time_ind] == "test"]
        train = df[df[time_ind] == "train"]
        sub_list = [train[features], train[classifier], test[features], test[classifier]]
        index_dict[time_ind] = sub_list

    return index_dict


# Reset Balance
def reset_balance(df, classifier, sample_times, cls0_size=1, cls1_size=1):
    '''
    This function is used to sample the majority in the unbalance classification
    Inputs:
        df: dataframe
        classifier: name of classifier in the decision tree
        sample_times: time used for resampling
        cls0_size: relative size of classification 0
        cls1_size: relative size of classification 1
    Returns: list of balanced dataframe
    '''
    df_list = []
    
    if cls0_size >= cls1_size:
        base_size = df[df[classifier] == 1].shape[0]
        sample_size = base_size * cls0_size
        base_df = df[df[classifier] == 1]
        sample_df = df[df[classifier] == 0]

    else:
        base_size = df[df[classifier] == 0].shape[0]
        sample_size = base_size * cls1_size
        base_df = df[df[classifier] == 0]
        sample_df = df[df[classifier] == 1]

    for i in range(sample_times):
        sub_sample_df = sample_df.sample(sample_size, axis=0)
        whole_sub_df = pd.concat([sub_sample_df, base_df], join="inner")
        df_list.append(whole_sub_df)

    return df_list