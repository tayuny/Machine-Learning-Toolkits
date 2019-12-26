import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sys

#########################################################################################
##                     Singular Values and Eigenvalues                                 ##
#########################################################################################
def plot_sigma(sigma):
    '''
    The function is used to plot the values of singular values
    Input: diagonal matrix sigma
    '''
    sin_index = np.arange(1, len(sigma) + 1, 1)
    plt.plot(sin_index, sigma)
    plt.xlabel("index for singular values")
    plt.ylabel("singular values")


def plot_accumulative_singular_value_importance(sigma, squared=False):
    '''
    The function is used to plot the accumulated proportion of singular values
    , if squared is True, eigenvalue is used instead
    Inputs:
        sigma: diagonal matrix sigma
        squared: boolean
    '''
    if squared:
        sigma_perc = (sigma ** 2) / np.sum(sigma ** 2)
    else:
        sigma_perc = sigma / np.sum(sigma)
    
    accu_perc_sigma_list = []
    accu_perc_sigma = 0
    for i, sig in enumerate(sigma_perc):
        accu_perc_sigma += sig
        accu_perc_sigma_list.append(accu_perc_sigma)
    
    sin_index = np.arange(1, len(sigma) + 1, 1)
    plt.plot(sin_index, np.array(accu_perc_sigma_list))
    plt.xlabel("index for singular values")
    plt.ylabel("accumulated percentage of singular values")


def svd_cal_for_evd(df):
    '''
    The function is used to implement eigenvalue decomposition
    with SVD
    Inputs: dataframe
    Return: Eigenvector and eigenvalue
    '''
    U, sigma, Vt = np.linalg.svd(df, full_matrices=False)
    eigvec = Vt.T
    eigval = sigma ** 2
    return eigvec, eigval


##########################################################################################
##                            SVD and Dimensional Reduction                             ##
##########################################################################################
def preprocess_sigma(sigma, matrix_n, matrix_p):
    '''
    The function is used to generate a matrix for sigma
    given sigularvalues and the size of sigma matrix
    Inputs:
        sigma: list of sigularvalues
        matrix_n: the number of rows for the sigma matrix
        matrix_p: the number of columns for the sigma matrix
    Return: sigma matrix
    '''
    diag_s = np.zeros([matrix_n, matrix_p])

    for i, sing in enumerate(sigma):
        diag_s[i, i] = sing

    return diag_s


def reverse_svd(U, sigma, Vt):
    '''
    The function is used to reversely calculate the matrix
    with SVD components
    Inputs: U, sigma, Vt
    Return: matrix (original or approximated)
    '''
    df = (U.dot(preprocess_sigma(sigma, U.shape[1], Vt.shape[0]))).dot(Vt)
    return df


def dimensional_reduction(df, k, y, allm=False, get_weight=False, 
                          get_test_df=False, test_df=""):
    '''
    The function is designed to make dimensional reduction with SVD method
    Inputs:
        df: original matrix
        k: the number of singular values taken
        y: label vector
        allm: if True, return the components of economic SVD
        get_weight: if True, return the weight calculated by the economic SVD
                    components
    Returns: approximated df with k singular values
    '''
    U_k = np.linalg.svd(df)[0][:, :k]
    sigma_k = np.linalg.svd(df)[1][:k]
    Vt_k = np.linalg.svd(df)[2][:k, :]
    
    if get_weight:
        return (np.transpose(Vt_k)).dot(((np.linalg.pinv(
                preprocess_sigma(sigma_k, k, k)).dot(U_k.T)).dot(y)))
    
    if allm:
        return U_k, sigma_k, Vt_k

    reduced_df = (Vt_k[:k, :].dot(df.T)).T
    
    if get_test_df:
        return reduced_df, (Vt_k[:k, :].dot(test_df.T)).T
    
    return reduced_df


def dimensional_reduction_two_direction(df, k, y, direction="rows", 
                                        coeff=False):
    '''
    This function is used to do dimensional reduction with
    the direction for the row and column
    Inputs:
        df: dataframe
        k: target dimension
        y: labels
        direction: "rows" or "columns"
        coeff: if true, return the loading of the principle components
    Return: the reduced form of dataframe
    '''
    U_k = np.linalg.svd(df)[0][:, :k]
    sigma_k = np.linalg.svd(df)[1][:k]
    Vt_k = np.linalg.svd(df)[2][:k, :]
    
    if direction == "rows":
        reduced_df = (Vt_k[:k, :].dot(df.T)).T
        if coeff:
            return reduced_df, np.diag(sigma_k).dot(U_k.T)
        return reduced_df

    if direction == "columns":
        reduced_df = (U_k[:, :3].T).dot(df)
        if coeff:
            return reduced_df, np.diag(sigma_k).dot(Vt_k)
        return reduced_df


def plotting_3D(df, x_col, y_col, z_col, labels):
    '''
    The function is used to plot the interactive 3D plot
    Inputs:
        df: the dataframe
        x_col: column for the x axis
        y_col: column for the y axis
        z_col: column for the z axis
        labels: list of labels of y used for coloring the points
    '''
    if labels == "no":
        plot_r = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, 
                           width=1000, height=800)
    else:
        plot_r = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, 
                               color=labels, width=1000, height=800)
    plot_r.show()


#########################################################################################
##                            Power Iteration                                          ##
#########################################################################################
def power_iteration(df, init_pi, tol=10**(-8)):
    '''
    The function is used to operation power iteration to find the
    eigenvector corresponding to the largest eigenvalue
    Inputs:
        df: training dataframe
        init_pi: initial guess of the eigenvector
        tol: tolerance of difference
    Return: eigenvector corresponding to the largest eigenvalue
    '''
    pi = init_pi
    for i in np.arange(0, max(df.shape[0], df.shape[1]), 1):
        new_pi = df.dot(pi) / ((np.sum( (df.dot(pi) ** 2))) ** 0.5)
        if ((np.sum((new_pi - pi)** 2)) ** 0.5) < tol:
            return pi
        else:
            if i < 5:
                print("pi is updated with", (np.sum((new_pi - pi)** 2)) ** 0.5)
            pi = new_pi


##########################################################################################
##                             Gram-Schmidt                                             ##
##########################################################################################
def gram_schmidt(X):
    '''
    This function is used to operate gram_schmidt
    methods to find orthogonal basis
    Input: X matrix
    Return: U matrix
    '''
    U = np.zeros([X.shape[0], X.shape[1]])
    
    if (np.sum(X[:, 0] ** 2, axis=0) ** (0.5)) != 0:
        U[:, 0] = X[:, 0] / (np.sum(X[:, 0] ** 2, axis=0) ** (0.5))  
    else: 
        U[:, 0] = 0
    current_U = U[:, [0]]
    
    for col in np.arange(1, X.shape[1]):

        if np.any(X[: , col]) != 0:
            X_bar = X[: , col] - current_U.dot(np.transpose(current_U)).dot(X[: , col])
        else:
            X_bar = np.zeros([X.shape[0], 1])

        current_U = np.zeros([X.shape[0], col + 1])
        current_U[:, :col] = U[:, :col]
        
        if np.sum(X_bar ** 2, axis=0) ** (0.5) != 0:
            U[:, col] = X_bar / (np.sum(X_bar ** 2, axis=0) ** (0.5))
        else:
            U[:, col] = 0
    
    return U