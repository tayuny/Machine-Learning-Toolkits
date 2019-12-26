import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sys

#################################################################################
##                             Preprocessing                                   ##
#################################################################################
def generate_multiple_labels(y):
    '''
    This function is used to generate multiple
    binary labels for different categories for y
    Input: y labels
    Return: matrix for binary labels
    '''
    categories = sorted(np.unique(y))
    Y = np.zeros([y.shape[0], len(categories)])
    for i, category in enumerate(categories):
        Y[:, i] = 0
        Y[np.where(y == category), i] = 1

    return Y


def binary_to_multiple_label(Y, label_list):
    '''
    This function is used to transform multiple
    binary labels to a categorical label y
    Input: 
        Y: matrices for multiple binary labels
        labels_list: label for the categorical variable
    Return: categorical label y
    '''
    y_class = np.zeros(Y.shape[0])
    for i, label in enumerate(label_list):
        y_class[np.where(Y[:, i] == 1)] = label

    return y_class


#################################################################################
##                             Optimizers                                      ##
#################################################################################
def get_least_square_weight(X, y):
    '''
    This function is used to calculate the weight with least square
    Inputs:
        X: matrix of features
        y: vector of the label
    Return: weight
    '''
    w = ((np.linalg.inv((np.transpose(X)).dot(X))).dot(
                         np.transpose(X))).dot(y)
    return w


def least_square_regularized(X, y, k, lamb, reg=True, full=False):
    '''
    This function is used to calculate the weight with different setting
    Inputs:
        X: dataframe
        y: label
        k: k dimension in SVD setting
        lamb: regularization parameters for ridge regression
        reg: (boolean) using ridge regresison or not
        full: use the full matrix after SVD or not
    Return: weights vector
    '''
    if full:
        U_k, sigma_k, Vt_k = np.linalg.svd(X, full_matrices=False)
    
    else:
        U_k, sigma_k, Vt_k = dimensional_reduction(X, k, y, allm=True, 
                                                   get_weight=False)
    
    if reg:
        inv_ele = np.linalg.inv((np.diag(sigma_k).T).dot(
                                np.diag(sigma_k)) + 
                                lamb * np.diag(np.repeat(1, np.diag(sigma_k).shape[0])))

        rweight = np.array(((Vt_k.T.dot(inv_ele)).dot(
                             np.diag(sigma_k).T)).dot(U_k.T)).dot(y)
    
    else:
        pinv = ((np.diag(sigma_k)).T).dot(
                 np.linalg.inv(np.diag(sigma_k).dot((np.diag(sigma_k)).T)))
        return Vt_k.T.dot(pinv).dot(U_k.T).dot(y)
        
    return rweight


def ridge_regression_gradient(X, y, learning_r, lamb, max_iter, init_w, tol=10 ** (-8)):
    '''
    The function is used to calculate the weight of ridge regression using gradient descent
    setting
    Inputs:
        X: feature matrix
        y: labels
        learning_r: assigned learning rate
        lamb: parameter for regularizer
        max_iter: maximum time of iterations
        init_w: initial values for weights
        tol: tolerance
    Return: weight
    '''
    # setting initital w and learning_r are critical
    if learning_r > (1 / (np.sum(X ** 2) ** 0.5)):
        print("the learing rate is too high")
        return
    
    w_old = init_w
    current_err = np.inf
    for i in np.arange(0, max_iter, 1):
        gradt = (-2 * (X.T.dot(y)) + 2 * (X.T.dot(X).dot(w_old)) + 2 * lamb * w_old)
        w_new = w_old - (learning_r * gradt)
        
        if i == max_iter:
            print("mat_iter is reached")

        if (np.sum((X.dot(w_new) - y) ** 2) ** (0.5)) > current_err:
            print("currently at i: ", i)
            print("the function is not converging")
            break
        
        current_err = (np.sum((X.dot(w_new) - y) ** 2) ** (0.5)) 
        if current_err < tol:
            break
        
        if sum((w_new - w_old) ** 2) ** (0.5) > tol:
            w_old = w_new

    return w_new


def stochastic_ridge_gradient(X, y, learning_r, lamb, max_iter, 
                              init_w=np.repeat(1, X.shape[1]), tol=10 ** (-10)):
    '''
    The function is used to calculate the weight of ridge regression using stochastic gradient descent
    setting
    Inputs:
        X: feature matrix
        y: labels
        learning_r: assigned learning rate
        lamb: parameter for regularizer
        max_iter: maximum time of iterations
        init_w: initial values for weights
        tol: tolerance
    Return: weight
    '''
    w_old = init_w
    current_err = np.inf
    for i in np.arange(0, max_iter, 1):
        idx = np.random.randint(0, X.shape[0])
        gradt = (-2 * (X[idx] * y[idx]) + 2 * (np.inner((X[idx] ** 2), w_old)) + 2 * lamb * w_old)
        w_new = w_old - (learning_r * gradt)
        
        current_err = (np.sum((X.dot(w_new) - y) ** 2) ** (0.5)) 
        if current_err < tol:
            print("the improvement of error is limited after iteration: ", i)
            break

        if sum((w_new - w_old) ** 2) ** (0.5) < tol:
            print("the improvement of weight is limited after iteration: ", i)
        else:
            w_old = w_new
        
        if i == max_iter - 1:
            print("max_iter is reached")
        
    return w_new


class kernel:
    '''
    The kernel object includes the value and functions used in kernel calculation
    '''
    def __init__(self, X, kernel_type, q):
        self.kernel_type = kernel_type
        self.X = X
        self.q = q

    def full_K(self):
        '''
        The method is used to calcualte the full kernel matrix
        '''
        if self.kernel_type == "polynomial_q_full":
            return np.apply_along_axis(self.polynomial_q_full_kernel, 1, self.X)

        if self.kernel_type == "polynomial_q":
            return np.apply_along_axis(self.polynomial_q_kernel, 1, self.X)

    def polynomial_q_full_kernel(self, y):
        '''
        The method provide the kernel calcultion with polynomial of degree <= q
        '''
        return (self.X.dot(y) + 1) ** self.q

    def polynomial_q_kernel(self, y):
        '''
        The method provide the kernel calcultion with polynomial of degree q
        '''
        return (self.X.dot(y)) ** self.q


def polynomial_ridge(X, kernel_type, q, lamb):
    '''
    The function is used to calculate the weight of ridge regression using kernel methods
    Inputs:
        X: feature matrix
        kernel_type: kernel function chosen
        q: degree of kernel functions
        lamb: parameter for regularizer
    Return: weight
    '''
    X_ker = X_ker = kernel(X, kernel_type, q)
    alpha_new = np.linalg.inv(X_ker.full_K() + lamb * np.diag(np.ones(X.shape[0]))).dot(y)
    return X.T.dot(alpha_new)


def SVM_hinge_gradient(X, y, q, lamb, init_alpha, learning_r, max_iter, kernel_type='polynomial_q_full', tol=10 ** (-6)):
    '''
    The function is used to calculated the weight using SVM with gradient descent setting
    Inputs:
        X: feature matrix
        y: labels
        q: degree of kernel method
        lamb: parameter for regularizer
        init_alpha: initial values for weights
        learning_r: assigned learning rate
        max_iter: maximum time of iterations
        kernel_type: kernel method chosen
        tol: tolerance
    Return: weight
    '''
    X_ker = kernel(X, kernel_type, q)
    K = X_ker.full_K()
    # kernel matrix K is symmetric
    print("K shape: ", K.shape)

    alpha_old = init_alpha
    for i in np.arange(0, max_iter, 1):
        hinge_indicator = np.zeros([X.shape[0], 1])
        alpha_tmp = np.multiply(y, K.T.dot(alpha_old))
        hinge_indicator[alpha_tmp > 1] = 1
        outer = 2 * lamb * K.dot(alpha_old)
        
        gradt = np.sum(hinge_indicator * (- y) * K.T, axis=0).reshape([X.shape[0], 1]) + outer
        alpha_new = alpha_old - (learning_r * gradt)
        
        if np.sum((alpha_new - alpha_old) ** 2, axis=0) ** (0.5) < tol:
            print("the function has error lower than the tolerance")
            return X.T.dot(alpha_new)
        else:
            alpha_old = alpha_new

    print("the function reachs its maxium iteration")
    return X.T.dot(alpha_new)


##################################################################################
##                             Predictions                                      ##
##################################################################################
def predict_y(X, w):
    '''
    This function is used to predict y with weight from least square
    method
    Inputs:
        X: matrix of features
        w: weight calculated from least square method
    Return: prediction for y
    '''
    y_hat = X.dot(w)
    return y_hat


def predict_y_class(X, w):
    '''
    This function is used to make the prediction for the class of y
    Inputs:
        X: matrix of features
        w: weight calculated from least square method
    Return: predicted class for y
    '''
    y_class = np.zeros((X.shape[0], 1))
    y_hat = X.dot(w)
    y_class[np.where(y_hat >= 0)] = 1
    y_class[np.where(y_hat < 0)] = -1
    return y_class


def classify_y_multi(Y_pred):
    '''
    This function is used to choose the label with highest
    predicted probability as the predicted label
    Input: predicted probabilities for multiple binary labels
    Return: y with categorical labels
    '''
    Y_class = np.zeros([Y_pred.shape[0], Y_pred.shape[1]])
    maxrow = np.argmax(Y_pred, axis=1)

    return maxrow


###################################################################################
##                             Evaluations                                       ##
###################################################################################
def performance_square_error(pred_y, real_y):
    '''
    This function is used to calculate the average squared
    error of the predicted value
    Inputs:
        pred_y: predicted y
        real_y: real label for y
    Return: average squared error
    '''
    return sum((pred_y - real_y) ** 2) / len(pred_y)


def validate_prediction(y_class, y_label):
    '''
    This function is used to calculated the error rate
    for the prediction
    Inputs:
        y_class: predicted y_class
        y_label: real y label
    Return: error rate
    '''
    count_vec = np.zeros([y_class.shape[0], 1])
    count_vec[y_label == y_class] = 1
    return 1 - (sum(count_vec) / y_class.shape[0])