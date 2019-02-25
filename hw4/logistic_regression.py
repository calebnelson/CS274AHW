## CS 274 2019, Homework 4, Skeleton Python Code for Logistic Regression
## Written by Caleb Nelson
## (Adapted from code originally written in Winter 2017 by Eric Nalisnick, UC Irvine)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import math

### Helper Functions ###

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_fn(z):
    return np.log(sigmoid(z))

def load_data_pairs(type_str):
    return pd.read_csv("data/"+type_str+"_x.csv").values, pd.read_csv("data/"+type_str+"_y.csv").values

def run_log_reg_model(x, beta):
    return sigmoid(np.dot(x, beta))

def calc_log_likelihood(x, y, beta):
    theta_hats = run_log_reg_model(x, beta)
    return np.sum(np.multiply(y, np.log(theta_hats+1e-100)) + np.multiply((1 - y), 1-np.log(theta_hats+1e-100))) / x.shape[0]

def calc_accuracy(x, y, beta):
    theta_hats = run_log_reg_model(x, beta)
    correct = 0
    for i in range(y.shape[0]):
        if ((theta_hats[i] >= 0.5) == y[i]):
            correct += 1
    return correct / y.shape[0]

def get_AdaM_update(alpha_0, grad, adam_values, b1=.95, b2=.999, e=1e-8):
    adam_values['t'] += 1
    
    # update mean
    adam_values['mean'] = b1 * adam_values['mean'] + (1-b1) * grad
    m_hat = adam_values['mean'] / (1-b1**adam_values['t'])

    # update variance
    adam_values['var'] = b2 * adam_values['var'] + (1-b2) * grad**2
    v_hat = adam_values['var'] / (1-b2**adam_values['t'])

    return alpha_0 * m_hat/(np.sqrt(v_hat) + e)


### Model Training ###

def train_logistic_regression_model(x, y, beta, learning_rate, batch_size, max_epoch, alr=False):
    beta = copy.deepcopy(beta)
    n_batches = math.ceil(x.shape[0]/batch_size)
    train_progress = []
    adam_values = {'mean': np.zeros(beta.shape), 'var': np.zeros(beta.shape), 't': 0}

    for epoch_idx in range(max_epoch):
        for batch_idx in range(n_batches):
            sob = batch_idx * batch_size
            eob = sob + batch_size
            x_batch = x[sob:eob]
            y_batch = y[sob:eob]
            theta_hats = run_log_reg_model(x_batch, beta)
            beta_grad = np.dot(x_batch.T, y_batch - theta_hats)

            if (alr == "R-M"):
                adam_values['t'] += 1
                beta += (learning_rate/adam_values['t']) * beta_grad
            elif (alr == "AdaM"):
                beta_update = get_AdaM_update(learning_rate, beta_grad, adam_values)
                beta += beta_update
            elif (alr == "Newton"):
                theta_matrix = np.diag(np.multiply(theta_hats, 1-theta_hats).flatten())
                hessian = np.linalg.multi_dot([x_batch.T, theta_matrix, x_batch])
                antisingular = np.zeros(hessian.shape)
                np.fill_diagonal(antisingular, 1e-10)
                hessian += antisingular
                beta += np.dot(np.linalg.inv(hessian), beta_grad)
            else:
                beta += learning_rate * beta_grad
        
        train_progress.append(calc_log_likelihood(x, y, beta))
        print ("Epoch %d.  Train Log Likelihood: %f" %(epoch_idx, train_progress[-1]))
        
    return beta

def problem_two():
    ### Set training parameters
    learning_rates = [1e-3, 1e-2, 1e-1]
    batch_sizes = [train_x.shape[0], 1000, 100, 10]
    max_epochs = 250
    
    ### Iterate over training parameters, testing all combinations
    valid_ll = []
    valid_acc = []
    test_acc = []
    results = []

    for lr in learning_rates:
        for bs in batch_sizes:
            ### train model
            final_params = train_logistic_regression_model(train_x, train_y, beta, lr, bs, max_epochs)
    
            ### evaluate model on validation and test data
            valid_ll.append( calc_log_likelihood(valid_x, valid_y, final_params) )
            acc = calc_accuracy(valid_x, valid_y, final_params)
            tacc = calc_accuracy(test_x, test_y, final_params)
            valid_acc.append(acc)
            test_acc.append(tacc)
            results.append((str(lr), str(bs), str(acc), str(tacc)))

    for result in results:
        print("Learning rate: "+result[0]+" Batch Size: "+result[1]+" Validation Acc: "+result[2]+" Test Acc: "+result[3])

def problem_three():
    ### Set training parameters
    batch_sizes = [200, 50]
    max_epochs = 250
    
    ### Iterate over training parameters, testing all combinations
    valid_ll = []
    valid_acc = []
    test_acc = []
    results = []

    for bs in batch_sizes:
        for alr in ["R-M", "AdaM", "Newton"]:
            ### train model
            lr = 0.1 if alr == "R-M" else 0.001
            final_params = train_logistic_regression_model(train_x, train_y, beta, lr, bs, max_epochs, alr)
    
            ### evaluate model on validation and test data
            valid_ll.append( calc_log_likelihood(valid_x, valid_y, final_params) )
            acc = calc_accuracy(valid_x, valid_y, final_params)
            tacc = calc_accuracy(test_x, test_y, final_params)
            valid_acc.append(acc)
            test_acc.append(tacc)
            results.append((alr, str(bs), str(acc), str(tacc)))

    for result in results:
        print("ALR: "+result[0]+" Batch Size: "+result[1]+" Validation Acc: "+result[2]+" Test Acc: "+result[3])

if __name__ == "__main__":

    ### Load the data
    train_x, train_y = load_data_pairs("train")
    valid_x, valid_y = load_data_pairs("valid")
    test_x, test_y = load_data_pairs("test")

    # add a one for the bias term                                                                                                                                                 
    train_x = np.hstack([train_x, np.ones((train_x.shape[0],1))])
    valid_x = np.hstack([valid_x, np.ones((valid_x.shape[0],1))])
    test_x = np.hstack([test_x, np.ones((test_x.shape[0],1))])

    ### Initialize model parameters
    beta = np.random.normal(scale=.001, size=(train_x.shape[1],1))

    print("Running code for problem 2")
    problem_two()

    print("Running code for problem 3")
    problem_three()