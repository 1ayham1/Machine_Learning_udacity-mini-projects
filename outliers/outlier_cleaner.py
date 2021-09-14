#!/usr/bin/python

import numpy as np


def outlierCleaner(predictions, ages, net_worths):
    """ Clean away the 10% of points that have the largest residual errors
    (difference between the prediction and the actual net worth).

    Parameters:
    
    predictions: predictions is a list of predicted targets that come from regression
    ages       : ages is the list of ages in the training set 
    net_worths : the actual value of the net worths in the training set. 
    
    Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    error = np.abs(np.array(predictions)-np.array(net_worths))
    sorted_error = np.argsort(error,axis=0)
    
    perc_to_remove = 0.1
    
    total_keep = int(len(sorted_error)*(1-perc_to_remove))
    
    sorted_idx = sorted_error[total_keep::-1].squeeze()


    for elem in sorted_idx:
        
        age = ages[elem]
        net_worth = net_worths[elem]
        error = sorted_error[elem]

        cleaned_data.append((age,net_worth,error))
   
    return cleaned_data

