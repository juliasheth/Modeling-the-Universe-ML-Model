#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:02:45 2018

@author: Julia
"""

import numpy as np
from matplotlib import pyplot as plt

def euclidean_distance(a,b):
    diff = a - b
    return np.sqrt(np.dot(diff, diff))


def load_data(csv_filename):
    exo_file = open(csv_filename,'r')
    file_list = []
    exo_lines = exo_file.readlines()
    for line in exo_lines[1:]:
        line = line.strip();
        parts = line.split(',')
        length = len(parts)
        parts = parts[1:length]
        parts2 = []
        for item in parts:
            parts2.append(float(item))
        file_list.append(parts2)
    dataset = np.array(file_list)
    return dataset
    
    
def split_data(dataset, ratio = 0.9):
    number_of_rows = dataset.shape[0]
    number_train_rows = int(number_of_rows * ratio)
    training_set = dataset[0:number_train_rows]
    test_set = dataset[number_train_rows+1:]
    train_test = (training_set,test_set)
    return train_test

    
def compute_centroid(data):
    centroid = sum(data)/float(len(data))
    return centroid
    

def experiment(exo_train, nonexo_train, exo_test, nonexo_test):
    exo_centroid = compute_centroid(exo_train)
    nonexo_centroid = compute_centroid(nonexo_train)
    correct_exo = 0
    total_exo = 0
    correct_nonexo = 0
    total_nonexo = 0
    for item in exo_test:
        label = ''
        exo_distance = euclidean_distance(item,exo_centroid)
        nonexo_distance = euclidean_distance(item,nonexo_centroid)
        if (exo_distance < nonexo_distance):
            label = 'exo'
            correct_exo += 1
        elif (nonexo_distance < exo_distance):
            label = 'nonexo'
        total_exo += 1
    for item in nonexo_test:
        label = ''
        exo_distance = euclidean_distance(item,exo_centroid)
        nonexo_distance = euclidean_distance(item,nonexo_centroid)
        if (exo_distance < nonexo_distance):
            label = 'exo'
        elif (nonexo_distance < exo_distance):
            label = 'nonexo'
            correct_nonexo += 1
        total_nonexo += 1
    accuracy_exo = correct_exo/total_exo
    accuracy_nonexo = correct_nonexo/total_nonexo
    total = total_exo + total_nonexo
    correct = correct_exo + correct_nonexo
    accuracy = (accuracy_exo*(total_exo/total)) + (accuracy_nonexo*(total_nonexo/total))
    print(("Total exo predictions: {}, Correct exo predictions: {}, Accuracy: {}").format(total_exo,correct_exo,accuracy_exo))
    print(("Total nonexo predictions: {}, Correct nonexo predictions: {}, Accuracy: {}").format(total_nonexo,correct_nonexo,accuracy_nonexo))
    print (("Total predictions: {}, Correct predictions: {}, Accuracy: {}").format(total,correct,accuracy))
    print()
    #could just divide added accuracies by 2 to get average accuracy
    return accuracy


def learning_curve(exo_training, nonexo_training, exo_test, nonexo_test):
    np.random.shuffle(exo_training)
    np.random.shuffle(nonexo_training)
    n = len(exo_training)
    accuracy_list = []
    for i in range(n-1):
        exo_train = exo_training[0:i+1]
        nonexo_train = nonexo_training[0:i+1]
        accuracy = experiment(exo_train,nonexo_train,exo_test,nonexo_test)
        accuracy_list.append(accuracy)
    x_values = np.linspace(1,n,n-1)
    y_values = accuracy_list
    return plt.plot(x_values,y_values)

    
def cross_validation(exo_data, nonexo_data, k):
    size_of_exo_partition = int(len(exo_data)/k)
    size_of_nonexo_partition = int(len(nonexo_data)/k)
    exo_set = []
    nonexo_set = []
    accuracy_total = 0
    for i in range(k):
        exo_partition = exo_data[((i)*size_of_exo_partition):((i+1)*size_of_exo_partition)]
        nonexo_partition = nonexo_data[((i)*size_of_nonexo_partition):((i+1)*size_of_nonexo_partition)]
        exo_set.append(exo_partition)
        nonexo_set.append(nonexo_partition)
    for x in range(k):
        exo_train_top = exo_data[0:((x)*size_of_exo_partition)]
        exo_train_bottom = exo_data[((x+1)*size_of_exo_partition):]
        exo_train = np.vstack((exo_train_top,exo_train_bottom))
        nonexo_train_top = nonexo_data[0:((x)*size_of_nonexo_partition)]
        nonexo_train_bottom = nonexo_data[((x+1)*size_of_nonexo_partition):]
        nonexo_train = np.vstack((nonexo_train_top,nonexo_train_bottom))
        exo_test = exo_set[x]
        nonexo_test = nonexo_set[x]
        accuracy = experiment(exo_train,nonexo_train,exo_test,nonexo_test)
        accuracy_total += accuracy
    average_accuracy = accuracy_total/k
    return average_accuracy
 
    
if __name__ == "__main__":
    
    exo_data = load_data('exoStars.csv')
    nonexo_data = load_data('nonExoStars.csv')

    #experiment
    exo_train, exo_test = split_data(exo_data, 0.5)
    nonexo_train, nonexo_test = split_data(nonexo_data, 0.5)
    experiment(exo_train, nonexo_train, exo_test, nonexo_test)
    
    #get learning curve
    exo_train, exo_test = split_data(exo_data, 0.5)
    nonexo_train, nonexo_test = split_data(nonexo_data, 0.5)
    learning_curve(exo_train, nonexo_train, exo_test, nonexo_test)
    
    #perform cross validation using partitions
    k = 10
    acc = cross_validation(exo_data, nonexo_data,k)
    print("{}-fold cross-validation accuracy: {}".format(k,acc))
    
