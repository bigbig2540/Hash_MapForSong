import os
import math
import random
import shutil
import re
from numpy import array
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
import pandas as pd
from shutil import copyfile
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#from pythainlp.segment import segment
from os import listdir
from collections import Counter
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC


import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

CLASS_NUM = 3
#file = open('preposition.txt','r', encoding='utf-8')
#pep = file.read().split(' ')
#print(pep)
for seed in range(1,2):
    #test_x_panda = pd.DataFrame(test_x).to_csv('test_x.csv')
    test_x = pd.read_csv('seed'+str(seed)+'test_x_tri_gram.csv',sep=',').values[:,1:].tolist()
    #print(test_x)
    #train_x_panda = pd.DataFrame(training_x).to_csv('train_x.csv')
    training_x = pd.read_csv('seed'+str(seed)+'train_x_tri_gram.csv',sep=',').values[:,1:].tolist()
    #print(training_x)
    #test_y_panda = pd.DataFrame(test_y).to_csv('test_y.csv')
    test_y = list(np.reshape(pd.read_csv('seed'+str(seed)+'test_y_tri_gram.csv',sep=',').values[:,1:].tolist(),-1))
    #print(test_y)
    #train_y_panda = pd.DataFrame(training_y).to_csv('train_y.csv')
    training_y = list(np.reshape(pd.read_csv('seed'+str(seed)+'train_y_tri_gram.csv',sep=',').values[:,1:].tolist(),-1))
    #print(training_y)


    pd_traning_x = pd.DataFrame(training_x)
    pd_test_x = pd.DataFrame(test_x)

    x_train_mean = pd_traning_x.mean()
    x_train_std = pd_traning_x.std()
    x_train_normalize = pd.DataFrame()
    x_test_normalize = pd.DataFrame()


    for i in range(len(training_x[0])):
        if(x_train_std[i]!=0):
            x_train_normalize[i] = (pd_traning_x[i]-x_train_mean[i])/x_train_std[i]
            x_test_normalize[i] = (pd_test_x[i] - x_train_mean[i]) / x_train_std[i]
        else:
            x_train_normalize[i] = 0
            x_test_normalize[i] = 0

    #print(x_test_normalize)

    ib = preprocessing.LabelBinarizer()
    ib.fit(test_y)
    test_y_trans=ib.transform(test_y)

    ib.fit(training_y)
    train_y_trans=ib.transform(training_y)
    h_layer = 5
    #5fol-d
    #print(train_y_trans)

    #ssss

    auroc=[]



    #nueron

    for h_layer in range(3,4):
        print(int(math.pow(10,h_layer)/2))
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5)
        skf.get_n_splits(x_train_normalize, train_y_trans)
        #print(skf)
        roc_sum = 0
        training_x = np.array(x_train_normalize)
        training_y = np.array(training_y)

        for train_index, test_index in skf.split(training_x, training_y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = training_x[train_index], training_x[test_index]
            y_train, y_test = training_y[train_index], training_y[test_index]
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=int(math.pow(10,h_layer)/2), random_state=1,max_iter=500)
            clf.fit(X_train, y_train)
            res = clf.predict_proba(X_test)
            pd_res = pd.DataFrame(res).idxmax(axis=1)


            #print ("test_y",len(np.reshape(ib.transform(y_test),-1)))
            #print("pred_y",len(np.reshape(ib.transform(pd_res.tolist()),-1)))

            fpr, tpr,_ = roc_curve(np.reshape(ib.transform(y_test),-1), np.reshape(ib.transform(pd_res.tolist()),-1))
            roc_auc = auc(fpr, tpr)
            roc_sum += roc_auc
    auroc.append([roc_sum*1.0/5, int(math.pow(10,h_layer)/2)])
    print(h_layer)

    print(x_train_normalize)
    print(train_y_trans)

    # print(training_x)
    # print(test_x)
    # print(train_y_trans)
    # print(x_train_normalize)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=h_layer, random_state=1)
    clf.fit(x_train_normalize, train_y_trans)
    res = clf.predict_proba(x_test_normalize)

    pd.DataFrame(res).to_csv('res.csv')
    # print(clf.out_activation_)
    pd_res = pd.DataFrame(res).idxmax(axis=1)

    # print(ib.transform(pd_res.tolist()))

    print('Acc:', accuracy_score(test_y_trans, ib.transform(pd_res.tolist())))
    fpr, tpr, _ = roc_curve(np.reshape(test_y_trans, -1), np.reshape(ib.transform(pd_res.tolist()), -1))
    print('Roc:', auc(fpr, tpr))

    print(confusion_matrix(test_y, pd_res.tolist()))

    # svm
'''
    for c in range(5,6):
        for g in range(5, 6):
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5)
            skf.get_n_splits(x_train_normalize, train_y_trans)
            # print(skf)
            roc_sum = 0
            training_x = np.array(x_train_normalize)
            training_y = np.array(training_y)
            print(math.pow(10, c), " ", math.pow(10, g))
            for train_index, test_index in skf.split(training_x, training_y):
                # print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = training_x[train_index], training_x[test_index]
                y_train, y_test = training_y[train_index], training_y[test_index]


                #clf = SVC(C=math.pow(10, c), cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovo', degree=3, gamma=math.pow(10, g), kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True, verbose=False)

                clf = SVC()
                #print(y_train)
                clf.fit(X_train, y_train)
                res = clf.predict(X_train)
                pd_res = pd.DataFrame(res).idxmax(axis=1)
                print(pd_res.tolist())
                # print ("test_y",len(np.reshape(ib.transform(y_test),-1)))
                # print("pred_y",len(np.reshape(ib.transform(pd_res.tolist()),-1)))

                fpr, tpr, _ = roc_curve(np.reshape(ib.transform(y_train), -1), np.reshape(ib.transform(pd_res.tolist()), -1))
                roc_auc = auc(fpr, tpr)
                roc_sum += roc_auc
            auroc.append([roc_sum * 1.0 / 5, c, g])

    auroc = np.array(auroc)
    roc_sum, c, g  = auroc[int(np.argmax(auroc[:,0]))]
    print(auroc)
    #print(int(math.pow(10,h_layer)/2))

    #print(x_train_normalize)
    #print(train_y_trans)

    #print(training_x)
    #print(test_x)
    #print(train_y_trans)
    #print(x_train_normalize)
    clf = SVC(C=math.pow(10, c), cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovo', degree=3,
              gamma=math.pow(10, g), kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
              verbose=False)
    training_x = np.array(x_train_normalize)
    training_y = np.array(training_y)
    clf.fit(training_x, training_y)
    #res = clf.predict(y_train)
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=int(math.pow(10,h_layer)/2), random_state=1,max_iter=500)

    res = clf.predict(x_test_normalize)


    pd.DataFrame(res).to_csv('res.csv')
    #print(clf.out_activation_)
    pd_res = pd.DataFrame(res).idxmax(axis=1)


    #print(ib.transform(pd_res.tolist()))

    print('Acc:',accuracy_score(test_y, pd_res.tolist()))
    fpr, tpr,_ = roc_curve(np.reshape(test_y_trans,-1), np.reshape(ib.transform(pd_res.tolist()),-1))
    print('Roc:',auc(fpr, tpr))
    matrix = np.array(confusion_matrix(test_y, pd_res.tolist()))
    print(confusion_matrix(test_y, pd_res.tolist()))



    with open('workfile4.txt', 'a') as f:
        f.write('\nSeed:' + str(seed))
        f.write('\nAcc:'+ str(accuracy_score(test_y_trans, ib.transform(pd_res.tolist()))))
        f.write('\nRoc:' + str(auc(fpr, tpr)))
        f.write('\n')
        for i in range(CLASS_NUM):
            f.write('[')
            for j in range(CLASS_NUM):
                f.write(str(matrix[i,j]) + ' ')
            f.write(']\n')
'''

'''
'''

'''
import csv
myfile = open('seed1_y.csv', 'w')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(training_y)


myfile = open('seed1_x.csv', 'w')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(training_x)


print(training_x)
print(training_y)
print(test_x)
print(test_y)

print(TrainDict['Total'])

print(length)

print(num_class)

print(num_class_train)

print(num_class_test)
'''
