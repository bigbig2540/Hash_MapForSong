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
from pythainlp.segment import segment
from os import listdir
from collections import Counter
import numpy as np
import pandas as pd
import os
from wordcut import Wordcut

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

CLASS_NUM = 3

def fileScan(directory):
    cut_data= list()
    name_list = os.listdir(directory)
    for name in name_list:
        file = open(directory+name,'r', encoding='utf-8')
        data = file.read()
        file.close()
        data = re.sub(r'[^ก-ูเ-์]','', data)
        cut_data.extend(segment(data))
    return cut_data

def getSong(directory):
    file = open(directory,'r', encoding='utf-8')
    data = file.read()
    file.close()
    data = re.sub(r'[^ก-ูเ-์]', '', data)
    return data

def findSong(directory):
    song_dict = dict()
    name_list = os.listdir(directory)
    for name in name_list:
        file = open(directory+name, 'r' , encoding='utf-8')
        data = file.read()
        file.close()
        data = re.sub(r'[^ก-ูเ-์]', '', data)
        new_data = segment(data)
        new_data = set(new_data)
        for word in new_data:
            if((word in song_dict)==False):
                song_dict[word]=1
            else:
                song_dict[word]+=1
    return song_dict

def genDictFeature(TrainDict):
    new_dic = dict()
    for word in TrainDict.keys():
        f = list()
        for i in range(CLASS_NUM*2):
            f.append(0)
        if(word!='Total'):
            sum_emo_word = sum([float(TrainDict[word][emoi]) / TrainDict['Total'][emoi] for emoi in range(CLASS_NUM)])
            sum_emo_song = sum([float(TrainDict[word][emoi]) / TrainDict['Total'][emoi] for emoi in range(CLASS_NUM,CLASS_NUM*2)])
            for idx_f in range(0, CLASS_NUM*2):
                if (idx_f < CLASS_NUM):
                    f[idx_f] = float(TrainDict[word][idx_f]) / TrainDict['Total'][idx_f] / (sum_emo_word)
                else:
                    f[idx_f] = float(TrainDict[word][idx_f]) / TrainDict['Total'][idx_f] / (sum_emo_song)
            new_dic[word]=f
    return new_dic

def genDict(directory):
    s_list=[]
    folder_name_list = os.listdir(directory)

    for x in range(0,CLASS_NUM):
        s=findSong(directory+'/'+folder_name_list[x]+'/')
        s_list.append(s)

    cut_data_list = []
    for x in range(0,CLASS_NUM):
        cut_data = Counter(fileScan(directory+'/'+folder_name_list[x]+'/'))
        cut_data_list.append(cut_data)

    total=[]
    for data_list in cut_data_list:
        total.extend(data_list)

    total = set(total)

    all_word = dict()
    for word in total:
        for x in range(0,CLASS_NUM):
            if((word in s_list[x])==False):
                s_list[x][word] =0

    total_dict = dict()

    for word in total:
        ls = list()
        for i in range(CLASS_NUM):
            ls.append(cut_data_list[i][word])
        for i in range(CLASS_NUM):
            ls.append(s_list[i][word])
        total_dict[word] = ls

    return total_dict

sum_acc=0.0
sum_roc=0.0
#Start here>>
for seed in range(1,11):
    print(seed)
    ratio = 0.3
    num_class = []
    num_class_test = []
    num_class_train = []
    index_class = []
    length = []
    sub_len = []

    folder_name = os.listdir("dataset")
    for f_name in folder_name:
        tmp = os.listdir("dataset" + '/' + f_name)
        num_class.append(len(tmp))

    for x in num_class:
        num_class_test.append(math.floor(x*0.3))

    for x in range(0,CLASS_NUM):
        num_class_train.append(num_class[x] - num_class_test[x])

    for x in num_class:
        sub_len=[]
        sub_len.extend(range(1,x+1))
        length.append(sub_len)

    random.seed(seed)

    for x in length:
        random.shuffle(x)


    folder_name2 = os.listdir("TrainingSet")
    folder_name3 = os.listdir("TestSet")


    i=0
    for x in length:
        for y in x:
            tmp = folder_name2[i]
            name_list2 = os.listdir("TrainingSet/"+folder_name2[i])
            name_list3 = os.listdir("TestSet/" + folder_name3[i])
            for name in name_list2:
                os.remove("TrainingSet" + '/' + folder_name2[i] + '/'+name)
            for name in name_list3:
                os.remove("TestSet" + '/' + folder_name3[i] + '/'+name)
        i = i + 1

    i=0
    for x in length:
        h = 0
        for y in x:
            if h < num_class_train[i] :
                shutil.copy("dataset" + '/' + folder_name2[i] + '/' + str(y)+".txt", "TrainingSet" + '/' + folder_name2[i] + '/')
            else:
                shutil.copy("dataset" + '/' + folder_name3[i] + '/' + str(y) + ".txt","TestSet" + '/' + folder_name3[i] + '/')
            h=h+1

        i = i + 1

    TrainDict = genDict("TrainingSet")

    Total = list()
    for i in range(CLASS_NUM*2):
        Total.append(0)
    for array in TrainDict:
        ct=0
        for num in TrainDict[array]:
            Total[ct] = Total[ct] + num
            ct+=1
    TrainDict['Total'] = Total
    #print(TrainDict['กก'])
    TrainDictFeature = genDictFeature(TrainDict)
    print("Gen dic features end....")
    training_x = list()
    training_y = list()
    test_x = list()
    test_y = list()
    print(TrainDict['Total'])
    folder_name2 = os.listdir("TrainingSet")
    folder_name3 = os.listdir("TestSet")
    i=0
    #print(TrainDict['ข้อแม้'])
    '''
    for x in length:
        tmp = folder_name2[i]
        name_list2 = os.listdir("TrainingSet/"+folder_name2[i])

        for name in name_list2:
            ls_text = segment(getSong("TrainingSet" + '/' + folder_name2[i] + '/'+name))
            f9 = len(ls_text)
            test = Counter(ls_text)
            f10 = len(Counter(test))
            f = list()
            for ii in range(CLASS_NUM * 2):
                f.append(0)
            for word in test.keys():
                if word in TrainDict.keys():
                    for idx_f in range(0,CLASS_NUM*2):
                        f[idx_f] += TrainDictFeature[word][idx_f]*test[word]
            f.append(f9)
            f.append(f10)
            training_x.append(f)
            training_y.append(i)


        name_list3 = os.listdir("TestSet/" + folder_name3[i])
        for name in name_list3:
            ls_text = segment(getSong("TestSet" + '/' + folder_name3[i] + '/'+name))
            f9 = len(ls_text)
            test = Counter(ls_text)
            f10 = len(Counter(test))
            f = list()
            for ii in range(CLASS_NUM * 2):
                f.append(0)
            for word in test.keys():
                if word in TrainDict.keys():
                   for idx_f in range(0,CLASS_NUM*2):
                        f[idx_f] += TrainDictFeature[word][idx_f]*test[word]

            f.append(f9)
            f.append(f10)
            test_x.append(f)
            test_y.append(i)


        i+=1
    '''
    #test_x_panda = pd.DataFrame(test_x).to_csv('19_10_2559/3Mood_2000Feature/seed'+str(seed)+'/'+'seed'+str(seed)+'_test_x.csv')
    test_x = pd.read_csv('19_10_2559/3Mood_2000Feature/seed'+str(seed)+'/'+'seed'+str(seed)+'_test_x.csv',sep=',').values[:,1:].tolist()
    #print(test_x)
    #train_x_panda = pd.DataFrame(training_x).to_csv('19_10_2559/3Mood_2000Feature/seed'+str(seed)+'/'+'seed'+str(seed)+'_train_x.csv')
    training_x = pd.read_csv('19_10_2559/3Mood_2000Feature/seed'+str(seed)+'/'+'seed'+str(seed)+'_train_x.csv',sep=',').values[:,1:].tolist()
    #print(training_x)
    #test_y_panda = pd.DataFrame(test_y).to_csv('19_10_2559/3Mood_2000Feature/seed'+str(seed)+'/'+'seed'+str(seed)+'_test_y.csv')
    test_y = list(np.reshape(pd.read_csv('19_10_2559/3Mood_2000Feature/seed'+str(seed)+'/'+'seed'+str(seed)+'_test_y.csv',sep=',').values[:,1:].tolist(),-1))
    #print(test_y)
    #train_y_panda = pd.DataFrame(training_y).to_csv('19_10_2559/3Mood_2000Feature/seed'+str(seed)+'/'+'seed'+str(seed)+'_train_y.csv')
    training_y = list(np.reshape(pd.read_csv('19_10_2559/3Mood_2000Feature/seed'+str(seed)+'/'+'seed'+str(seed)+'_train_y.csv',sep=',').values[:,1:].tolist(),-1))
    #print(training_y)

    #///

    '''max_train_index = num_class_train.index(max(num_class_train))
    print(max_train_index)
    print(num_class_train)
    p_dict = dict()

    for i in range(0,len(num_class_train)):
        if i != max_train_index:
            percent = num_class_train[max_train_index]- num_class_train[i]
            p_dict[i] = percent
    print(p_dict)


    for i in p_dict.keys():
        idx = (np.where(np.array(training_y) == i))
        tmp = np.array(training_x)
        tmp = tmp[idx]
        for j in range(p_dict[i]):
            training_x.append(list(tmp[j%len(idx)]))
            training_y.append(i)
    print(training_x)
    '''
    pd_traning_x = pd.DataFrame(training_x)
    pd_test_x = pd.DataFrame(test_x)

    x_train_mean = pd_traning_x.mean()
    x_train_std = pd_traning_x.std()
    x_train_normalize = pd.DataFrame()
    x_test_normalize = pd.DataFrame()

    for i in range(0, CLASS_NUM * 2 + 2):
        if (x_train_std[i] != 0):
            x_train_normalize[i] = (pd_traning_x[i] - x_train_mean[i]) / x_train_std[i]
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


    auroc=[]

    for h_layer in range(1,51):
        #print(h_layer)
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
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=h_layer, random_state=1)
            clf.fit(X_train, y_train)
            res = clf.predict_proba(X_test)
            pd_res = pd.DataFrame(res).idxmax(axis=1)


            #print ("test_y",len(np.reshape(ib.transform(y_test),-1)))
            #print("pred_y",len(np.reshape(ib.transform(pd_res.tolist()),-1)))

            fpr, tpr,_ = roc_curve(np.reshape(ib.transform(y_test),-1), np.reshape(ib.transform(pd_res.tolist()),-1))
            roc_auc = auc(fpr, tpr)
            roc_sum += roc_auc
        auroc.append([roc_sum*1.0/5, h_layer])

    print(auroc)
    auroc = np.array(auroc)
    h_layer = int((auroc[np.argmax(auroc[:,0]),1]))
    print(h_layer)


    #print(training_x)
    #print(test_x)
    #print(train_y_trans)
    #print(x_train_normalize)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=h_layer, random_state=1)
    clf.fit(x_train_normalize,train_y_trans)
    res = clf.predict_proba(x_test_normalize)


    pd.DataFrame(res).to_csv('res.csv')
    #print(clf.out_activation_)
    pd_res = pd.DataFrame(res).idxmax(axis=1)


    #print(ib.transform(pd_res.tolist()))

    print('Acc:',accuracy_score(test_y_trans, ib.transform(pd_res.tolist())))
    fpr, tpr,_ = roc_curve(np.reshape(test_y_trans,-1), np.reshape(ib.transform(pd_res.tolist()),-1))
    print('Roc:',auc(fpr, tpr))
    sum_roc+=auc(fpr, tpr)
    sum_acc+=accuracy_score(test_y_trans, ib.transform(pd_res.tolist()))

    print(confusion_matrix(test_y, pd_res.tolist()))


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
roc_mean=sum_roc/10.0
acc_mean=sum_acc/10.0
print('Sum Acc = ',acc_mean)
print('Sum Roc = ',roc_mean)
