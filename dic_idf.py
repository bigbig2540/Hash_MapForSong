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

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

CLASS_NUM = 3
file = open('preposition.txt','r', encoding='utf-8')
pep = file.read().split(' ')
#print(pep)

def removekey(dict):
    n_dict =dict
    for pepword in pep:
        if(pepword in n_dict.keys()):
            del n_dict[pepword]
    return n_dict

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

        text = list(new_data)


        new_data.extend(ngram(text, 3)) #tri
        new_data.extend(ngram(text, 2))
       # print(ngram(text, 3))

        new_data = set(new_data)
        for word in new_data:
            if((word in song_dict)==False):
                song_dict[word]=1
            else:
                song_dict[word]+=1
    return song_dict,len(name_list)

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
    each_num_song = []
    num_song=0
    for x in range(0,CLASS_NUM):
        s,num_song_tmp=findSong(directory+'/'+folder_name_list[x]+'/')
        each_num_song.append(num_song_tmp)
        num_song+=num_song_tmp
        s_list.append(s)
    entropy = 0
    for x in range(0,CLASS_NUM):
        entropy+= (1.0*each_num_song[x]/num_song)*math.log10(1.0*each_num_song[x]/num_song)
    '''
    print(num_song)
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
'''
    total=[]
    for dic_tmp in s_list:
        for word in dic_tmp.keys():
            total.append(word)


    for word in total:
        for x in range(0, CLASS_NUM):
            if ((word in s_list[x]) == False):
                s_list[x][word] = 0
   # print(s_list)
    total_dict = dict()

    for word in total:
        ls = list()
        for i in range(CLASS_NUM):
            ls.append(s_list[i][word])
        if  (sum(ls) >5):
            total_dict[word] = math.log10(1.0*num_song/sum(ls))

    return total_dict,entropy*-1


def genDictOldVer(directory):
    s_list = []
    folder_name_list = os.listdir(directory)
    num_song = 0

    for x in range(0, CLASS_NUM):
        s, num_song_tmp = findSong(directory + '/' + folder_name_list[x] + '/')
        num_song += num_song_tmp
        s_list.append(s)


   # print(num_song)
    cut_data_list = []
    for x in range(0,CLASS_NUM):
        cut_data = Counter(fileScan(directory+'/'+folder_name_list[x]+'/'))
        cut_data_list.append(cut_data)
    '''
    total=[]
    for data_list in cut_data_list:
        total.extend(data_list)

    total = set(total)

    all_word = dict()
    for word in total:
        for x in range(0,CLASS_NUM):
            if((word in s_list[x])==False):
                s_list[x][word] =0
'''
    total = []
    for dic_tmp in s_list:
        for word in dic_tmp.keys():
            total.append(word)

    for word in total:
        for x in range(0, CLASS_NUM):
            if ((word in s_list[x]) == False):
                s_list[x][word] = 0
  #  print(s_list)
    total_dict = dict()

    for word in total:
        ls = list()
        for i in range(CLASS_NUM):
            ls.append(cut_data_list[i][word])
        for i in range(CLASS_NUM):
            ls.append(s_list[i][word])
        total_dict[word] = ls

    return total_dict

def ngram(input,n):
    output = []
    for i in range(len(input)-n+1):
        tmp = ""
        for x in range(n):
            tmp += input[x+i]
            #output.append(input[i:i+n])
        output.append(tmp)
    return output




#Start here>>
#print(ngram(['ฉัน','รัก','เธอ','นะ'],3))
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



    TrainDict,Entropy = genDict("TrainingSet")
    TrainDictVerOld = genDictOldVer("TrainingSet")
    TrainDictFre=dict()
    TrainDictIG = dict()
    #print(TrainDictVerOld['เธอ'][2])


    for word in TrainDictVerOld:
        entropy_word = 0
        sum_word=0
        word_x = []
        for x in range(CLASS_NUM,CLASS_NUM*2):
            word_x.append(TrainDictVerOld[word][x])
            sum_word += TrainDictVerOld[word][x]
          #  print(word)
          #  print(TrainDictVerOld[word][x])
       # print(word_x)
        for x in range(0,CLASS_NUM):
            entropy_word+=(1.0*word_x[x]/sum_word)*math.log10(1.0*word_x[x]/sum_word+1e-9 )
      #  print(entropy_word)
        entropy_word = entropy_word* -1
        TrainDictFre[word] = sum_word
        if(Entropy-entropy_word < 0):
            TrainDictIG[word] =0
        else:
            TrainDictIG[word] = Entropy-entropy_word
    IG_list = []
    for word in TrainDictIG:
        if(TrainDictIG[word] <= 0):
            pep.append(word)
        elif(TrainDictFre[word] <= 1):
            pep.append(word)
       # IG_list.append(list([word,TrainDictFre[word],TrainDictIG[word]]))
   # pd.DataFrame(IG_list).to_csv('Ig_list2.csv')
   # print(TrainDictVerOld['รู้ว่าคงไม่'])
    print('Entropy'+str(Entropy))

   # print(TrainDictIG['รู้ว่าคงไม่'])
   # print(TrainDict.keys())
   # print(TrainDict['ของ'])
    pep=list(set(pep))
    TrainDict = removekey(TrainDict)
    Traindic_list = []
    for word in TrainDict:
        Traindic_list.append(list([word,TrainDictFre[word],TrainDictIG[word]]))
    pd.DataFrame(Traindic_list).to_csv('TrainDictRemove.csv')
    '''
    Total = list()
    for i in range(CLASS_NUM*2):
        Total.append(0)
    for array in TrainDict:
        ct=0
        for num in TrainDict[array]:
            Total[ct] = Total[ct] + num
            ct+=1
    TrainDict['Total'] = Total
    '''

    print(len(TrainDict))
    Word_list = []
    for word in TrainDict.keys():
        Word_list.append(word)

    training_x = list()
    training_y = list()
    test_x = list()
    test_y = list()

    folder_name2 = os.listdir("TrainingSet")
    folder_name3 = os.listdir("TestSet")
    i=0


    for x in length:
        tmp = folder_name2[i]
        name_list2 = os.listdir("TrainingSet/"+folder_name2[i])

        print(tmp)

        for name in name_list2:
            #print(name)
            ls_text = segment(getSong("TrainingSet" + '/' + folder_name2[i] + '/'+name))
            text = list(ls_text)
            ls_text.extend(ngram(text, 2))
            ls_text.extend(ngram(text, 3)) #tri
            #print((ngram(text, 3)))
            test = Counter(ls_text)
            f = list()
            #print(f_bi)
            #print(name)

            for ii in range(len(Word_list)):
                f.append(0)

            for word in test.keys():
                if word in Word_list:
                    f[Word_list.index(word)] += test[word]*TrainDict[word]

            training_x.append(f)
            training_y.append(i)


        name_list3 = os.listdir("TestSet/" + folder_name3[i])
        for name in name_list3:
            ls_text = segment(getSong("TestSet" + '/' + folder_name3[i] + '/'+name))

            text = list(ls_text)
            ls_text.extend(ngram(text, 2))
            ls_text.extend(ngram(text, 3))  # tri

            test = Counter(ls_text)
            f = list()

            for ii in range(len(Word_list)):
                f.append(0)
            for word in test.keys():
                if word in Word_list:
                    f[Word_list.index(word)] += test[word]*TrainDict[word]

            test_x.append(f)
            test_y.append(i)


        i+=1

    #print(training_x)
    #print(test_x)
    print("finish")
    print(len(training_x[0]))





    #////////////////////////////////////////////////////////////////////////////////////////////////////////
    '''
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

    test_x_panda = pd.DataFrame(test_x).to_csv('seed'+str(seed)+'test_x_tri_gram.csv')
    #test_x = pd.read_csv(str(seed)+'mood/seed'+str(seed)+'/'+'seed'+str(seed)+'_test_x.csv',sep=',').values[:,1:].tolist()
    #print(test_x)
    train_x_panda = pd.DataFrame(training_x).to_csv('seed'+str(seed)+'train_x_tri_gram.csv')
    #training_x = pd.read_csv(str(seed)+'mood/seed'+str(seed)+'/'+'seed'+str(seed)+'_train_x.csv',sep=',').values[:,1:].tolist()
    #print(training_x)
    test_y_panda = pd.DataFrame(test_y).to_csv('seed'+str(seed)+'test_y_tri_gram.csv')
    #test_y = list(np.reshape(pd.read_csv(str(seed)+'mood/seed'+str(seed)+'/'+'seed'+str(seed)+'_test_y.csv',sep=',').values[:,1:].tolist(),-1))
    #print(test_y)
    train_y_panda = pd.DataFrame(training_y).to_csv('seed'+str(seed)+'train_y_tri_gram.csv')
    #training_y = list(np.reshape(pd.read_csv(str(seed)+'mood/seed'+str(seed)+'/'+'seed'+str(seed)+'_train_y.csv',sep=',').values[:,1:].tolist(),-1))
    #print(training_y)

    #///
    '''
    max_train_index = num_class_train.index(max(num_class_train))
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
    '''
    pd_traning_x = pd.DataFrame(training_x)
    pd_test_x = pd.DataFrame(test_x)

    x_train_mean = pd_traning_x.mean()
    x_train_std = pd_traning_x.std()
    x_train_normalize = pd.DataFrame()
    x_test_normalize = pd.DataFrame()


    for i in range(len(Word_list)):
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

    for h_layer in range(1,5):
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


    auroc = np.array(auroc)
    h_layer = int(np.argmax(auroc[:,0]))+1
    print(auroc)
    print(int(math.pow(10,h_layer)/2))



    #print(x_train_normalize)
    #print(train_y_trans)

    #print(training_x)
    #print(test_x)
    #print(train_y_trans)
    #print(x_train_normalize)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=int(math.pow(10,h_layer)/2), random_state=1,max_iter=500)
    clf.fit(x_train_normalize,train_y_trans)
    res = clf.predict_proba(x_test_normalize)


    pd.DataFrame(res).to_csv('res.csv')
    #print(clf.out_activation_)
    pd_res = pd.DataFrame(res).idxmax(axis=1)


    #print(ib.transform(pd_res.tolist()))

    print('Acc:',accuracy_score(test_y_trans, ib.transform(pd_res.tolist())))
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
