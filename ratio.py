import os
import math
import random
import shutil
import re
from shutil import copyfile

from pythainlp.segment import segment
from os import listdir
from collections import Counter
import numpy as np
import pandas as pd
import os

def fileScan(directory):
    cut_data=""
    name_list = os.listdir(directory)
    for name in name_list:
        file = open(directory+name,'r', encoding='utf-8')
        data = file.read()
        file.close()
        data = re.sub(r'[^ก-ูเ-์]','', data)
        cut_data = cut_data + data
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

def genDict(directory):
    s_list=[]
    folder_name_list = os.listdir(directory)

    for x in range(0,4):
        s=findSong(directory+'/'+folder_name_list[x]+'/')
        s_list.append(s)

    cut_data_list = []
    for x in range(0,4):
        cut_data = Counter(segment(fileScan(directory+'/'+folder_name_list[x]+'/')))
        cut_data_list.append(cut_data)

    total=[]
    for data_list in cut_data_list:
        total.extend(data_list)

    total = set(total)

    all_word = dict()
    for word in total:
        for x in range(0,4):
            if((word in s_list[x])==False):
                s_list[x][word] =0

    total_dict = dict()

    for word in total:
        total_dict[word] = [cut_data_list[0][word],cut_data_list[1][word],cut_data_list[2][word],cut_data_list[3][word],s_list[0][word],s_list[1][word],s_list[2][word],s_list[3][word]]

    return total_dict

#Start here>>
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

for x in range(0,4):
        num_class_train.append(num_class[x] - num_class_test[x])

for x in num_class:
    sub_len=[]
    sub_len.extend(range(1,x+1))
    length.append(sub_len)

seed = 1
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

Total = [0, 0, 0, 0, 0, 0, 0, 0]
for array in TrainDict:
    ct=0
    for num in TrainDict[array]:
        Total[ct] = Total[ct] + num
        ct+=1
TrainDict['Total'] = Total

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
    for y in x:
        tmp = folder_name2[i]
        name_list2 = os.listdir("TrainingSet/"+folder_name2[i])
        for name in name_list2:
            ls_text = segment(getSong("TrainingSet" + '/' + folder_name2[i] + '/'+name))
            f9 = len(ls_text)
            f10 = len(Counter(ls_text))
            f = [0, 0, 0, 0, 0, 0, 0, 0]
            test = set(ls_text)
            for word in test:
                if word in TrainDict.keys():
                    #print(word)
                    #print(folder_name2[i], name, word)
                    sum_emo_word = sum([float(TrainDict[word][emoi]) /TrainDict['Total'][emoi] for  emoi in [0,1,2,3]])
                    sum_emo_song = sum([float(TrainDict[word][emoi]) / TrainDict['Total'][emoi] for emoi in [4, 5, 6, 7]])
                    for idx_f in range(0,8):
                        #print(word,idx_f, float(TrainDict[word][idx_f]),TrainDict['Total'][idx_f], sum_emo_song, float(TrainDict[word][1]) / TrainDict['Total'][1], float(TrainDict[word][3]) / TrainDict['Total'][3],
                        #float(TrainDict[word][5]) / TrainDict['Total'][5], float(TrainDict[word][7]) / TrainDict['Total'][7])
                        #print(folder_name2[i],name, word)
                        if(idx_f<4):
                            f[idx_f] += float(TrainDict[word][idx_f])/TrainDict['Total'][idx_f]/(sum_emo_word)
                        else:
                            f[idx_f] += float(TrainDict[word][idx_f]) / TrainDict['Total'][idx_f] / (sum_emo_song)
            f.append(f9)
            f.append(f10)
            training_x.append(f)
            training_y.append(i)


     """   name_list3 = os.listdir("TestSet/" + folder_name3[i])
        for name in name_list3:
            ls_text = segment(getSong("TestSet" + '/' + folder_name3[i] + '/'+name))
            f9 = len(ls_text)
            f10 = len(Counter(ls_text))
            f = [0, 0, 0, 0, 0, 0, 0, 0]
            for word in ls_text:
                if word in TrainDict.keys():
                    sum_emo_word = sum([float(TrainDict[word][emoi])/TrainDict['Total'][emoi] for  emoi in [0,1,2,3]])
                    sum_emo_song = sum([float(TrainDict[word][emoi]) / TrainDict['Total'][emoi] for emoi in [4, 5, 6, 7]])
                    for idx_f in range(0,8):
                        if(idx_f%2 == 0):
                            f[idx_f] += float(TrainDict[word][idx_f])/TrainDict['Total'][idx_f]/(sum_emo_word)
                        else:
                            f[idx_f] += float(TrainDict[word][idx_f]) / TrainDict['Total'][idx_f] / (sum_emo_song)
            f.append(f9)
            f.append(f10)
            test_x.append(f)
            test_y.append(i)
    """

    print(i)
    i+=1

print(training_x)
print(training_y)
print(test_x)
print(test_y)

print(TrainDict['Total'])

print(length)

print(num_class)

print(num_class_train)

print(num_class_test)