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
        file = open(directory+name, 'r'  , encoding='utf-8')
        data = file.read()
        file.close()
        data = data.replace(" ", "")
        data = data.replace("\n", "")
        data = data.replace("(", "")
        data = data.replace(")", "")
        data = data.replace(".", "")
        data = data.replace("[", "")
        data = data.replace("]", "")
        data = data.replace("…", "")
        data = data.replace("-", "")
        cut_data = cut_data + data
    return cut_data

def findSong(directory):
    song_dict = dict()
    name_list = os.listdir(directory)
    for name in name_list:
        file = open(directory+name, 'r' , encoding='utf-8')
        data = file.read()
        file.close()
        data = data.replace(" ", "")
        data = data.replace("\n", "")
        data = data.replace("(", "")
        data = data.replace(")", "")
        data = data.replace(".", "")
        data = data.replace("[", "")
        data = data.replace("]", "")
        data = data.replace("…", "")
        data = data.replace("-", "")
        new_data = segment(data)
        new_data = set(new_data)
        for word in new_data:
            if((word in song_dict)==False):
                song_dict[word]=1
            else:
                song_dict[word]+=1
    return song_dict

def genDict(dicrectory):
#directory = input("Your Directory : ")
#directory = "dataset"
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
    total_dict[word] = [cut_data_list[0][word],cut_data_list[1][word],cut_data_list[2][word],cut_data_list[3][word], (cut_data_list[0][word]+cut_data_list[1][word]+cut_data_list[2][word]+cut_data_list[3][word]),s_list[0][word],s_list[1][word],s_list[2][word],s_list[3][word]]

print(total_dict['รัก'])

