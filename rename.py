import os

folder_name = os.listdir("Dataset")
for f_name in folder_name:
    tmp = os.listdir("Dataset"+'/'+f_name)
    index =1
    for x in tmp:
        os.rename("Dataset"+'/'+f_name+'/'+x  ,  "Dataset"+'/'+f_name+'/'+str(index)+".txt")
        index=index+1

