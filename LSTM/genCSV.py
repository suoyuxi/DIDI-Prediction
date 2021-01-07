import torch.nn.functional as F 
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from dataset import getData
from model import NET

import os
import csv
import functools
import random
import pandas as pd
import pickle
# from tqdm import tqdm
pd.options.display.max_columns=100

'''
作者：中国科学院大学 索玉玺
内容：数据集成和生成csv结果文件
'''

def lcmp(x,y):

    for i in range(3):
        if float(x[i])>float(y[i]):
            return -1
        if float(x[i])<float(y[i]):
            return 1
    return 0

def predict():

    # #处理嵌入向量
    # with open('embedding_pretrained/linkattr_embedding_dict_N2.pkl','rb') as f:
    #     linkattr_embedding_dict = pickle.load(f)

    # linkattr_embedding_dict = linkattr_embedding_dict.reset_index(drop=True)

    # link_list = []
    # for i in range(15370):
    #     link_list.append(linkattr_embedding_dict.loc[i].values[0])

    #读取数据
    with open('test.txt','r') as f:
        lines = f.readlines()

    with open('attr.txt','r') as f:
        attrs = f.readlines()

    #load model
    net = NET()

    checkpoint = torch.load('NET/net.pt7')
    net.load_state_dict(checkpoint['net'])

    info = []

    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    for i in range(len(lines)):
        data = getData(attrs,lines,i)

        # #处理嵌入向量
        # link_idx_raw = float(data[1])
        # link_idx = link_list.index(link_idx_raw)
        # link_attr1 = [linkattr_embedding_dict.loc[link_idx].values[i] for i in range(2,22)]
        # link_attr2 = [linkattr_embedding_dict.loc[link_idx].values[i] for i in range(47,52)]
        # link_attr = link_attr1 + link_attr2

        current_slice_id = data[2]
        future_slice_id = data[3]        
        index = data[1]

        current_seq = data[4]
        history_seq_1 = data[5][0]
        history_seq_2 = data[5][1]
        history_seq_3 = data[5][2]
        history_seq_4 = data[5][3]
        history_seq_5 = data[6][0]
        history_seq_6 = data[6][1]
        history_seq_7 = data[6][2]
        history_seq_8 = data[6][3]
        history_seq_9 = data[6][4]

        current_seq = Variable(torch.tensor(current_seq)).unsqueeze(0)
        history_seq_1 = Variable(torch.tensor(history_seq_1)).unsqueeze(0)
        history_seq_2 = Variable(torch.tensor(history_seq_2)).unsqueeze(0)
        history_seq_3 = Variable(torch.tensor(history_seq_3)).unsqueeze(0)
        history_seq_4 = Variable(torch.tensor(history_seq_4)).unsqueeze(0)
        history_seq_5 = Variable(torch.tensor(history_seq_5)).unsqueeze(0)
        history_seq_6 = Variable(torch.tensor(history_seq_6)).unsqueeze(0)
        history_seq_7 = Variable(torch.tensor(history_seq_7)).unsqueeze(0)
        history_seq_8 = Variable(torch.tensor(history_seq_8)).unsqueeze(0)
        history_seq_9 = Variable(torch.tensor(history_seq_9)).unsqueeze(0)
        # link_attr = Variable(torch.tensor(link_attr)).type(torch.FloatTensor).unsqueeze(0)

       
        #prediction 
        # #使用嵌入向量
        # x = net(history_seq_1,history_seq_2,history_seq_3,history_seq_4,history_seq_5,history_seq_6,history_seq_7,history_seq_8,history_seq_9,current_seq,link_attr,batch_size = 1)
        #不使用嵌入向量
        x = net(history_seq_1,history_seq_2,history_seq_3,history_seq_4,history_seq_5,history_seq_6,history_seq_7,history_seq_8,history_seq_9,current_seq,batch_size = 1)
        
        data_out = x

        _,label_pre = torch.max(data_out,1)

        label_pre = label_pre.data.item() + 1

        if label_pre == 3:
            print(i)
            cnt3 = cnt3 + 1

        if label_pre == 2:
            cnt2 = cnt2 + 1

        if label_pre == 1:
            cnt1 = cnt1 + 1  

        newinfo = [str(index),str(current_slice_id),str(future_slice_id),str(label_pre)]

        info.append(newinfo)
    
    print('拥堵：',cnt3)
    print('缓行：',cnt2)
    print('畅行：',cnt1)
    random.shuffle(info)
    return info

#投票集成输出
def voke():

    num = 176056
    #需要自行生成csv，然后根据线上分数加权投票
    weight = [0.47,0.471,0.4728,0.4748,0.45,0.451,0.452,0.4583,0.4674]
    root = 'result'
    info = []
    result = []
    for i in weight:
        result.append([])
    for i in range(len(weight)):
        path = 'result/'+str(weight[i])+'/result.csv'
        with open(path, 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                result[i].append(row)
        del result[i][0]



    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    for i in range(num):
        vote = [0.0,0.0,0.0]
        for j in range(len(weight)):
            for k in range(3):
                if result[j][i][3] == str(k+1):
                    if k == 2:
                        vote[k] = vote[k] + weight[j]*(k+1)*(k+1)
                    else:
                        vote[k] = vote[k] + weight[j]*(k+1)*(k+1)

        vote_result = vote.index(max(vote)) + 1
        vote_result = str(vote_result)
        newinfo = []
        newinfo.append(result[0][i][0])
        newinfo.append(result[0][i][1])
        newinfo.append(result[0][i][2])
        newinfo.append(vote_result)

        info.append(newinfo)

        
        if vote_result == '1':
            cnt1 = cnt1 + 1
        if vote_result == '2':
            cnt2 = cnt2 + 1
        if vote_result == '3':
            cnt3 = cnt3 + 1
    print(cnt1)
    print(cnt2)
    print(cnt3)

    return info

#写入csv结果
def genCSV(infos,path):

    file = open(path,'a',newline='')
    csv_write = csv.writer(file,dialect='excel',lineterminator='\n')
    csv_head = ['link','current_slice_id','future_slice_id','label']
    csv_write.writerow(csv_head)

    #重新排序 id 当前时间片 待预测时间片 label后 写入result
    for info in infos:
        csv_write.writerow(info)
    file.close()

#主控函数
if __name__ == "__main__":
    
    #生成提交结果
    infos = predict()
    genCSV(infos,'result.csv')
    infos = sorted(infos, key=functools.cmp_to_key(lcmp),reverse=True)
    genCSV(infos,'result_sort.csv')

    # #投票集成
    # infos = voke()
    # genCSV(infos,'result.csv')