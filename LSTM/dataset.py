import torch as t
import numpy as np
import os
from torch.utils import data
import math
import random

'''
作者：中国科学院大学 索玉玺
内容：获取数据
'''

def getLength(attrs,index):

    attr = attrs[index].split('\t')
    length = float(attr[1])

    if length <=60:
        length = 0.01*length
    elif (length>60) and (length<=100):
        length = 0.005*(length-60)+0.6
    elif (length>100) and (length<=400):
        length = 0.0005*(length-100)+0.8
    elif (length>400) and (length<1000):
        length = 0.05*(length-400)/600 + 0.95
    else:
        length = 1.0

    return length

def data_norm(current_seq, history_seq):

    #处理current_seq
    for i in range(5):
        current_seq[i][0] = (current_seq[i][0] - 30) / 10#处理速度
        current_seq[i][1] = (current_seq[i][1] - 30) / 10#处理eta速度
        current_seq[i][2] = (current_seq[i][2] - 2) / 2#处理state
        current_seq[i][3] = (current_seq[i][3] - 14) / 18#车辆数目，均值14，标准差18

    #处理history_seq
    for j in range(4):
        for i in range(5):
            history_seq[j][i][0] = (history_seq[j][i][0] - 30) / 10#处理速度
            history_seq[j][i][1] = (history_seq[j][i][1] - 30) / 10#处理eta速度
            history_seq[j][i][2] = (history_seq[j][i][2] - 2) / 2#处理state
            history_seq[j][i][3] = (history_seq[j][i][3] - 14) / 18#均值14，标准差18
    return current_seq, history_seq

def getData(attrs,lines,idx):

    line = lines[idx]

    splitBySpace = line.split(' ')


    #int型的id，获取当前路段的id
    index = int(splitBySpace[0]) 

    #获得分段处理后的道路长度
    length = getLength(attrs,index)

    #int型的label，即待预测时间片的标签
    label = int(splitBySpace[1])
        
    #当前时间片
    time_current = int(splitBySpace[2])
    time_slice = (float(time_current)-360)/360

    #待预测时间片
    time_prediction = int(splitBySpace[3].split(';')[0])

    #时间差
    time_diff = time_prediction - time_current

    '''
    float32型的np二维数组shape(5,4)
    当前节点从time_current-4到time_current的特征序列(5序列长)
    信息分别是速度、eta速度、label、车辆数目(4特征数)，又添加了道路id以增添其区域位置的特征关系，共计5个特征
    '''
    node_seq = []
    node_seq.append(splitBySpace[3].split(';')[1])
    for i in range(4,8):
        node_seq.append(splitBySpace[i])

    for i in range(len(node_seq)):
        node_seq[i] = node_seq[i].split(':')[1]
        node_seq[i] = node_seq[i].split(',')
        for j in range(len(node_seq[i])):
            if ';' in node_seq[i][j]:
                node_seq[i][j] = node_seq[i][j].split(';')[0]

            node_seq[i][j] = float(node_seq[i][j])

        local_index = (float(index)-300000)/600000
        node_seq[i].append(float(local_index))
        node_seq[i].append(length)
        node_seq[i].append(time_slice)       

    '''
    float32型的三维数组shape(4,5,4)
    第一维的4代表了-28，-21，-14，-7四天的数据,这个维度相当于C（channel）
    第二维的5是这五个时间片，prediction_time到prediction_time+4
    第三维的5是当天某个时间片的四个特征，特征信息分别是速度、eta速度、速度除以eta速度、label、车辆数目(4特征数)
    '''
    history_seq = [[], [], [], []]
    splitBySemicolon = line.split(';')[2:6]
    for i in range(4):
        history_seq[i] = splitBySemicolon[i].split(' ')
        for j in range(5):
            history_seq[i][j] = history_seq[i][j].split(':')[1]
            history_seq[i][j] = history_seq[i][j].split(',')
            for k in range(4):
                if i+j+k == 11:
                    history_seq[i][j][k] = history_seq[3][4][3].strip('\n')

                history_seq[i][j][k] = float(history_seq[i][j][k])
            history_seq[i][j].append(float(local_index))
            history_seq[i][j].append(length)
            history_seq[i][j].append(time_slice)

    # history_seq = fill_history(history_seq)
    # node_seq, history_seq = drop(local_index,length,time_slice,node_seq,history_seq)
    history_seq_1 = np.asarray(history_seq, np.float32)
    node_seq = np.asarray(node_seq, np.float32)
    current_seq, history_seq_1 = data_norm(node_seq, history_seq_1)

    '''
    float32型的三维数组shape(5,4,4)
    第一维的5是这五个时间片，prediction_time到prediction_time+4
    第二维的4代表了-28，-21，-14，-7四天的数据,这个维度相当于C（channel）
    第三维的5是当天某个时间片的四个特征，特征信息分别是速度、eta速度、速度除以eta速度、label、车辆数目(4特征数)
        '''
    history_seq_2 = np.swapaxes(history_seq_1, 0, 1)
                
    # 为了方便作交叉熵损失，label减去1
    return label-1, index, time_current, time_prediction, current_seq.copy(), history_seq_1.copy(), history_seq_2.copy(), time_diff

#获得验证集
def getValidSet():

    #读取相关数据
    with open('20190729.txt','r') as f:
        lines = f.readlines()
    with open('attr.txt','r') as f:
        attrs = f.readlines()

    #初始化计数列表
    list1 = []
    list2 = []
    list3 = []
    for i in range(500000):
        data = getData(attrs,lines,i)
        if data[0] == 0:
            list1.append(i)
        elif data[0] == 1:
            list2.append(i)
        else:
            list3.append(i)
            print(i)
    
    valid1 = random.sample(list1, 118889)
    valid2 = random.sample(list2, 3597)
    valid3 = random.sample(list3, 1060)

    valid = valid1+valid2+valid3

    #写入
    for i in valid:

        with open('valid.txt','a') as f:
            f.write(lines[i])  

if __name__ == '__main__':

    with open('20190730.txt','r') as f:
        lines = f.readlines()

    with open('attr.txt','r') as f:
        attrs = f.readlines()

    data = getData(attrs,lines,4399)
    print(data[1])
    

    # getValidSet()
    
 

