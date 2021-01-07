import os
import random

import numpy as np
#加载用于训练的所有道路的特征及label
from xgboostclassify.SaveDataToDict import readjsontodict

#按指定label数目做测试数据
def loadtrainftdataby_date_lbnum(traindatadir,testdate,links,lbnums):
    # 加载训练数据
    train_X = []  # 样本特征
    train_Y = []  # 样本label

    # 临时存储样本的变量
    countturnnum = 0  # 每读取1000条道路后，按labelproportion统计一次
    tmp_X = []
    tmp_Y = []
    maxcountnum = len(links)
    for linkid in links:
        # 读取样本特征
        ftpath = os.path.join(traindatadir, linkid, 'features.json')
        linkftsdict = readjsontodict(ftpath)
        # 读取样本label
        lbpath = os.path.join(traindatadir, linkid, 'labels.json')
        linklbsdict = readjsontodict(lbpath)

        for date in testdate:
            # 有些道路的部分日期缺失：
            if str(date) in linkftsdict.keys():
                tmp_X = tmp_X + linkftsdict[str(date)]
                tmp_Y = tmp_Y + linklbsdict[str(date)]

        countturnnum += 1
        if countturnnum % 1000 == 0:
            # countturnnum = 0
            tmp_X, tmp_Y = getdatabylblbnums(tmp_X, tmp_Y, lbnums)
            train_X = train_X + tmp_X
            train_Y = train_Y + tmp_Y
            tmp_X = []
            tmp_Y = []
            print('1000 links training data sampled ' + str(countturnnum))
        elif countturnnum == maxcountnum:
            tmp_X, tmp_Y = getdatabylblbnums(tmp_X, tmp_Y, lbnums)
            train_X = train_X + tmp_X
            train_Y = train_Y + tmp_Y
            tmp_X = []
            tmp_Y = []
            print('all links training data sampled ' + str(countturnnum))
    # 转换成numpy数组的形式
    train_X = np.array((train_X))
    train_Y = np.array((train_Y))

    return train_X, train_Y
def getdatabylblbnums(X,Y,labelnums):
    # 获取各个label的index
    label1_indexes = getlistitmindex(Y, 1)
    label2_indexes = getlistitmindex(Y, 2)
    label3_indexes = getlistitmindex(Y, 3)

    # lb1num = len(label1_indexes)
    # lb2num = len(label2_indexes)
    # lb3num = len(label3_indexes)

    label1_indexes = getlbindexesbylbnum(label1_indexes, labelnums[0])
    label2_indexes = getlbindexesbylbnum(label2_indexes, labelnums[1])
    label3_indexes = getlbindexesbylbnum(label3_indexes, labelnums[2])

    # 按各个label的抽样后的index list，从原始训练数据中提取训练样本
    label_indexes = label1_indexes + label2_indexes + label3_indexes
    sample_X = getsamplefrlistindex(X, label_indexes)
    sample_Y = getsamplefrlistindex(Y, label_indexes)

    return sample_X, sample_Y

#从label_index的list中按给定label数目
def getlbindexesbylbnum(label_indexes,labelnum):
    random.seed(10)#设置种子使得每次抽样结果相同
    if len(label_indexes)>=labelnum:
        return random.sample(label_indexes, int(labelnum))
    else:
        print('this label num'+str(len(label_indexes))+'is less equal'+str(labelnum))
        return label_indexes
#按道路训练数据日期加载数据
def loadtrainftdataby_date(traindatadir,testdate,links):
    train_X = []  # 样本特征
    train_Y = []  # 样本label

    for linkid in links:
        #读取样本特征
        ftpath = os.path.join(traindatadir,linkid,'features.json')
        linkftsdict = readjsontodict(ftpath)
        # 读取样本label
        lbpath = os.path.join(traindatadir, linkid, 'labels.json')
        linklbsdict = readjsontodict(lbpath)
        for date in testdate:
            #有些道路的部分日期缺失：
            if str(date) in linkftsdict.keys():
                train_X=train_X+linkftsdict[str(date)]
                train_Y = train_Y+linklbsdict[str(date)]

    # 转换成numpy数组的形式
    train_X = np.array((train_X))
    train_Y = np.array((train_Y))
    return train_X, train_Y

#按道路训练数据的label比例和日期加载数据
def loadtrainftdataby_lbpp_date(traindatadir,traindate,links,labelproportion):
    # 加载训练数据
    train_X = []  # 样本特征
    train_Y = []  # 样本label

    #临时存储样本的变量
    countturnnum = 0#每读取1000条道路后，按labelproportion统计一次
    tmp_X = []
    tmp_Y = []
    maxcountnum = len(links)
    for linkid in links:
        # 读取样本特征
        ftpath = os.path.join(traindatadir, linkid, 'features.json')
        linkftsdict = readjsontodict(ftpath)
        # 读取样本label
        lbpath = os.path.join(traindatadir, linkid, 'labels.json')
        linklbsdict = readjsontodict(lbpath)

        for date in traindate:
            #有些道路的部分日期缺失：
            if str(date) in linkftsdict.keys():
                tmp_X=tmp_X+linkftsdict[str(date)]
                tmp_Y = tmp_Y+linklbsdict[str(date)]

        countturnnum += 1
        if countturnnum%1000 == 0:
            # countturnnum = 0
            tmp_X,tmp_Y = getdatabylbpropt(tmp_X,tmp_Y,labelproportion)
            train_X = train_X+tmp_X
            train_Y = train_Y+tmp_Y
            tmp_X = []
            tmp_Y = []
            print('1000 links training data sampled '+str(countturnnum))
        elif countturnnum == maxcountnum:
            tmp_X, tmp_Y = getdatabylbpropt(tmp_X, tmp_Y, labelproportion)
            train_X = train_X + tmp_X
            train_Y = train_Y + tmp_Y
            tmp_X = []
            tmp_Y = []
            print('all links training data sampled ' + str(countturnnum))
    #转换成numpy数组的形式
    # for l in train_X:
    #     if len(l) != 55:
    #         print()
    train_X = np.array((train_X))
    train_Y = np.array(train_Y)

    return train_X,train_Y


def loadtrainftdata(traindatadir,iscsv):
    links = os.listdir(traindatadir)

    #加载训练数据
    train_X = []#样本特征
    train_Y = []#样本label

    for linkid in links:
        #读取样本特征
        ftpath = os.path.join(traindatadir,linkid,linkid+'_features'+iscsv)
        lines = np.genfromtxt(ftpath,delimiter=",")#按指定数据格式读取numpy数字数据
        train_X=train_X+list(lines)

        # 读取样本label
        lbpath = os.path.join(traindatadir, linkid, linkid + '_labels'+iscsv)
        lines = np.genfromtxt(lbpath,delimiter=",")#按指定数据格式读取numpy数字数据
        train_Y = train_Y+list(lines)

    #转换成numpy数组的形式
    train_X = np.array((train_X))
    train_Y = np.array((train_Y))

    return train_X,train_Y


#加载用于训练的所有道路的特征及label，因数据量太大，按划分的份分次加载和训练
def loadtrainftdatabynum(traindatadir,iscsv,links):
    #加载训练数据
    train_X = []#样本特征
    train_Y = []#样本label

    for linkid in links:
        #读取样本特征
        ftpath = os.path.join(traindatadir,linkid,linkid+'_features'+iscsv)
        lines = np.genfromtxt(ftpath,delimiter=",")#按指定数据格式读取numpy数字数据
        train_X=train_X+list(lines)

        # 读取样本label
        lbpath = os.path.join(traindatadir, linkid, linkid + '_labels'+iscsv)
        lines = np.genfromtxt(lbpath,delimiter=",")#按指定数据格式读取numpy数字数据
        train_Y = train_Y+list(lines)

    #转换成numpy数组的形式
    train_X = np.array((train_X))
    train_Y = np.array((train_Y))

    return train_X,train_Y

#按label的比例，以label3的数量作为基数获取其他label的比例
#如果一条道路label3或label2缺失，则其他状态按一定数量采样
def loadtrainftdataby_lbproportion(traindatadir,iscsv,links,labelproportion):
    # 加载训练数据
    train_X = []  # 样本特征
    train_Y = []  # 样本label

    #临时存储样本的变量
    countturnnum = 0#每读取1000条道路后，按labelproportion统计一次
    tmp_X = []
    tmp_Y = []
    maxcountnum = len(links)
    for linkid in links:
        #读取样本特征
        ftpath = os.path.join(traindatadir,linkid,linkid+'_features'+iscsv)
        lines = np.genfromtxt(ftpath,delimiter=",")#按指定数据格式读取numpy数字数据
        tmp_X=tmp_X+list(lines)

        # 读取样本label
        lbpath = os.path.join(traindatadir, linkid, linkid + '_labels'+iscsv)
        lines = np.genfromtxt(lbpath,delimiter=",")#按指定数据格式读取numpy数字数据
        tmp_Y = tmp_Y+list(lines)

        countturnnum += 1
        if countturnnum%1000 == 0:
            # countturnnum = 0
            tmp_X,tmp_Y = getdatabylbpropt(tmp_X,tmp_Y,labelproportion)
            train_X = train_X+tmp_X
            train_Y = train_Y+tmp_Y
            tmp_X = []
            tmp_Y = []
            print('1000 links training data sampled '+str(countturnnum))
        elif countturnnum == maxcountnum:
            tmp_X, tmp_Y = getdatabylbpropt(tmp_X, tmp_Y, labelproportion)
            train_X = train_X + tmp_X
            train_Y = train_Y + tmp_Y
            tmp_X = []
            tmp_Y = []
            print('all links training data sampled ' + str(countturnnum))
    #转换成numpy数组的形式
    train_X = np.array((train_X))
    train_Y = np.array((train_Y))

    return train_X,train_Y

#按label比例，从全体训练数据中抽取样本
def getdatabylbpropt(X,Y,labelpropotion):
    #获取各个label的index
    label1_indexes = getlistitmindex(Y,1)
    label2_indexes = getlistitmindex(Y,2)
    label3_indexes = getlistitmindex(Y,3)

    lb3num = len(label3_indexes)
    #按label3的数量或者（如果label2、3数目较少）按一定数量比例抽取label1和label2中的样本
    if lb3num > 2000:
        label1_indexes = getlbindexesbylbppt(label1_indexes,lb3num,labelpropotion[0])
        label2_indexes = getlbindexesbylbppt(label2_indexes,lb3num,labelpropotion[1])
    else:
        label1_indexes = getlbindexesbylbppt(label1_indexes,2000,3)
        label2_indexes = getlbindexesbylbppt(label2_indexes,2000,1)


    #按各个label的抽样后的index list，从原始训练数据中提取训练样本
    label_indexes = label1_indexes + label2_indexes + label3_indexes
    sample_X = getsamplefrlistindex(X,label_indexes)
    sample_Y = getsamplefrlistindex(Y,label_indexes)

    return sample_X,sample_Y
#按原始list的index list从原始list中提取子样本list
def getsamplefrlistindex(originlist,indexlist):
    samplelist = []
    for index in indexlist:
        samplelist.append(originlist[index])
    return samplelist

#从label_index的list中按label3的数目和label比例抽取样本
def getlbindexesbylbppt(label_indexes,basenum,lbpropotion):
    random.seed(10)#设置种子使得每次抽样结果相同
    if len(label_indexes)>=basenum*lbpropotion:
        return random.sample(label_indexes, int(basenum*lbpropotion))
    else:
        print('this label num'+str(len(label_indexes))+'is less equal'+str(basenum*lbpropotion))
        return label_indexes

#获取list中指定元素item的所有索引
def getlistitmindex(lst, item):
        return [index for (index, value) in enumerate(lst) if value == item]


if __name__ == '__main__':
    datadir = 'test'
    loadtrainftdataby_lbproportion()
    print()