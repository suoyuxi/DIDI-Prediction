#读取已经重新整理的json格式道路数据，提取特征，并存储
import csv
import json

#读取存储为json格式的dict数据
import os

count = 0

def readjsontodict(filepath):
    # 读取
    with open(filepath, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)
    return load_dict

#提取一条道路的所有特征信息，以list形式返回
def extractlinkfeatures(linkinfo,linkid,savepath):
    # MissedftsFold = 'E:/My competitions/didi road condition/code/processeddata/MissExtractedfts'
    #获取道路的属性信息
    linkattr = getlinkattr(linkinfo[linkid][-1][0])#[speedlimit,pathclass,speedclass,LaneNum,level,lenth,width]
    # linktimeft = []#[current_slice_id,future_slice_id,curfutdicabs,week,workday]
    linkallfts = {}#link所有提取的fts汇总，采用dict，将训练数据按日期分开
    linkalllbs = {}#link所有labels汇总
    #获取一条道路数据的时间片和日期特征，并按不同日期分开存储
    for onedayfts in linkinfo[linkid][:-1]:
        for ft in onedayfts:
            date = ft[-2]
            # if date == 0:
            #     print(0)
            linktimeft,label = getlinktimeft_label(linkid,ft)#[current_slice_id,future_slice_id,curfutdicabs,week,workday]
            recentft = getlinkrecentft(ft[1],round(float(linkattr[0]), 3))
            # if recentft == 0:#去除缺失值
            #     print(ft)
            #     continue
            historyft = getlinkhistoryft(ft[2:6],linkattr[0])
            if historyft == 0:#去除缺失值,并保存到对应的日期文件中
                # # 缺失值数据信息，保存文件位置
                # Missedftsbydate = os.path.join(MissedftsFold,str(date))
                # Missedftsfile = os.path.join(Missedftsbydate,'traindatamissed.csv')
                # if not os.path.exists(Missedftsbydate):  # 判断是否存在文件夹如果不存在则创建为文件夹
                #     os.makedirs(Missedftsbydate)
                # # 写入csv文件并去除\r\n->\n
                # csvFile = open(Missedftsfile, 'a+', newline='')
                # csv_write = csv.writer(csvFile, dialect='excel', lineterminator='\n')
                # csv_write.writerow(ft)
                # csvFile.close()
                continue

            features = [int(linkid)]#加入linkid
            features.extend(linktimeft)  # 所有特征合并为一个list
            features.extend(linkattr)
            features.extend(recentft)
            for ft in historyft:
                features.extend(ft)
            global count
            count+=1
            if len(features) !=55:
                print(len(features))
                print(features)
            #按日期将道路的特征区分开来
            if date in linkallfts.keys():  # 如果该日期内已经有数据记录
                linkallfts[date].append(features)
                linkalllbs[date].append(label)
            else:  # 如果该日期首次出现
                linkallfts.setdefault(date, []).append(features)
                linkalllbs.setdefault(date, []).append(label)

    saveextractedftsbydatetojson(savepath, linkid, linkallfts, linkalllbs)#保证模块性

#保存提取的特征和label格式json
def saveextractedftsbydatetojson(savepath,linkid,linkallfts,linkalllbs):
    #创建文件夹
    savepath = os.path.join(savepath, linkid)
    # 按道路id存储道路所有的提取的特征
    if not os.path.exists(savepath):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(savepath)  # makedirs 创建文件时如果路径不存在会创建这个路径
    with open(os.path.join(savepath,'features.json'), 'w') as outfile:
        json.dump(linkallfts, outfile, ensure_ascii=False)
        outfile.write('\n')
    outfile.close()
    with open(os.path.join(savepath,'labels.json'), 'w') as outfile:
        json.dump(linkalllbs, outfile, ensure_ascii=False)
        outfile.write('\n')
    outfile.close()

#保存提取的特征和label格式csv
def saveextractedftsbydate(savepath,linkid,linkallfts,linkalllbs):
    #创建文件夹
    savepath = os.path.join(savepath, linkid)
    # 按道路id存储道路所有的提取的特征
    if not os.path.exists(savepath):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(savepath)  # makedirs 创建文件时如果路径不存在会创建这个路径
    # 按日期对道路数据进行存储
    for date in linkallfts.keys():
        # 先创建对应的日期文件夹
        savedirbydate = os.path.join(savepath, str(date))
        if not os.path.exists(savedirbydate):
            os.makedirs(savedirbydate)
        saveftsdata(savedirbydate,linkallfts[date], linkalllbs[date])#保证模块性
#存储数据模块
def saveftsdata(savepath,linkfts,linklbs):
    #所有features的汇总csv
    # alllinksftscsv = 'E:/My competitions/didi road condition/code/processeddata/alllinksfts.csv'
    # alllinklbscsv = 'E:/My competitions/didi road condition/code/processeddata/alllinkslbs.csv'

    # #测试集用
    # alllinksftscsv = 'E:/My competitions/didi road condition/code/processeddata/alltestlink730fts_lbs/test730/test730_features.csv'
    # alllinklbscsv = 'E:/My competitions/didi road condition/code/processeddata/alltestlink730fts_lbs/test730/test730_labels.csv'

    # # 按道路id存储道路所有的提取的特征
    # savefolder = savepath + str(linkid)
    # if not os.path.exists(savefolder):  # 判断是否存在文件夹如果不存在则创建为文件夹
    #     os.makedirs(savefolder)  # makedirs 创建文件时如果路径不存在会创建这个路径
    # 写入每条道路分开的fts
    linkftsfileapart = os.path.join(savepath,'features.csv')
    linkftsfileapf = open(linkftsfileapart, 'w', newline='')
    # 写入每条道路分开的labels
    linklbsfileapart = os.path.join(savepath,'labels.csv')
    linklbsfileapf = open(linklbsfileapart, 'w', newline='')

    # # 所有道路fts和lbs汇总
    # linkftsfileall = alllinksftscsv
    # linkftsfileallf = open(linkftsfileall, 'a+', newline='')
    #
    # linklbsfileall = alllinklbscsv
    # linklbsfileallf = open(linklbsfileall, 'a+', newline='')

    for (ft,lb) in zip(linkfts,linklbs):
        linkftsapcsv_write = csv.writer(linkftsfileapf, dialect='excel', lineterminator='\n')
        linkftsapcsv_write.writerow(ft)
        #
        linklbsapcsv_write = csv.writer(linklbsfileapf, dialect='excel', lineterminator='\n')
        linklbsapcsv_write.writerow([lb])

        # linkftsallcsv_write = csv.writer(linkftsfileallf, dialect='excel', lineterminator='\n')
        # linkftsallcsv_write.writerow(ft)
        #
        # linklbsallcsv_write = csv.writer(linklbsfileallf, dialect='excel', lineterminator='\n')
        # linklbsallcsv_write.writerow([lb])

    linkftsfileapf.close()
    linklbsfileapf.close()
    # linkftsfileallf.close()
    # linklbsfileallf.close()




#提取4个历史时期5个时间片的路况特征，各个时期分开及一个汇总的特征
def getlinkhistoryft(historyfts,speedlimit):
    timeapartfts = [] #各个时期的历史特征分开提取的特征
    # 各个时期的历史特征合并提取的特征
    alllabels = [0, 0, 0]  # 所有道路路况状态比值
    allavgspeed = 0  # 各时期特征合并后的各时期特征合并后的平均路况速度
    allavgetaspeed = 0  # 各时期特征合并后的平均eta速度
    allavgspeed_eta = 0  # 各时期特征合并后的平均路况速度/平均eta速度
    allavgspeed_limit = 0  # 各时期特征合并后的平均路况速度/link限速
    allavgeta_limit = 0  # 各时期特征合并后的平均eta速度/link限速
    allcountcarnum = 0  # 各时期特征合并后的参与路况计算的车辆总数
    alluseslice = 0
    ftnum = 0  # 可用的特征数目
    for ft in historyfts:
        ft = ft.split(' ')
        labels = [0, 0, 0]  # 道路的路况状态
        useslicenum = 0  # 非0,0,0,0时间片个数便于求平均
        avgspeed = 0  # 平均路况速度
        avgetaspeed = 0  # 平均eta速度
        # avgspeed_eta = 0  # 平均路况速度/平均eta速度
        # avgspeed_limit = 0  # 平均路况速度/link限速
        # avgeta_limit = 0  # 平均eta速度/link限速
        countcarnum = 0  # 参与路况计算的车辆总数
        for sliceft in ft:
            ft = sliceft.split(':')[1]
            if ft == '0,0,0,0':  # 去除全为0,0,0,0的特征
                continue
            ft = ft.split(',')
            useslicenum += 1
            avgspeed += float(ft[0])
            avgetaspeed += float(ft[1])
            label = int(ft[2])
            if label > 3:
                label=3
            elif label<1:
                label=1
            labels[label-1] += 1
            countcarnum += int(ft[3])
        if useslicenum == 0:
            timeapartfts.append([0,0,0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            avgspeed = avgspeed / useslicenum
            avgetaspeed = avgetaspeed / useslicenum
            avgspeed_eta = avgspeed / avgetaspeed
            avgspeed_limit = avgspeed / speedlimit
            avgeta_limit = avgetaspeed / speedlimit

            mostlabel = getbiggestindex(labels)

            timeapartfts.append([labels[0], labels[1],labels[2],mostlabel, round(labels[mostlabel - 1] / useslicenum, 3),
                                 round(avgspeed, 3), round(avgetaspeed, 3),round(avgspeed_eta, 3),
                                 round(avgspeed_limit, 3), round(avgeta_limit, 3), countcarnum])
            # 各个时期的历史特征合并提取的特征
            ftnum+=1
            alllabels[0]+=labels[0]
            alllabels[1]+=labels[1]
            alllabels[2]+=labels[2]

            allavgspeed+=avgspeed
            allavgetaspeed+=avgetaspeed
            allavgspeed_eta+=avgspeed_eta
            allavgspeed_limit+=avgspeed_limit
            allavgeta_limit+=avgeta_limit
            allcountcarnum+=countcarnum
            alluseslice+=useslicenum
    if ftnum == 0:
        alltimeft = 0#[0, 0, 0, 0, 0, 0, 0, 0, 0]#返回0，去除当前的缺失信息
        return alltimeft
        # alltimeft = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        allmostlabel = getbiggestindex(alllabels)
        alltimeft = [round(alllabels[0]/ftnum, 3),round(alllabels[1]/ftnum, 3),round(alllabels[2]/ftnum, 3),allmostlabel, round(alllabels[mostlabel - 1] / alluseslice, 3),
                     round(allavgspeed/ftnum, 3),round(allavgetaspeed/ftnum, 3),round(allavgspeed_eta/ftnum, 3),
                     round(allavgspeed_limit/ftnum, 3),round(allavgeta_limit/ftnum, 3),allcountcarnum]

    extractedhistoryfts = []
    # for ft in timeapartfts:
    #     extractedhistoryfts.append(ft)
    extractedhistoryfts.append(timeapartfts[-1])
    extractedhistoryfts.append(timeapartfts[-2])#仅使用-7-14星期的历史特征的均值，如果前一星期的历史特征缺失，则选取相邻的日期的历史特征的均值
    extractedhistoryfts.append(alltimeft)
    return extractedhistoryfts

#提取近期n个时间片路况特征
#[label1,label2,label3,avgspeed,avgetaspeed,avgspeed_eta,avgspeed_limit,avgeta_limit,countcarnum]
def getlinkrecentft(recentft,speedlimit):
    recentft = recentft.split(' ')
    labels = [0,0,0]#道路的路况状态
    useslicenum = 0#非0,0,0,0时间片个数便于求平均
    avgspeed = 0#平均路况速度
    avgetaspeed = 0#平均eta速度
    # avgspeed_eta = 0#平均路况速度/平均eta速度
    # avgspeed_limit = 0#平均路况速度/link限速
    # avgeta_limit = 0#平均eta速度/link限速
    countcarnum = 0#参与路况计算的车辆总数
    for sliceft in recentft:
        ft = sliceft.split(':')[1]
        if int(ft.split(',')[-2]) == 0:#去除权威0,0,0,0的特征
            continue
        ft = ft.split(',')
        useslicenum+=1
        avgspeed+=float(ft[0])
        avgetaspeed+=float(ft[1])
        label = int(ft[2])
        if label > 3:
            label = 3
        elif label < 1:
            label = 1
        labels[label-1]+=1
        countcarnum+=int(ft[3])
    if useslicenum == 0:
        return [0,0,0,0,0,0,0,0,0,0,0]
        # return 0#[0,0,0,0,0,0,0,0,0]#返回0，去除当前的缺失信息

    avgspeed = avgspeed/useslicenum
    avgetaspeed = avgetaspeed/useslicenum
    avgspeed_eta = avgspeed/avgetaspeed#平均路况速度/平均eta速度
    avgspeed_limit = avgspeed/speedlimit#平均路况速度/link限速
    avgeta_limit = avgetaspeed/speedlimit#平均eta速度/link限速

    mostlabel = getbiggestindex(labels)

    return [labels[0],labels[1],labels[2],mostlabel,round(labels[mostlabel-1]/useslicenum,3),
            round(avgspeed, 3),round(avgetaspeed, 3),round(avgspeed_eta, 3),round(avgspeed_limit, 3),round(avgeta_limit, 3),countcarnum]


def getbiggestindex(labels):
    mostlbnum = 0
    mostlbid = 0

    labelid = 0
    for lb in labels:
        labelid += 1
        if lb > mostlbnum:
            mostlbnum = lb
            mostlbid = labelid
    return mostlbid


##获取一条道路数据的时间片和日期等特征
def getlinktimeft_label(linkid,ft):
     headinfo = ft[0].split(' ')

     label = int(headinfo[1])#当前信息片的label
     if label==4:
         label=3
     elif label == 0:
         print(linkid+ft+' has 0!')

     current_slice_id = int(headinfo[2])#当前时间片
     future_slice_id = int(headinfo[3])#待预测时间片
     curfutdicabs = abs(current_slice_id-future_slice_id)#待预测时间片和当前时间片的差值的绝对值
     # week = ft[-2]%7#星期几
     # if week == 0:
     #     week = 7
     # if week<6:
     #    workday = 1#是否工作日
     # else:
     #     workday = 0
     return [current_slice_id,future_slice_id,curfutdicabs],label



#提取道路的属性特征，各特征的数值属性重新写入
def getlinkattr(linkattr):
    lenth = int(linkattr[1])#link的长度，以m为单位
    pathclass = int(linkattr[3])#link的的功能等级
    speedclass = int(linkattr[4])#link的速度限制等级
    LaneNum = int(linkattr[5])#link的车道数
    speedlimit = round(float(linkattr[6]), 3)#link的限速，以m/s为单位
    level = int(linkattr[7])#link的level
    width = int(linkattr[8])#link的宽度，以m为单位

    return [speedlimit,pathclass,speedclass,LaneNum,level,lenth,width]


def extractalltraindata(datadir,saveftpath):

    for link in os.listdir(datadir):
        linkid = link.split('.')[0]
        filepath = datadir+link
        linkinfo = readjsontodict(filepath)
        extractlinkfeatures(linkinfo, linkid, saveftpath)

if __name__ == '__main__':
    # linkid = '3070'
    # linkinfo = readjsontodict('E:/My competitions/didi road condition/code/processeddata/LinkinfoToDict/3070.json')
    # saveftpath = '../processeddata/extractedfeatures/'
    # extractlinkfeatures(linkinfo,linkid,saveftpath)
    # #提取训练数据特征
    # traindatadir = 'E:/My competitions/didi road condition/code/processeddata/LinkinfoToDict/'
    # saveftpath = 'E:/My competitions/didi road condition/code/dataSelfcompleted/newfeatures4/extractedfeatures/'
    # extractalltraindata(traindatadir,saveftpath)
    # #提取测试数据特征
    # testdatadir = 'E:/My competitions/didi road condition/code/processeddata/TestDataToDict/'
    # saveftpath = '../processeddata/extractedtestdatafts/'
    # extractalltraindata(testdatadir,saveftpath)

    #提取补全后的测试数据特征
    testdatadir = 'E:/My competitions/didi road condition/code/dataSelfcompleted/TestData801ToDict_ht-7-14缺失用traindata补全/'
    saveftpath = 'E:/My competitions/didi road condition/code/dataSelfcompleted/newfeatures4/extractedtestdatafts_去全零ht-7-14缺失用traindata补全/'
    extractalltraindata(testdatadir, saveftpath)
    print(count)
