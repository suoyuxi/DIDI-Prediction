
import os
#读取单个训练数据文件内容，并按当天的时间片排序后返回
from xgboostclassify.datacleaning import sum_list, getfeaturelabels, readjsontodict

count = 0

#自补全：$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def selfcomplete(linksinfo):
    for linkid in linksinfo:
        for currentslicedict in linksinfo[linkid]:
            linksinfo[linkid][currentslicedict]['currentslice'] = rowselfcomplete_currentft(linksinfo[linkid][currentslicedict]['currentslice'])
            for futureslice in linksinfo[linkid][currentslicedict]:
                if futureslice == 'currentslice':
                    continue
                linksinfo[linkid][currentslicedict][futureslice] = Rowsselfcomplete_historyft(linksinfo[linkid][currentslicedict][futureslice])
                # linksinfo[linkid][currentslicedict][futureslice] = Front_Nextlinecomplete_htft(linksinfo[linkid][currentslicedict][futureslice])
    return linksinfo
#historyfts按前后关系补全，一行缺失，按+7行补全，+7行缺失再按-7行补全：&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def Front_Nextlinecomplete_htft(historyfts):
    for htft in historyfts:
        isall0 = 0
        for ft in historyfts[htft]:
            if ft == 0:
                isall0 += 1
        if isall0 >= 4:#仅对缺失的行进行补全

            # print('补全:--------------------')
            # print('前：', historyfts)

            historyfts[htft] = findFNlinecomplete_historyft(htft,historyfts[htft],historyfts)

            # print('后：', historyfts)
            # global count
            # count += 1

    return historyfts
#查找满足条件的补全行对当前行进行补全
def findFNlinecomplete_historyft(all0htftid,all0htft,historyfts):
    all0htftidV = int(all0htftid.split('ht')[1])
    if all0htftidV == 4:
        # frontline = historyfts[('ht'+str(all0htftidV+1))]
        nextline = historyfts[('ht'+str(all0htftidV-1))]
        if iscompleted(nextline):
            return nextline
        else:
            return all0htft
    elif all0htftidV == 1:
        frontline = historyfts[('ht' + str(all0htftidV + 1))]
        if iscompleted(frontline):
            return frontline
        else:
            return all0htft
    else:
        frontline = historyfts[('ht' + str(all0htftidV + 1))]
        nextline = historyfts[('ht' + str(all0htftidV - 1))]
        if iscompleted(frontline):
            return frontline
        elif iscompleted(nextline):
            return nextline
        else:
            return all0htft
#判断historyft是否全为0
def iscompleted(historyft):
    isall0 = 0
    for ht in historyft:
        if ht == 0:
            isall0+=1
    if isall0 == len(historyft):
        return 0
    else:
        return 1
#historyfts按前后关系补全，一行缺失，按+7行补全，+7行缺失再按-7行补全：&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


#historyfts按行补全：-------------------------------------------------------------------------------------------------------
#futurefts,历史特征同行自补全
def Rowsselfcomplete_historyft(historyfts):
    #单行补全，全为0，或者仅有1个ft则不操作，否则找左右相邻的值进行补全
    for htft in historyfts:
        historyfts[htft] = rowselfcomplete_historyft(historyfts[htft])
    return historyfts
#历史ft单行补全，全为0，或者仅有1个ft则不操作，否则找左右相邻的值进行补全
def rowselfcomplete_historyft(historyft):
    isall0 = 0
    for ft in historyft:
        if ft == 0:
            isall0 += 1
    if isall0 >= 3 or isall0 == 0:
        return historyft
    completedfts = []
    ftid = -1
    for ft in historyft:
        ftid += 1
        if ft == 0:
            ft = rowfindhistorypltv(ftid, historyft)
        completedfts.append(ft)

    # print('补全:--------------------')
    # print('前：', historyft)
    # print('后：', completedfts)
    # global count
    # count += 1

    return completedfts
#在一行中查找用于补全historyfts的值
def rowfindhistorypltv(ftid,historyft):
    rclosestftid = ftid
    rfind = 0
    lclosestftid = ftid
    lfind = 0
    if rclosestftid != 4:
        #从ftid开始往右边搜索
        for ft in historyft[ftid+1:]:
            rclosestftid += 1
            if ft != 0:
                rfind=1
                break

    else:
        rclosestftid = 100
    if rfind == 0:
        rclosestftid = 100
    if lclosestftid != 0:
        #从ftid开始往左边搜索
        for ft in historyft[0:ftid][::-1]:
            lclosestftid -= 1
            if ft != 0:
                lfind=1
                break
    else:
        lclosestftid = -100
    if lfind == 0:
        lclosestftid = -100
    if ftid-lclosestftid <= rclosestftid-ftid:
        lclosestft = str(int(historyft[lclosestftid].split(':')[0]) + ftid - lclosestftid) + ':' + historyft[lclosestftid].split(':')[1]
        # lclosestft = ":".join([lclosestft.split(':')[0], '0,0,0,0'])
        return lclosestft
    else:
        rclosestft = str(int(historyft[rclosestftid].split(':')[0]) - rclosestftid + ftid) + ':' + historyft[rclosestftid].split(':')[1]
        # rclosestft = ":".join([rclosestft.split(':')[0], '0,0,0,0'])
        return rclosestft
#historyfts按行补全：-------------------------------------------------------------------------------------------------------


#currentfts按行补全：*******************************************************************************************************
#补全currentfts，全为0则不操作，否则找左右相邻的值进行补全
def rowselfcomplete_currentft(currentsliceinfo):
    isall0 = 0
    for ft in currentsliceinfo:
        if ft == 0:
            isall0+=1
    if isall0 >= 3 or isall0 == 0:
        return currentsliceinfo
    completedfts = []
    ftid = -1
    for ft in currentsliceinfo:
        ftid += 1
        if ft == 0:
            ft = rowfindcurrentpltv(ftid,currentsliceinfo)
        completedfts.append(ft)

    # print('补全:--------------------')
    # print('前：',currentsliceinfo)
    # print('后：',completedfts)
    # global count
    # count+=1

    return completedfts
#在一行中查找用于补全currentfts的值
def rowfindcurrentpltv(ftid,currentsliceinfo):
    rclosestftid = ftid
    rfind = 0
    lclosestftid = ftid
    lfind = 0
    if rclosestftid != 4:
        #从ftid开始往右边搜索
        for ft in currentsliceinfo[ftid+1:]:
            rclosestftid += 1
            if ft != 0:
                rfind=1
                break
    else:
        rclosestftid = 100
    if rfind == 0:
        rclosestftid = 100
    if lclosestftid != 0:
        #从ftid开始往左边搜索
        for ft in currentsliceinfo[0:ftid][::-1]:
            lclosestftid -= 1
            if ft != 0:
                lfind=1
                break
    else:
        lclosestftid = -100
    if lfind == 0:
        lclosestftid = -100
    if ftid-lclosestftid < rclosestftid-ftid:
        lclosestft = str(int(currentsliceinfo[lclosestftid].split(':')[0]) + ftid - lclosestftid) + ':' + currentsliceinfo[lclosestftid].split(':')[1]
        # lclosestft = ":".join([lclosestft.split(':')[0],'0,0,0,0'])
        return lclosestft
    else:
        rclosestft = str(int(currentsliceinfo[rclosestftid].split(':')[0]) - rclosestftid+ftid) + ':' + currentsliceinfo[rclosestftid].split(':')[1]
        # rclosestft = ":".join([rclosestft.split(':')[0], '0,0,0,0'])
        return rclosestft
#currentfts按行补全：*******************************************************************************************************


#提取每一行的fts，将缺省的ft置为0
def extractftexpzero(features):
    expfts = []
    for ft in features.split(' '):
        if ft.split(',')[-2] == '0':
            expfts.append(0)
        else:
            expfts.append(ft)
    return expfts
#对于测试数据的提取结构：
#{linkid:
#   {currentslice:
#       {'currentslice':[currentft1,currentft2,currentft3,currentft4,currentft5]}
#       {ftslice1:[futureft1,futureft2,futureft3,futureft4,futureft5]}
#       {ftslice2:[futureft1,futureft2,futureft3,futureft4,futureft5]}
#   }
# }
def gettestdatatodict(testfilepath):
    linkinfo = {}  # 单个训练数据中的所有道路的样本信息
    # 读取训练数据文件内容
    f = open(testfilepath, "r")
    lines = f.readlines()
    for line in lines:
        info = line.strip('\n')
        info = info.split(';')
        headinfo = info[0].split(' ')  # 获取头部部分道路的信息
        linkid = int(headinfo[0])
        currentslice = int(headinfo[2])
        futureslice = int(headinfo[3])

        # 如果linkid是第一次出现
        if linkid not in linkinfo.keys():
            linkinfo.setdefault(linkid, {})
        # 如果currentslice是第一次出现
        if currentslice not in linkinfo[linkid].keys():
            linkinfo[linkid].setdefault(currentslice, {})
        if 'currentslice' not in linkinfo[linkid][currentslice].keys():
            linkinfo[linkid][currentslice].setdefault('currentslice', [])
        linkinfo[linkid][currentslice]['currentslice']=extractftexpzero(info[1])
        if futureslice not in linkinfo[linkid][currentslice].keys():
            linkinfo[linkid][currentslice].setdefault(futureslice, {})
            linkinfo[linkid][currentslice][futureslice].setdefault('ht1', []).extend(extractftexpzero(info[2]))
            linkinfo[linkid][currentslice][futureslice].setdefault('ht2', []).extend(extractftexpzero(info[3]))
            linkinfo[linkid][currentslice][futureslice].setdefault('ht3', []).extend(extractftexpzero(info[4]))
            linkinfo[linkid][currentslice][futureslice].setdefault('ht4', []).extend(extractftexpzero(info[5]))

    f.close()
    return linkinfo


#将dict格式的links信息写入txt文件中
def writedictlinksinfointxt(linksinfo,selfcompletedfilepath):
    # 写入txt结果
    # 创建txt文件
    with open(selfcompletedfilepath, "w") as f:  # 设置文件对象
        for linkid in linksinfo:
            for currentslicedict in linksinfo[linkid]:
                currentfts = getcurrentfts(linksinfo[linkid][currentslicedict]['currentslice'],currentslicedict)
                for futureslice in linksinfo[linkid][currentslicedict]:
                    if futureslice == 'currentslice':
                        continue
                    headinfo = " ".join([str(linkid),'-1',str(currentslicedict),str(futureslice)])
                    historyfts = gethistoryfts(linksinfo[linkid][currentslicedict][futureslice],futureslice)
                    onesample = ";".join([headinfo,currentfts,historyfts])+'\n'
                    f.writelines(onesample)
    f.close()

def getcurrentfts(currentfts,currentsliceid):
    result = []
    ftsliceid = currentsliceid-len(currentfts)
    for ft in currentfts:
        ftsliceid+=1
        if ft == 0:
            result.append(str(ftsliceid)+':'+'0,0,0,0')
        else:
            result.append(ft)
    return " ".join(result)
def gethistoryfts(historyfts,futuresliceid):
    result = []
    for ft in historyfts:
        htft = gethtft(historyfts[ft],futuresliceid)
        result.append(htft)
    return ";".join(result)

def gethtft(historyfts,futuresliceid):
    result = []
    ftsliceid = futuresliceid
    for ft in historyfts:
        ftsliceid+=1
        if ft == 0:
            result.append(str(ftsliceid)+':'+'0,0,0,0')
        else:
            result.append(ft)
    return " ".join(result)
#自补全：$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


#训练数据补全￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥
def traindatacomplete(linksinfo):
    for linkid in linksinfo:
        traindatasliceinfo = gettraindatainfo(linkid)
        if traindatasliceinfo == 0:
            continue
        for currentslicedict in linksinfo[linkid]:
            # linksinfo[linkid][currentslicedict]['currentslice'] = traindatacomplete_currentft(linksinfo[linkid][currentslicedict]['currentslice'])
            for futureslice in linksinfo[linkid][currentslicedict]:
                if futureslice == 'currentslice':
                    continue
                headinfo = [linkid,currentslicedict,futureslice]

                linksinfo[linkid][currentslicedict][futureslice] = RowsTrainDatacomplete_historyft(linksinfo[linkid][currentslicedict][futureslice],headinfo,traindatasliceinfo)
                # linksinfo[linkid][currentslicedict][futureslice] = Front_Nextlinecomplete_htft(linksinfo[linkid][currentslicedict][futureslice])
    return linksinfo

def RowsTrainDatacomplete_historyft(historyfts,headinfo,traindatasliceinfo):
        # 仅补全-7和-14的缺失值
        historyfts['ht4'] = traindatacomplete_historyft(historyfts['ht4'],headinfo,traindatasliceinfo)
        historyfts['ht3'] = traindatacomplete_historyft(historyfts['ht3'],headinfo,traindatasliceinfo)

        return historyfts

def traindatacomplete_historyft(historyft,headinfo,traindatasliceinfo):
    isall0 = 0
    for ft in historyft:
        if ft == 0:
            isall0 += 1
    if isall0 <= 1:
        return historyft
    # if isall0 >=2:
    #     print()
    completedfts = []
    ftid = -1
    for ft in historyft:
        ftid += 1
        if ft == 0:
            global count
            count+=1
            ft = FindhtftInTraindata(ftid, headinfo,traindatasliceinfo)
        completedfts.append(ft)

    # print('补全:--------------------')
    # print('前：', historyft)
    # print('后：', completedfts)
    # global count
    # count += 1

    return completedfts

#从训练数据中对缺失的数据进行补全
def FindhtftInTraindata(ftid,headinfo,traindatasliceinfo):
    linkid = str(headinfo[0])
    # currentslice = int(headinfo[1])
    futureslice = int(headinfo[2])
    ftsliceid = futureslice+ftid
    # for dateinfo in traindatasliceinfo[linkid]:
    #     if ftsliceid not in dateinfo.keys():

    date = list(traindatasliceinfo[linkid].keys())
    date.sort(reverse=True)
    find = 0
    for d in date:
        try:
            return traindatasliceinfo[linkid][d][ftsliceid]
        except:
            continue
    print(linkid, ' not find ', ftsliceid)
    # global count
    # count+=1
    if find == 0:
        return str(ftsliceid)+':'+'0,0,0,0'


#按sliceid和日期读取训练数据
def gettraindatainfo(linkid):
    traindatafile = str(linkid)+'.json'
    traindatarootpath = 'E:/My competitions/didi road condition/code/processeddata/LinkinfoToDict'
    traindatafilepath = os.path.join(traindatarootpath,traindatafile)

    date = [30,29,26,25,24,23,22,19,18,17,16,15,12,11,10,9,8,5,4,3,2,1]
    traindatalinkinfo = gettraindatabysliceid_date(linkid,traindatafilepath,date)


    return traindatalinkinfo

def gettraindatabysliceid_date(linkid,traindatafilepath,date):
    linkid = str(linkid)
    linksinfobysliceid_date = {}
    linksinfobysliceid_date.setdefault(linkid, {})
    try:
        linksdictinfo = readjsontodict(traindatafilepath)
    except:
        print(traindatafilepath,' no exist!')
        return 0
    print()

    for dateinfo in linksdictinfo[linkid]:
        tmpdate = dateinfo[0][-2]
        if tmpdate not in date:
            continue
        else:
            linksinfobysliceid_date[linkid].setdefault(tmpdate, {})
            for sample in dateinfo:
                for ft in sample[1:6]:
                    slicesinfo = ft.split(' ')
                    for slice in slicesinfo:
                        sliceid = int(slice.split(':')[0])
                        label = slice.split(':')[1].split(',')[-2]
                        if int(label) != 0 and sliceid not in linksinfobysliceid_date[linkid][tmpdate].keys():
                            linksinfobysliceid_date[linkid][tmpdate].setdefault(sliceid, {})
                            linksinfobysliceid_date[linkid][tmpdate][sliceid] = slice
    return linksinfobysliceid_date
#训练数据补全￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥


#主控函数
if __name__ == "__main__":
    testdatadir = 'E:/My competitions/didi road condition/traindata/20190801_testdata_update/20190801.txt'
    testdatatodict = gettestdatatodict(testdatadir)

    #缺失值自补全
    #①同行缺失自补全
    linksinfo = selfcomplete(testdatatodict)
    # selfcompletedfilepath = 'E:/My competitions/didi road condition/code/dataSelfcompleted/每行未缺失值小于等于3-自补全/20190801_selfcomplete.txt'
    # #将linksinfo写入txt文件
    # writedictlinksinfointxt(linksinfo,selfcompletedfilepath)

    #使用训练数据对历史特征的缺失值进行补全
    linksinfo = traindatacomplete(linksinfo)

    # historydatadir = 'E:/My competitions/didi road condition/traindata/traffic'
    # historydata = ['20190704.txt','20190711.txt','20190718.txt','20190725.txt']
    # historydata = ['20190725.txt']
    # for htdata in historydata:
    #     htdatapath = os.path.join(historydatadir,htdata)
    #     htdatatodict = loadtxtdatatodict(htdatapath)
    #
    # traindatadir = 'E:/My competitions/didi road condition/code/processeddata/TestData801ToDict/'
    # datapartitionrootpath = 'E:/My competitions/didi road condition/code/datapartition/'
    # completemisseddata(traindatadir, htdatatodict)
    # traindatacompletedfilepath = 'E:/My competitions/didi road condition/code/dataSelfcompleted/ht-7-14缺失用traindata补全/20190801_traindatacomplete.txt'
    # writedictlinksinfointxt(linksinfo, traindatacompletedfilepath)
    print(count)
