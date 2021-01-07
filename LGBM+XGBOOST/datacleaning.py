import json

#读取存储为json格式的dict数据
import os

count = 0


def sum_list(L):
    li = []
    for x in L:
        if type(x) is int:
            li.append(x)
        else:
            li.append(sum_list(x))
    return sum(li)
#将获取的道路信息dict存储为json格式
def savedicttojson(dictinfo,dirpath):
    # 保存
    # f = open(filepath, 'w')
    # f.write(str(dictinfo))
    # f.close()
    #因为文件过大，将道路信息按linkid分开存储
    print("开始将道路信息按linkid写入json文件")
    filenum = 0
    for key in dictinfo.items():
        filepath = os.path.join(dirpath,key[0])
        key = [key]#元组转list，通过dict(key)转为字典存储
        with open(filepath + '.json', 'w') as outfile:
            json.dump(dict(key), outfile, ensure_ascii=False)
            outfile.write('\n')
        outfile.close()
        filenum+=1
        if filenum%1000 ==1:
            print("已写入"+str(filenum)+"个文件")
    print("道路数据整理完成！")
#统计features的label数量
def getfeaturelabels(features):
    has_4 = 0
    labels = []
    # 统计各道路features状态
    for line in features:
        feature = line.split(' ')

        # 统计当前道路的recent_feature的各个状态
        road_states = [0, 0, 0]  # 临时存储feature的状态
        for ft in feature:
            if int(ft.split(',')[2]) == 1:
                road_states[0] += 1
            elif int(ft.split(',')[2]) == 2:
                road_states[1] += 1
            elif int(ft.split(',')[2]) > 2:
                road_states[2] += 1
            # if int(ft.split(',')[2]) == 4:
            #     global count
            #     count+=1
            #     has_4 = 1
        # 根据recent_feature获取label
        tmp = sum(road_states)
        # if sum(road_states) == 0:
        #     global count
        #     count+=1
        #     print(line)
        labels.append(road_states)
    return labels,has_4

def getallftmisseddata(linkid,date,saverootpath,onedayfts,linkinfo):
    misseddatarootpath = os.path.join(saverootpath,'misseddata')
    rclabels = []
    htlabels = []
    for ft in onedayfts:
        recentlabels,has_4 = getfeaturelabels([ft[1]])
        historylabels,has_4 = getfeaturelabels(ft[2:-2])
        usehtlabels = historylabels[-1:]#只使用后面两个进行判断
        rclabels.append(recentlabels)
        htlabels.append(usehtlabels)

        shred = 1
        # if has_4:
        #     global count
        #     count+=1
        #     print(ft)
        # elif sum_list(recentlabels) <= 3:
        #     # global count
        #     count += 1
        #     # print(ft)
        # el
        # if sum_list(historylabels[0]) <= shred:
        #     global count
        #     count += 1
        #     # print(ft)
        # elif sum_list(historylabels[1]) <= shred:
        #     # global count
        #     count += 1
        #     # print(ft)
        # elif sum_list(historylabels[2]) <= shred:
        #     # global count
        #     count += 1
        #     # print(ft)
        # elif sum_list(historylabels[3]) <= shred:
        #     # global count
        #     count += 1
        #     # print(ft)


    rclabelsvalue = sum_list(rclabels)
    htlabelsvalue = sum_list(htlabels)

    if rclabelsvalue == 0 or htlabelsvalue <= 1:
        # global count
        # count += len(onedayfts)
        # print(onedayfts)
    # if htlabelsvalue <= 4*len(onedayfts):
        # 创建对应文件夹
        allftmissdir = os.path.join(os.path.join(misseddatarootpath, str(date)), linkid)
        # if not os.path.exists(allftmissdir):
        #     os.makedirs(allftmissdir)
        # savedicttojson(linkinfo,allftmissdir)
        return 1

#提取可回归的，预测futureslice具有连续性的样本数据
def getallftslcontinue(linkid,date,saverootpath,onedayfts,linkinfo,gap=3,minconnum=3):
    ftslicecontinuerootpath = os.path.join(saverootpath, 'ftslicecontinuedata')

    ftslices = []
    for ft in onedayfts:
        ft[0].split(' ')
        ftslice = int(ft[0].split(' ')[-1])
        ftslices.append(ftslice)
        currentslice = int(ft[0].split(' ')[-2])
        if abs(ftslice - currentslice) <= gap:
            global count
            count+=1

    islinkftsctue = islinkftslicecontinue(ftslices,gap,minconnum)
    if islinkftsctue:
        # global count
        # count+=len(onedayfts)
        print(linkid,' ',ftslices)


#确认当前道路是否满足ftslice具有连续性
def islinkftslicecontinue(ftslices,gap,minconnum):
    maxgapnum = 0
    if len(ftslices) >=minconnum:
        for sl in ftslices:
            gapnum = countgapnum(sl,ftslices,gap)
            if gapnum > maxgapnum:
                maxgapnum = gapnum

    if maxgapnum>=minconnum:
        # global count
        # count += len(ftslices)
        return 1
    else:
        return 0

#计算ftslice往后连续的个数：
def countgapnum(slice,slices,gap):
    connum = 0#当前ftslice与之连续的slice的个数
    slices.sort()
    for ft in slices:
        if abs(slice -ft)<= gap:
            connum +=1
    return connum
    

#提取一条道路的所有特征信息，以list形式返回
def linkftpartition(linkinfo,linkid,saverootpath):
    #获取一条道路数据的时间片和日期特征，并按不同的数据划分
    for onedayfts in linkinfo[linkid][:-1]:
        date = onedayfts[0][-2]
        #recentft和historyft(后两个缺失)全部缺省的linkid
        ismisseddata = getallftmisseddata(linkid,date,saverootpath,onedayfts,linkinfo)
        # if ismisseddata:#如果数据缺失，则进行补全操作
        #     print()
        # elif ismisseddata != 1:
        #     #提取可回归的，预测futureslice具有连续性的样本数据
        #     getallftslcontinue(linkid,date,saverootpath,onedayfts,linkinfo,gap=5,minconnum=5)



def readjsontodict(filepath):
    # 读取
    with open(filepath, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)
    return load_dict
def extractalltraindata(datadir,datapartitionrootpath):

    for link in os.listdir(datadir):
        linkid = link.split('.')[0]
        filepath = datadir+link
        linkinfo = readjsontodict(filepath)
        linkftpartition(linkinfo, linkid, datapartitionrootpath)
    print(count)




#主控函数
if __name__ == "__main__":
    traindatadir = 'E:/My competitions/didi road condition/code/dataSelfcompleted/TestData801ToDict/'
    datapartitionrootpath = 'E:/My competitions/didi road condition/code/datapartition/'
    extractalltraindata(traindatadir, datapartitionrootpath)