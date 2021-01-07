import json
import os
#做了所有数据集的整理，添加日期特征，并按linkid将信息按时间片排序，并分开存储为json格式的dict

# attr.txt
# 读取所有道路linkid及道路属性
def getlinkattr(linkattrpath):
    f = open(linkattrpath, "r")
    lines = f.readlines()
    alllinkinfo = {}  # 使用字典存储所有道路的数据{key:linkid,value:[attr,feature1,feature2...]}
    # linkid length direction pathclass speedclass LaneNum speedlimit level width
    # 因attr文件格式简洁，不需多次循环
    for line in lines:
        attrinfo = line.strip('\n').split('\t')  # 提取道路信息，list分割
        linkid = attrinfo[0]
        alllinkinfo.setdefault(linkid, []).append(attrinfo)  # 使用list作为dict值
    f.close()
    return alllinkinfo

#删除一条信息中的'0,0,0,0'的数据
def removeallzerofeature(info):
    for ft in info[1:-2]:
        if ft.split(' ')[0].split(':')[1] == '0,0,0,0':
            info.remove(ft)
    return info

#读取单个训练数据文件内容，并按当天的时间片排序后返回
def gettraindatalinkinfo(trainfilepath,filedate):
    linkinfo = {}  # 单个训练数据中的所有道路的样本信息
    # 读取训练数据文件内容
    f = open(trainfilepath, "r")
    lines = f.readlines()
    for line in lines:
        info = line.strip('\n')
        info = info.split(';')
        # removeallzerofeature(info)
        headinfo = info[0].split(' ')#获取第一部分道路的信息
        linkid = headinfo[0]
        link_timeslice = int(headinfo[2])  # 获取时间片后，便于初步按当天时间片对道路状态排序
        info.append(filedate)  # 在初始训练数据中加上 日期 特征
        info.append(link_timeslice)#在道路的一条训练数据最后加入时间片便于排序
        if linkid in linkinfo.keys():  # 如果linkid已经有数据记录
            linkinfo[linkid].append(info)
        else:  # 如果linkid首次出现
            linkinfo.setdefault(linkid, []).append(info)
    # 获取完道路的一日数据后按时间片排序
    linkinfo = getresortedtrafficinfo(linkinfo)
    f.close()
    return linkinfo

#输出按当天时间片重新排序后的道路信息
def getresortedtrafficinfo(linkinfo):
    #对获取的道路信息，按当前时间片的顺序重新排序，并按照原格式重新存储-去掉(,recentsliceid)
    for key in linkinfo:
        linkinfo[key].sort(key=takeSecond)
    return linkinfo

# 获取列表的第三个元素
def takeSecond(elem):
    return elem[-1]

#加载数据集，读取指定文件夹里的训练数据集
def loaddata(linkattrpath,traindatadir):
    #返回的数据格式:
    #linkinfo:
    # {key:linkid,
    # value:[
    # 第1天的数据：
    #        [['linkid label current_slice_id future_slice_id','recent_feature','history_feature',filedate,current_slice_id],
    #        ['linkid label current_slice_id future_slice_id','recent_feature','history_feature',filedate,current_slice_id],
    #        ...],
    # 第2天的数据：
    #        [['linkid label current_slice_id future_slice_id','recent_feature','history_feature',filedate,current_slice_id],
    #        ['linkid label current_slice_id future_slice_id','recent_feature','history_feature',filedate,current_slice_id],
    #        ...],
    #        ...
    # 道路属性：
    #        [linkid,length,direction,pathclass,speedclass,LaneNum,speedlimit,level,width]#attr
    #       ]}

#因给定的训练数据中并不是所有道路都有样本数据，故为了加快处理速度，仅将训练集中有的道路的信息进行提取
    alllinkattr = getlinkattr(linkattrpath)# 读取所有道路linkid及道路属性
    #traindata.txt
    linkinfo = {}#训练数据中的所有道路的样本信息
    traindirfiles = os.listdir(traindatadir)#训练数据文件夹下所有文件名称
    #读取所有训练数据的文件
    for trainfile in traindirfiles:
        trainfilepath = traindatadir+'/'+trainfile
        filedate = int(trainfile.split('.')[0].split('907')[1])#记录训练数据对应的日期，保留 节假日-工作日-休息日 的特征
        print(filedate)
        # 读取单个训练数据文件内容，并按当天的时间片排序
        onedaylinkinfo = gettraindatalinkinfo(trainfilepath,filedate)
        #将处理好并排序后的一天内的道路信息增加到总的linkinfo中：
        adddict(linkinfo,onedaylinkinfo)
        del onedaylinkinfo
        print('linkinfo '+trainfile+' added')
    #添加道路的属性信息
    for key in linkinfo:
        linkinfo[key].append(alllinkattr[key])
    print(len(linkinfo.keys()))
    return linkinfo

#在一个字典dict中添加另一个字典adddict,数据格式为{key:linkid,value:[[feature1,feature2,...],[feature1,feature2,...]]}
def adddict(dict,adddict):
    for key in adddict:
        if key in dict.keys():# 如果该键值已存在于dict中
            dict[key].append(adddict[key])
        else:# 如果该键值在dict中没有记录
            dict.setdefault(key, []).append(adddict[key])

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
        filepath = dirpath + key[0]
        key = [key]#元组转list，通过dict(key)转为字典存储
        with open(filepath + '.json', 'w') as outfile:
            json.dump(dict(key), outfile, ensure_ascii=False)
            outfile.write('\n')
        outfile.close()
        filenum+=1
        if filenum%1000 ==1:
            print("已写入"+str(filenum)+"个文件")
    print("道路数据整理完成！")
#读取存储为json格式的dict数据
def readjsontodict(filepath):
    # 读取
    with open(filepath, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)
    return load_dict


if __name__=="__main__":
    linkinfo = loaddata('E:/My competitions/didi road condition/traindata/attr.txt',
                        'E:/My competitions/didi road condition/traindata/traffic/')
    # # 保存获取的linkinfo
    savedicttojson(linkinfo, '../processeddata/LinkinfoToDict/')
    # linkinfo = loaddata('E:/My competitions/didi road condition/traindata/attr.txt',
    #                     'E:/My competitions/didi road condition/test/test/')
    # #保存获取的linkinfo
    # savedicttojson(linkinfo,'../processeddata/TestDataToDict/')
    #读取保存为json格式的linkinfo到dict
    # linkinfo = readjsontodict('../processeddata/AllLinkinfoToDict.txt')

    # #将730的数据作为测试集提取
    # linkinfo = loaddata('E:/My competitions/didi road condition/traindata/attr.txt',
    #                     'E:/My competitions/didi road condition/traindata/20190801_testdata_update/')
    # # 保存获取的linkinfo
    # savedicttojson(linkinfo, '../processeddata/TestData801ToDict/')

    #数据补全后的测试集
    # linkinfo = loaddata('E:/My competitions/didi road condition/traindata/attr.txt',
    #                     'E:/My competitions/didi road condition/code/dataSelfcompleted/ht-7-14缺失用traindata补全/')
    # # 保存获取的linkinfo
    # savedicttojson(linkinfo, 'E:/My competitions/didi road condition/code/dataSelfcompleted/TestData801ToDict_ht-7-14缺失用traindata补全/')