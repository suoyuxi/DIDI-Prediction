#统计features的label数量
def getrcfeaturelabel(rcfeature):
    # 统计recent_features状态
    feature = rcfeature.split(' ')

    # 统计当前道路的recent_feature的各个状态
    road_states = [0, 0, 0]  # 临时存储feature的状态
    isall0 = 0
    for ft in feature:
        if ft.split(':')[1] == '0,0,0,0':
            isall0+=1
            continue
        elif int(ft.split(':')[1].split(',')[-2])==0:
            isall0+=1
            continue
        if int(ft.split(',')[2]) == 1:
            road_states[0] += 1
        elif int(ft.split(',')[2]) == 2:
            road_states[1] += 1
        elif int(ft.split(',')[2]) > 2:
            road_states[2] += 1
    # 根据recent_feature获取label
    # tmp = sum(road_states)
    if isall0 == len(feature):#如果没有currentft
        return 100
    if sum(road_states) == 0:
        label = 0
        return label
    else:
        label = road_states.index(max(road_states)) + 1
        return label



#读取原始测试数据集的csv,并以dict格式返回，用于提取更好的label
# def gettestdata(filepath):
#     linkinfo = {}# {linkid:{futureslice:[[currentslice,label],[currentslice,label]]}}
#     # 读取训练数据文件内容
#     f = open(filepath, "r")
#     lines = f.readlines()
#     for line in lines:
#         infoline = line.strip('\n')
#         infoline = infoline.split(';')
#
#         headinfo=infoline[0].split(' ')
#         recent_features=infoline[1]
#
#         linkid = int(float(headinfo[0]))
#         currentslice = int(float(headinfo[2]))
#         futuresilce = int(float(headinfo[3]))
#         rclabel = int(float(getrcfeaturelabel(recent_features)))
#         if rclabel == 100:
#             print(line)
#             continue
#         #如果linkid是第一次出现
#         if linkid not in linkinfo.keys():
#             linkinfo.setdefault(linkid, {})
#         #如果futureslice是第一次出现
#         if futuresilce not in linkinfo[linkid].keys():
#             linkinfo[linkid].setdefault(futuresilce, [])
#         linkinfo[linkid][futuresilce].append([currentslice,rclabel])
#     f.close()
#     return linkinfo

#读取提交的测试数据集的csv,并以dict格式返回
def getsubmitdata(filepath):
    linkinfo = {}# {linkid:{futureslice:[[currentslice,label],[currentslice,label]]}}
    # 读取训练数据文件内容
    f = open(filepath, "r")
    lines = f.readlines()
    for line in lines[1:]:
        info = line.strip('\n')
        info = info.split(',')
        linkid = int(float(info[0]))
        currentslice = int(float(info[1]))
        futuresilce = int(float(info[2]))
        label = int(float(info[3]))
        #如果linkid是第一次出现
        if linkid not in linkinfo.keys():
            linkinfo.setdefault(linkid, {})
        #如果futureslice是第一次出现
        if futuresilce not in linkinfo[linkid].keys():
            linkinfo[linkid].setdefault(futuresilce, [])
        linkinfo[linkid][futuresilce].append([currentslice,label])
    f.close()
    return linkinfo

# #根据最近的currentslice修改同一linkid-futureslice下的label
# def modifylabelbycurlb_test1(linkinfo,rclinkinfo):
#     count = 0
#     #逐linkid
#     for linkid in linkinfo.keys():
#         #逐futureslice
#         for futureslice in linkinfo[linkid].keys():
#             #按currentslice从小到大排序
#             linkinfo[linkid][futureslice].sort(key=takeFirst)
#             rclinkinfo[linkid][futureslice].sort(key=takeFirst)
#
#             prelabel = linkinfo[linkid][futureslice][-1][1]
#             rclabel = rclinkinfo[linkid][futureslice][-1][1]
#             futureslicev = int(futureslice)
#             currentslicev = int(linkinfo[linkid][futureslice][-1][0])#注意是对于一个futureslice的多个currentslice，仅选取最后一个离futureslice最近的currentslice作为所有的结果
#
#             if rclabel == 0:
#                 rclabel = 0#可能出现0,0,0,0的情况
#                 # print(0)
#             rc_or_pre = 0#指示使用prelabel和rclabel的情况
#             # 如果futureslice 和 currentslice 相差较小比如5以内，则使用 rclabel，否则使用模型的预测label
#             if prelabel == rclabel:
#                 label = prelabel
#             elif abs(futureslicev - currentslicev) <= 3:#可以测试5-10
#                 # rc_or_pre = 1
#                 label = rclabel
#             else:
#                 label = prelabel
#             #重新对标注结果进行时间穿越修改
#             for lt in linkinfo[linkid][futureslice]:
#                 if rc_or_pre==1:
#                     count += 1
#                     print(futureslicev, ' ', lt[0], ' ', rclabel, ' ', prelabel)
#                 lt[1] = label
#
#             #测试用
#             # if linkid == 2955 and futureslice==250:
#             #     print()
#     print(count)
#     return linkinfo
# 获取列表的第一个元素
def takeFirst(elem):
    return elem[0]

#写入csv结果
def writeincsvresult(resultinfo,resultfilepath):
    # resultinfo.sort(key=takeFirst)
#创建csv文件
    import csv
    #写入csv文件并去除\r\n->\n
    resultcsvFile = open(resultfilepath,'w',newline='')
    csv_write = csv.writer(resultcsvFile,dialect='excel',lineterminator='\n')
    csv_head = ['link','current_slice_id','future_slice_id','label']
    csv_write.writerow(csv_head)

    #重新排序 id 当前时间片 待预测时间片 label后 写入result
    for info in resultinfo:
        csv_write.writerow(info)
    resultcsvFile.close()

#对按时间穿越特征重新修改后的测试数据，重新写入csv文件中
def writetimecrosslinkinfo(linkinfo,resultfilepath):
    result = []
    #按linkid和futureslice整理成list后写入
    for linkid in linkinfo.keys():
        for futureslice in linkinfo[linkid].keys():
            for lt in linkinfo[linkid][futureslice]:
                tmp = [linkid,lt[0],futureslice,lt[1]]
                result.append(tmp)
    writeincsvresult(result,resultfilepath)


##读取提交的测试数据集的csv,并以dict格式仅返回{linkid:[[currentslice,label],[currentslice,label],[currentslice,label],[currentslice,label]]}
def gettestdataonlyrclb(filepath):
    linkinfo_onlyrclb = {}  # {linkid:[[currentslice,label],[currentslice,label],[currentslice,label],[currentslice,label]]}
    # 读取训练数据文件内容
    f = open(filepath, "r")
    lines = f.readlines()
    for line in lines:
        infoline = line.strip('\n')
        infoline = infoline.split(';')

        headinfo = infoline[0].split(' ')
        recent_features = infoline[1]

        linkid = int(float(headinfo[0]))
        currentslice = int(float(headinfo[2]))
        # futuresilce = int(float(headinfo[3]))
        rclabel = int(float(getrcfeaturelabel(recent_features)))
        # if rclabel == 100:
        #     print(line)
        #     continue
        # 如果linkid是第一次出现
        if linkid not in linkinfo_onlyrclb.keys():
            linkinfo_onlyrclb.setdefault(linkid, [])
        linkinfo_onlyrclb[linkid].append([currentslice, rclabel])
    f.close()

    # #对linkid下的[currentslice,label]按currentslice从小到大排序
    # for linkid in linkinfo_onlyrclb.keys():
    #     linkinfo_onlyrclb[linkid].sort
    return linkinfo_onlyrclb

#根据一条道路下的全局最近的currentslice修改同一linkid-futureslice下的label
def modifylabelbycurlb_onlyrclb(linkinfo,rclinkinfo_onlyrclb,shred=3,isfuture=0):
    count = 0
    #逐linkid
    for linkid in linkinfo.keys():
        #逐futureslice
        for futureslice in linkinfo[linkid].keys():
            #按currentslice从小到大排序
            linkinfo[linkid][futureslice].sort(key=takeFirst)
            rclinkinfo_onlyrclb[linkid].sort(key=takeFirst)

            prelabel = linkinfo[linkid][futureslice][-1][1]
            # rclabel = rclinkinfo[linkid][futureslice][-1][1]
            futureslicev = int(futureslice)
            # currentslicev = int(linkinfo[linkid][futureslice][-1][0])#注意是对于一个futureslice的多个currentslice，仅选取最后一个离futureslice最近的currentslice作为所有的结果
            if isfuture == 1:
                rclabel,currentslicev = searchmostrecentslicelb(futureslicev,rclinkinfo_onlyrclb[linkid],isfuture=1)
            else:
                rclabel,currentslicev = searchmostrecentslicelb(futureslicev,rclinkinfo_onlyrclb[linkid])#在linkid的全局搜索距离futureslice最近的currentslice及其label

            if rclabel == 0:
                rclabel = 1#可能出现0,0,0,0的情况
                print(0)
            elif rclabel == 100:#过滤掉全是0的数据
                rclabel = prelabel
            rc_or_pre = 0#指示使用prelabel和rclabel的情况
            # 如果futureslice 和 currentslice 相差较小比如5以内，则使用 rclabel，否则使用模型的预测label
            if prelabel == rclabel:
                label = prelabel
                # rc_or_pre = 1
            elif abs(futureslicev - currentslicev) <= shred:#可以测试5-10
                rc_or_pre = 1
                label = rclabel
            else:
                label = prelabel
            #重新对标注结果进行时间穿越修改
            for lt in linkinfo[linkid][futureslice]:
                if rc_or_pre==1:
                    count += 1
                    print(futureslicev, ' ', lt[0],' ',currentslicev, ' ', rclabel, ' ', prelabel,' ',label)
                lt[1] = label

            #测试用
            # if linkid == 2955 and futureslice==250:
            #     print()
    print(count)
    return linkinfo

##在linkid的全局搜索距离futureslice最近的currentslice及其label
def searchmostrecentslicelb(futureslice,linkcurrentsliceinfo,isfuture=0):
    linksliceinfo = []
    if isfuture == 1:
        for info in linkcurrentsliceinfo:
            if info[0]==futureslice:
                continue
            linksliceinfo.append(info)
    else:
        linksliceinfo = linkcurrentsliceinfo
    linkcurrentslice_ft = []

    for slice in linksliceinfo:
        linkcurrentslice_ft.append([abs(slice[0]-futureslice),slice[1]])
    linkcurrentslice_ft.sort(key=takeFirst)
    if len(linkcurrentslice_ft)==0:
        return 1,futureslice+100
    # try/:
    label=linkcurrentslice_ft[0][1]
    slicev = linkcurrentslice_ft[0][0] + futureslice
# /    except:
#         print()
    return label,slicev

##对于通过recentft的label修正过的预测结果，使用自身的futureslice，进行再次的自校正
#获取修正后的预测结果的linkid:[futureslice,label],[futureslice,label],[futureslice,label]
def getmodifieddataonlyftlb(modifiedfilepath):
    linkinfo_onlyftlb = {}  # {linkid:[futureslice,label],[futureslice,label],[futureslice,label]}
    # 读取训练数据文件内容
    f = open(modifiedfilepath, "r")
    lines = f.readlines()
    for line in lines[1:]:
        infoline = line.strip('\n')
        infoline = infoline.split(',')
        linkid = int(float(infoline[0]))
        futuresilce = int(float(infoline[2]))
        ftlabel = int(float(infoline[3]))
        # 如果linkid是第一次出现
        if linkid not in linkinfo_onlyftlb.keys():
            linkinfo_onlyftlb.setdefault(linkid, [])
        linkinfo_onlyftlb[linkid].append([futuresilce, ftlabel])
    f.close()

    # #对linkid下的[currentslice,label]按currentslice从小到大排序
    # for linkid in linkinfo_onlyrclb.keys():
    #     linkinfo_onlyrclb[linkid].sort
    return linkinfo_onlyftlb

if __name__=='__main__':

    submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/lgboostresult_lgb_new_selfcompleted.csv'
    linkinfo = getsubmitdata(submitdatafilepath)
    testdatafilepath = 'E:/My competitions/didi road condition/traindata/20190801_testdata_update/20190801.txt'
    # linkinfo_testd = gettestdata(testdatafilepath)
    linkinfo_test_onlyrclb = gettestdataonlyrclb(testdatafilepath)
    resultfilepath = 'E:/My competitions/didi road condition/code/predictresult/lgboostresult_lgb_new_cross.csv'
    modifiedlinkinfo = modifylabelbycurlb_onlyrclb(linkinfo,linkinfo_test_onlyrclb,5)
    # modifiedlinkinfo = modifylabelbycurlb(linkinfo,linkinfo_testd)
    writetimecrosslinkinfo(modifiedlinkinfo,resultfilepath)

    #对于通过recentft的label修正过的预测结果，使用自身的futureslice，进行再次的自校正
    # rclbmodifiedfilepath = 'E:/My competitions/didi road condition/code/predictresult/lgboostresult_lgb_new.csv'
    # linkinfo_onlyftlb = getmodifieddataonlyftlb(rclbmodifiedfilepath)
    # linkinfo = getsubmitdata(rclbmodifiedfilepath)
    # modifiedlinkinfo = modifylabelbycurlb_onlyrclb(linkinfo, linkinfo_onlyftlb,3,isfuture=1)
    # resultfilepath = 'E:/My competitions/didi road condition/code/predictresult/lgboostresult_lgb_new.csv'
    # writetimecrosslinkinfo(modifiedlinkinfo, resultfilepath)
    