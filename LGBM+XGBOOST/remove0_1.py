from collections import Counter

from xgboostclassify.voteresult import gettestdataoriginhtlb


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
        #如果currentslice是第一次出现
        if currentslice not in linkinfo[linkid].keys():
            linkinfo[linkid].setdefault(currentslice, {})
        # 如果futuresilce是第一次出现
        if futuresilce not in linkinfo[linkid][currentslice].keys():
            linkinfo[linkid][currentslice].setdefault(futuresilce, {})
        linkinfo[linkid][currentslice][futuresilce] = label
    f.close()
    return linkinfo

def getconresult(link1info,link2info):
    count = 0
    for linkid in link1info:
        if linkid not in link2info.keys():
            for currentslice in link1info[linkid]:
                for futureslice in link1info[linkid][currentslice]:
                    link1info[linkid][currentslice][futureslice] = 1
                    count += 1
                    print([linkid, ' ', currentslice, ' ', futureslice, ' ',
                           link1info[linkid][currentslice][futureslice]])
        else:
            for currentslice in link1info[linkid]:
                if currentslice not in link2info[linkid].keys():
                    for futureslice in link1info[linkid][currentslice]:
                        link1info[linkid][currentslice][futureslice] = 1
                        count += 1
                        print([linkid, ' ', currentslice, ' ', futureslice, ' ',link1info[linkid][currentslice][futureslice]])
                else:
                    for futureslice in link1info[linkid][currentslice]:
                        try:
                            if futureslice not in link2info[linkid][currentslice].keys():
                                link1info[linkid][currentslice][futureslice] = 1
                                count+=1
                                print([linkid,' ',currentslice,' ',futureslice,' ',link1info[linkid][currentslice][futureslice]])
                        except:
                            print()
    print(count)
    return link1info

#对按时间穿越特征重新修改后的测试数据，重新写入csv文件中
def writetimecrosslinkinfo(linkinfo,resultfilepath):
    result = []
    resultlabel = []
    #按linkid和futureslice整理成list后写入
    for linkid in linkinfo.keys():
        for currentslice in linkinfo[linkid].keys():
            for futureslice in linkinfo[linkid][currentslice]:
                tmp = [linkid,currentslice,futureslice,linkinfo[linkid][currentslice][futureslice]]
                result.append(tmp)
                resultlabel.append(linkinfo[linkid][currentslice][futureslice])
    print(Counter(resultlabel))
    # writeincsvresult(result,resultfilepath)

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

def showresultCounter(linkinfo):
    resultlabel = []
    #按linkid和futureslice整理成list后写入
    for linkid in linkinfo.keys():
        for currentslice in linkinfo[linkid].keys():
            for futureslice in linkinfo[linkid][currentslice]:
                resultlabel.append(linkinfo[linkid][currentslice][futureslice])
    print(Counter(resultlabel))

def getdifferentresult(lk1,lk2,lk3,lk4,lklist):
    count_lk12_3 = 0
    count_lk3_support = 0
    count_lk123_diff = 0
    countall = 0

    for linkid in lk1.keys():
        for currentslice in lk1[linkid].keys():
            for futureslice in lk1[linkid][currentslice]:
                if lk1[linkid][currentslice][futureslice] == lk3[linkid][currentslice][futureslice] and lk2[linkid][currentslice][futureslice] == lk3[linkid][currentslice][futureslice]:
                    continue
                else:
                    changelb = lk3[linkid][currentslice][futureslice]
                    ischange = 0
                    if lk1[linkid][currentslice][futureslice] == lk2[linkid][currentslice][futureslice]:
                    #     # ht_ogalllbnum = int(lklist[0][linkid][currentslice][futureslice][lk1[linkid][currentslice][futureslice]-1].split('*')[1])
                    #     # ht_scalllbnum = int(lklist[0][linkid][currentslice][futureslice][lk1[linkid][currentslice][futureslice]-1].split('*')[1])
                    #     # if ht_ogalllbnum >= 8:
                    #     #     count += 1
                    #     #     changelb = lk1[linkid][currentslice][futureslice]
                        count_lk12_3 += 1
                        changelb = lk1[linkid][currentslice][futureslice]
                        ischange = 1

                        ht_ogalllb_nums_og = getmaxhtfeature_lb_nums(lklist[0][linkid][currentslice][futureslice])
                        ht_scalllb_nums_sc = getmaxhtfeature_lb_nums(lklist[1][linkid][currentslice][futureslice])

                        lk3_getsupport_og = getsupportfromhtft(ht_ogalllb_nums_og,lk3[linkid][currentslice][futureslice])
                        lk3_getsupport_sc = getsupportfromhtft(ht_scalllb_nums_sc, lk3[linkid][currentslice][futureslice])
                        if lk3_getsupport_og and lk3_getsupport_sc:
                            changelb = lk3[linkid][currentslice][futureslice]
                            count_lk3_support += 1
                            ischange = 1

                    elif lk1[linkid][currentslice][futureslice] != lk2[linkid][currentslice][futureslice] and lk1[linkid][currentslice][futureslice] != lk3[linkid][currentslice][futureslice] and lk2[linkid][currentslice][futureslice] != lk3[linkid][currentslice][futureslice]:
                        ht_ogalllb_nums = getmaxhtfeature_lb_nums(lklist[0][linkid][currentslice][futureslice])
                        ht_scalllb_nums = getmaxhtfeature_lb_nums(lklist[1][linkid][currentslice][futureslice])

                        satisfycondition1,changelb_c1 = issatisfycondition1(ht_ogalllb_nums)
                        satisfycondition2,changelb_c2 = issatisfycondition2(ht_ogalllb_nums,ht_scalllb_nums)

                        if satisfycondition1:
                            changelb = changelb_c1
                            count_lk123_diff+=1
                            ischange = 1
                        elif satisfycondition2:
                            changelb = changelb_c2
                            count_lk123_diff+=1
                            ischange=1
                        # ischange=1

                    if ischange:
                        print(linkid,' ',currentslice,' ',futureslice,' ',
                              lk1[linkid][currentslice][futureslice],lk2[linkid][currentslice][futureslice],lk3[linkid][currentslice][futureslice],lk4[linkid][currentslice][futureslice],
                              ' ', changelb,
                              lklist[0][linkid][currentslice][futureslice],
                              # lklist[1][linkid][currentslice][futureslice],
                              # lklist[2][linkid][currentslice][futureslice],
                              lklist[1][linkid][currentslice][futureslice])
                        lk3[linkid][currentslice][futureslice] = changelb

                    # print(linkid, ' ', currentslice, ' ', futureslice, ' ',
                    #       lk1[linkid][currentslice][futureslice], lk2[linkid][currentslice][futureslice],
                    #       lk3[linkid][currentslice][futureslice], lk4[linkid][currentslice][futureslice],
                    #       # ' ', changelb,
                    #       lklist[0][linkid][currentslice][futureslice],
                    #       # lklist[1][linkid][currentslice][futureslice],
                    #       # lklist[2][linkid][currentslice][futureslice],
                    #       lklist[1][linkid][currentslice][futureslice])

                    # lk3[linkid][currentslice][futureslice] = changelb
                    countall+=1

    print('count_lk12_3:',count_lk12_3)
    print('count_lk3_support',count_lk3_support)
    print('count_lk123_diff:',count_lk123_diff)
    print('countall:',countall)
    return lk3

def getsupportfromhtft(ht_ogalllb_nums,lk3lb):
    lk3index = -1
    for lbnums in ht_ogalllb_nums:
        lk3index += 1
        if lbnums[0] == lk3lb:
            break

    if lk3index == 0 and ht_ogalllb_nums[0][1] > 7 and ht_ogalllb_nums[1][1]<5:
        return 1
    else:
        return 0
def issatisfycondition1(ht_ogalllb_nums):
    if ht_ogalllb_nums[0][1] > 7 and ht_ogalllb_nums[1][1]<5:
        return 1,ht_ogalllb_nums[0][0]
    else:
        return 0,0

def issatisfycondition2(ht_ogalllb_nums,ht_scalllb_nums):

    if (ht_ogalllb_nums[0][1] > 7 and ht_ogalllb_nums[0][1] > 2*ht_ogalllb_nums[1][1])\
    or (ht_scalllb_nums[0][1] > 7 and ht_scalllb_nums[0][1] > 2*ht_scalllb_nums[1][1]):
        return 1,ht_ogalllb_nums[0][0]
    else:
        return 0,0

def getmaxhtfeature_lb_nums(ogallht):
    ht_ogalllb1num = int(ogallht[0].split('*')[1])
    ht_ogalllb2num = int(ogallht[1].split('*')[1])
    ht_ogalllb3num = int(ogallht[2].split('*')[1])
    htogall_lb_nums = [[1,ht_ogalllb1num],[2,ht_ogalllb2num],[3,ht_ogalllb3num]]
    htogall_lb_nums.sort(key=takeSecond,reverse=True)

    # ht_scalllb1num = int(scallht[0].split('*')[1])
    # ht_scalllb2num = int(scallht[1].split('*')[1])
    # ht_scalllb3num = int(scallht[2].split('*')[1])
    # htscall_lb_nums = [[1,ht_scalllb1num],[2,ht_scalllb2num],[3,ht_scalllb3num]]
    # htscall_lb_nums.sort(key=takeSecond)

    return htogall_lb_nums

def takeSecond(elem):
    return elem[1]


if __name__ == '__main__':
    # submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/first_0.49.csv'
    # link1info = getsubmitdata(submitdatafilepath)
    #
    # submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/lgboostresult_lgb_ht-7-14缺失用traindata补全_newfeatures4_去全零.csv'
    # link2info = getsubmitdata(submitdatafilepath)
    #
    #
    # linkinfo = getconresult(link1info,link2info)
    # resultfilepath = 'E:/My competitions/didi road condition/code/predictresult/new/lgboostresult_lgb_ht-7-14缺失用traindata补全_newfeatures4_去全零_049.csv'
    # writetimecrosslinkinfo(linkinfo,resultfilepath)

    submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/最好的结果/49.csv'
    linkinfo_49 = getsubmitdata(submitdatafilepath)
    submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/最好的结果/4748.csv'
    linkinfo_4748 = getsubmitdata(submitdatafilepath)
    submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/最好的结果/4747.csv'
    linkinfo_4747 = getsubmitdata(submitdatafilepath)
    submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/最好的结果/4986.csv'
    linkinfo_4986 = getsubmitdata(submitdatafilepath)

    testdatafilepath = 'E:/My competitions/didi road condition/code/dataSelfcompleted/每行未缺失值小于等于3-自补全/20190801.txt'
    linkoginfo_htall, linkoginfo_ht7 = gettestdataoriginhtlb(testdatafilepath)
    testdatafilepath = 'E:/My competitions/didi road condition/code/dataSelfcompleted/ht-7-14缺失用traindata补全/20190801.txt'
    linkscinfo_htall, linkscinfo_ht7 = gettestdataoriginhtlb(testdatafilepath)

    print('0.49')
    showresultCounter(linkinfo_49)
    print('0.4748')
    showresultCounter(linkinfo_4748)
    print('0.4747')
    showresultCounter(linkinfo_4747)
    print('0.4986')
    showresultCounter(linkinfo_4986)
    # print('origin test data')
    # showresultCounter(linkoginfo_htall)
    # showresultCounter(linkoginfo_ht7)
    # print('completed test data')
    # showresultCounter(linkscinfo_htall)
    # showresultCounter(linkscinfo_ht7)

    print('linkid ctsliceid ftsliceid 4747 4748 49 4986 ogall scall')
    voteresult = getdifferentresult(linkinfo_4747,linkinfo_4748,linkinfo_49,linkinfo_4986,
                       [linkoginfo_htall,
                        # linkoginfo_ht7,
                        linkscinfo_htall
                        ])
    resultfilepath = 'E:/My competitions/didi road condition/code/predictresult/new/vote/47_48_49取众数.csv'
    writetimecrosslinkinfo(voteresult,resultfilepath)

