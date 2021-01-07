import os
from collections import Counter

from xgboostclassify.voteresult import gettestdataoriginhtlb


def getlinks(dir):
    links = []
    for lk in os.listdir(dir):
        links.append(lk.split('.')[0])
    return links

def getnotintrainlinks(testlinks,trainlinks):
    notinlks = []
    for lkid in testlinks:
        if lkid not in trainlinks:
            notinlks.append(lkid)
    return notinlks

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

def getdifferentresult(lk1,lk2,lk3,lk4,lklist,test_not_in_train_links,lkcheck1,lkcheck2,show=True):
# def getdifferentresult(lk1, lk2, lk3, lk4, lklist, test_not_in_train_links,  show=True):
    count_lk12_3 = 0
    count_lk3_support1 = 0
    count_lk3_support2 = 0
    count_lk123_diff = 0
    count_test_not_in_train_links=0
    count_change_5012_diff = 0
    countall = 0

    for linkid in lk1.keys():
        for currentslice in lk1[linkid].keys():
            for futureslice in lk1[linkid][currentslice]:
                # if lkcheck1[linkid][currentslice][futureslice] == lk1[linkid][currentslice][futureslice] \
                #         and lkcheck2[linkid][currentslice][futureslice] == lk1[linkid][currentslice][futureslice] \
                #         and lk1[linkid][currentslice][futureslice] == lk3[linkid][currentslice][futureslice] \
                #         and lk2[linkid][currentslice][futureslice] == lk3[linkid][currentslice][futureslice]:
                if lk1[linkid][currentslice][futureslice] == lk3[linkid][currentslice][futureslice] \
                        and lk2[linkid][currentslice][futureslice] == lk3[linkid][currentslice][futureslice]:
                    # if linkid == showid:
                    #     print(linkid, ' ', currentslice, ' ', futureslice, ' ',
                    #           lk1[linkid][currentslice][futureslice], lk2[linkid][currentslice][futureslice],
                    #           lk3[linkid][currentslice][futureslice], lk4[linkid][currentslice][futureslice],
                    #           ' ', lk4[linkid][currentslice][futureslice], ' ',
                    #           lkcheck1[linkid][currentslice][futureslice], lkcheck2[linkid][currentslice][futureslice],
                    #           lklist[0][linkid][currentslice][futureslice],
                    #           # lklist[1][linkid][currentslice][futureslice],
                    #           # lklist[2][linkid][currentslice][futureslice],
                    #           lklist[1][linkid][currentslice][futureslice])
                    continue
                else:
                    changelb = lk3[linkid][currentslice][futureslice]
                    ischange = 0
                    # ischange=1
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


                        if _2_1_issatisfycondition1(ht_ogalllb_nums_og,ht_scalllb_nums_sc,lk3[linkid][currentslice][futureslice]):
                            changelb = lk3[linkid][currentslice][futureslice]
                            count_lk3_support1 += 1
                            # ischange = 0
                        elif _2_1_issatisfycondition2(ht_ogalllb_nums_og,ht_scalllb_nums_sc,lk3[linkid][currentslice][futureslice]):
                            changelb = lk3[linkid][currentslice][futureslice]
                            count_lk3_support2 += 1
                            # ischange = 0

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

                    show_change_5012_diff = ''
                    if changelb != lk4[linkid][currentslice][futureslice]:
                        count_change_5012_diff+=1
                        show_change_5012_diff = '!!!!!!!!!!!!!!!: ' + str(count_change_5012_diff)
                        ischange=1


                    if ischange:
                    # if ischange and str(linkid) in test_not_in_train_links:
                    #     count_test_not_in_train_links +=1
                        if show:
                            print(linkid,' ',currentslice,' ',futureslice,' ',
                                  lk1[linkid][currentslice][futureslice],lk2[linkid][currentslice][futureslice],lk3[linkid][currentslice][futureslice],lk4[linkid][currentslice][futureslice],
                                  ' ', changelb,' ',
                                  lkcheck1[linkid][currentslice][futureslice],lkcheck2[linkid][currentslice][futureslice],
                                  lklist[0][linkid][currentslice][futureslice],
                                  # lklist[1][linkid][currentslice][futureslice],
                                  # lklist[2][linkid][currentslice][futureslice],
                                  lklist[1][linkid][currentslice][futureslice],show_change_5012_diff)
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
    print('count_lk3_support1',count_lk3_support1)
    print('count_lk3_support2', count_lk3_support2)
    print('count_lk123_diff:',count_lk123_diff)
    print('count_test_not_in_train_links:',count_test_not_in_train_links)
    print('count_change_5012_diff:',count_change_5012_diff)
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
def _2_1_issatisfycondition1(ht_ogalllb_nums_og,ht_scalllb_nums_sc,lk3lb):
    lk3_getsupport_og = getsupportfromhtft(ht_ogalllb_nums_og, lk3lb)
    lk3_getsupport_sc = getsupportfromhtft(ht_scalllb_nums_sc, lk3lb)
    if lk3_getsupport_og and lk3_getsupport_sc:
        return 1
    else:
        return 0

def _2_1_issatisfycondition2(ht_ogalllb_nums,ht_scalllb_nums,lk3lb):
    if (ht_ogalllb_nums[0][1] > 7 and ht_ogalllb_nums[0][1] > 2*ht_ogalllb_nums[1][1])\
    and (ht_scalllb_nums[0][1] > 7 and ht_scalllb_nums[0][1] > 2*ht_scalllb_nums[1][1]):
        if ht_ogalllb_nums[0][0] == lk3lb:
            return 1
        else:
            return 0
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
    writeincsvresult(result,resultfilepath)

if __name__ == '__main__':

    test801dir = 'E:/My competitions/didi road condition/code/processeddata/TestData801ToDict'
    testlinks = getlinks(test801dir)

    traindatadir = 'E:/My competitions/didi road condition/code/processeddata/LinkinfoToDict'
    trainlinks = getlinks(traindatadir)

    test_not_in_train_links = getnotintrainlinks(testlinks,trainlinks)

    submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/最好的结果/49.csv'
    linkinfo_49 = getsubmitdata(submitdatafilepath)
    submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/最好的结果/4748.csv'
    linkinfo_4748 = getsubmitdata(submitdatafilepath)
    submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/最好的结果/4747.csv'
    linkinfo_4747 = getsubmitdata(submitdatafilepath)
    submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/最好的结果/5012.csv'
    linkinfo_5012 = getsubmitdata(submitdatafilepath)

    testdatafilepath = 'E:/My competitions/didi road condition/code/dataSelfcompleted/每行未缺失值小于等于3-自补全/20190801.txt'
    linkoginfo_htall, linkoginfo_ht7 = gettestdataoriginhtlb(testdatafilepath)
    testdatafilepath = 'E:/My competitions/didi road condition/code/dataSelfcompleted/ht-7-14缺失用traindata补全/20190801.txt'
    linkscinfo_htall, linkscinfo_ht7 = gettestdataoriginhtlb(testdatafilepath)

    checkdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/check/最后的挣扎/result.csv'
    linkinfo_check1 = getsubmitdata(checkdatafilepath)
    checkdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/check/最后的挣扎/result_sort.csv'
    linkinfo_check2 = getsubmitdata(checkdatafilepath)

    print('0.49')
    showresultCounter(linkinfo_49)
    print('0.4748')
    showresultCounter(linkinfo_4748)
    print('0.4747')
    showresultCounter(linkinfo_4747)
    print('0.5012')
    showresultCounter(linkinfo_5012)
    print('check')
    showresultCounter(linkinfo_check1)
    showresultCounter(linkinfo_check2)

    # print('origin test data')
    # showresultCounter(linkoginfo_htall)
    # showresultCounter(linkoginfo_ht7)
    # print('completed test data')
    # showresultCounter(linkscinfo_htall)
    # showresultCounter(linkscinfo_ht7)

    print('linkid ctsliceid ftsliceid 4747 4748 49 5012 ogall scall')
    voteresult = getdifferentresult(linkinfo_4747, linkinfo_4748, linkinfo_49, linkinfo_5012,
                                    [linkoginfo_htall,
                                     # linkoginfo_ht7,
                                     linkscinfo_htall
                                     ],
                                    test_not_in_train_links,
                                    linkinfo_check1,linkinfo_check2)
    resultfilepath = 'E:/My competitions/didi road condition/code/predictresult/new/vote/47_48_49取众数_融合4FH_NO.csv'
    writetimecrosslinkinfo(voteresult, resultfilepath)