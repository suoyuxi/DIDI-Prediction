




#{linkid:
#   {currentslice:
#       {ftslice1:label}
#       {ftslice2:label}
#   }
# }
#读取提交的测试数据集的csv,并以dict格式返回
from collections import Counter


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

#{linkid:
#   {currentslice:
#       {ftslice1:-7htlabel,allhtlabel}
#       {ftslice2:-7htlabel,allhtlabel}
#   }
# }
def gettestdataoriginhtlb(filepath):
    linkinfo_onhtlb_all = {}
    linkinfo_onhtlb_7 = {}
    # 读取训练数据文件内容
    f = open(filepath, "r")
    lines = f.readlines()
    for line in lines:
        infoline = line.strip('\n')
        infoline = infoline.split(';')

        headinfo = infoline[0].split(' ')
        linkid = int(float(headinfo[0]))
        currentslice = int(float(headinfo[2]))
        futuresilce = int(float(headinfo[3]))

        allht_features = infoline[2:]
        ht_7_feature = infoline[-1]
        allhtlabel = getallhtftslabel(allht_features)
        ht_7_label = get_htftlabel(ht_7_feature)

        #如果linkid是第一次出现
        if linkid not in linkinfo_onhtlb_all.keys():
            linkinfo_onhtlb_all.setdefault(linkid, {})
            linkinfo_onhtlb_7.setdefault(linkid, {})
        #如果currentslice是第一次出现
        if currentslice not in linkinfo_onhtlb_all[linkid].keys():
            linkinfo_onhtlb_all[linkid].setdefault(currentslice, {})
            linkinfo_onhtlb_7[linkid].setdefault(currentslice, {})
        # 如果futuresilce是第一次出现
        if futuresilce not in linkinfo_onhtlb_all[linkid][currentslice].keys():
            linkinfo_onhtlb_all[linkid][currentslice].setdefault(futuresilce, {})
            linkinfo_onhtlb_7[linkid][currentslice].setdefault(futuresilce, {})

        linkinfo_onhtlb_all[linkid][currentslice][futuresilce] = allhtlabel
        linkinfo_onhtlb_7[linkid][currentslice][futuresilce] = ht_7_label
    f.close()

    return linkinfo_onhtlb_all,linkinfo_onhtlb_7

def getallhtftslabel(historyfts):

    labels = [0,0,0]
    # 统计各道路recent_features状态
    for line in historyfts:
        feature = line.split(' ')

        for ft in feature:
            if int(ft.split(',')[2]) == 1:
                labels[0] += 1
            elif int(ft.split(',')[2]) == 2:
                labels[1] += 1
            elif int(ft.split(',')[2]) > 2:
                labels[2] += 1

    if sum(labels) == 0:
        # return 0
        return [str(1)+'*'+str(labels[0]),str(2)+'*'+str(labels[1]),str(3)+'*'+str(labels[2])]
    else:
        label1num = labels[0]
        label2num = labels[1]
        label3num = labels[2]
        # if label1num > label2num+label3num:
        #     return str(1)+'*'+str(sum(labels))
        # elif label2num>label3num:
        #     return str(2)+'*'+str(sum(labels))
        # else:
        #     return str(3)+'*'+str(sum(labels))
        # if labels[2] == 20:
        #     print()
        return [str(1)+'*'+str(labels[0]),str(2)+'*'+str(labels[1]),str(3)+'*'+str(labels[2])]



def get_htftlabel(recentft):
    # 统计各道路recent_features状态
    feature = recentft.split(' ')

    # 统计当前道路的recent_feature的各个状态
    labels = [0,0,0]  # 临时存储feature的状态
    for ft in feature:
        if int(ft.split(',')[2]) == 1:
            labels[0] += 1
        elif int(ft.split(',')[2]) == 2:
            labels[1] += 1
        elif int(ft.split(',')[2]) > 2:
            labels[2] += 1

    if sum(labels) == 0:
        return 0
    else:
        label1num = labels[0]
        label2num = labels[1]
        label3num = labels[2]
        if label1num > label2num+label3num:
            return 1
        elif label2num>label3num:
            return 2
        else:
            return 3

def voteforresult(linkinfo,linkhtinfo_all,linkhtinfo_7):
    diffnum = 0
    samenum = 0
    bothhtzero = 0
    ht_7zero = 0
    ht_allzero = 0

    label3 = 0
    label2 = 0

    clabel1 = 0
    clabel2 = 0
    clabel3 = 0

    changematrix = [[0,0,0],
                    [0,0,0],
                    [0,0,0]]
    labelmatrix = [ [0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0]]

    threelb_3 = 0
    for linkid in linkinfo:
        for currentslice in linkinfo[linkid]:
            for futureslice in linkinfo[linkid][currentslice]:
                predlb = linkinfo[linkid][currentslice][futureslice]
                ftalllb = linkhtinfo_all[linkid][currentslice][futureslice]
                ft_7lb = linkhtinfo_7[linkid][currentslice][futureslice]

                labelmatrix[0][predlb] += 1
                labelmatrix[1][ftalllb] += 1
                labelmatrix[2][ft_7lb] += 1

                # if  ftalllb == 2 and ft_7lb != 2 and ft_7lb!=0:
                #     threelb_3+=1

                if ft_7lb != predlb and ft_7lb==2:
                    linkinfo[linkid][currentslice][futureslice] = ft_7lb
                    diffnum+=1

                # if ftalllb == ft_7lb and ft_7lb == 0:
                #     continue
                # elif ftalllb == ft_7lb and predlb != ftalllb:
                #     diffnum += 1
                #     linkinfo[linkid][currentslice][futureslice] = ft_7lb
                #     changematrix[predlb-1][ft_7lb-1] += 1

                # if ftalllb == ft_7lb and ft_7lb == 0:
                #     bothhtzero += 1
                # elif predlb == 0:
                #     ht_7zero+=1
                # elif ftalllb == 0:
                #     ht_allzero+=1

                # if predlb == ftalllb and predlb == ft_7lb:
                #     samenum += 1
                # elif ftalllb == ft_7lb and ft_7lb == 0:
                #     bothhtzero += 1
                # elif predlb == 0:
                #     ht_7zero+=1
                # elif ftalllb == 0:
                #     ht_allzero+=1
                # elif ftalllb == ft_7lb and predlb != ftalllb:
                #     diffnum+=1
                #     if predlb == 2:
                #         label2+=1
                #     elif predlb == 3:
                #         label3+=1
                #     if ftalllb == 1:
                #         clabel1+=1
                #     elif ftalllb == 2:
                #         clabel2+=1
                #     elif ftalllb == 3:
                #         clabel3+=1
                #     print('pred:', predlb, ' allft:', ftalllb, ' ft_7:', ft_7lb)


                # if predlb != ftalllb and ftalllb!=0:
                #     diffnum+=1
                #     print('pred:',predlb,' allft:',ftalllb,' ft_7:',ft_7lb)
                # elif predlb != ft_7lb and ft_7lb!=0:
                #     diffnum += 1
                #     print('pred:', predlb, ' allft:', ftalllb, ' ft_7:', ft_7lb)
                # else:
                #     samenum+=1
    # print(diffnum)
    # print(samenum)
    # print(bothhtzero)
    # print(ht_7zero)
    # print(ht_allzero)
    # print(label3)
    # print(label2)
    #
    # print(clabel3)
    # print(clabel2)
    # print(clabel1)

    for l in labelmatrix:
        print(l)
    print(threelb_3)
    return linkinfo


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
if __name__=='__main__':

    submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/lgboostresult_lgb_new_nocomplt_htvote_all_7.csv'
    linkinfo = getsubmitdata(submitdatafilepath)


    testdatafilepath = 'E:/My competitions/didi road condition/code/dataSelfcompleted/testdata/20190801.txt'
    linkhtinfo_all,linkhtinfo_7 = gettestdataoriginhtlb(testdatafilepath)

    linkinfovoteresult = voteforresult(linkinfo,linkhtinfo_all,linkhtinfo_7)

    resultfilepath = 'E:/My competitions/didi road condition/code/predictresult/lgboostresult_lgb_new_nocomplt_htvote_all_7.csv'
    writetimecrosslinkinfo(linkinfovoteresult,resultfilepath)













