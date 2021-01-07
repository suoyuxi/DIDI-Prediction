from xgboostclassify.final_merge import getdifferentresult
from xgboostclassify.remove0_1 import getsubmitdata, showresultCounter
from xgboostclassify.voteresult import gettestdataoriginhtlb



def showdifferent_check(lk1,lk2,lk3,lk4,voteresult,lklist,lkcheck1,lkcheck2,lkcheck3):
    countdiff = 0
    check1_2_diff = 0
    check1_diff_vote = 0
    check2_diff_vote = 0
    check3_diff_vote = 0

    for linkid in lk1.keys():
        for currentslice in lk1[linkid].keys():
            for futureslice in lk1[linkid][currentslice]:
                if lk1[linkid][currentslice][futureslice] == lk3[linkid][currentslice][futureslice] and lk2[linkid][currentslice][futureslice] == lk3[linkid][currentslice][futureslice]\
                        and lk3[linkid][currentslice][futureslice] == lkcheck1[linkid][currentslice][futureslice] \
                        and lkcheck1[linkid][currentslice][futureslice] == lkcheck2[linkid][currentslice][futureslice]:
                    continue
                else:
                    # countdiff+=1
                    # print(linkid, ' ', currentslice, ' ', futureslice, ' ',
                    #       lk1[linkid][currentslice][futureslice], lk2[linkid][currentslice][futureslice],
                    #       lk3[linkid][currentslice][futureslice], lk4[linkid][currentslice][futureslice],
                    #       lkcheck[linkid][currentslice][futureslice],
                    #       lklist[0][linkid][currentslice][futureslice],
                    #       lklist[1][linkid][currentslice][futureslice])
                    if lkcheck1[linkid][currentslice][futureslice] == lkcheck2[linkid][currentslice][futureslice] \
                            and lkcheck1[linkid][currentslice][futureslice] == voteresult[linkid][currentslice][futureslice]:
                        continue
                    else:
                        countdiff += 1
                        print(linkid, ' ', currentslice, ' ', futureslice, ' ',
                              lk1[linkid][currentslice][futureslice], lk2[linkid][currentslice][futureslice],
                              lk3[linkid][currentslice][futureslice], lk4[linkid][currentslice][futureslice],' ',
                              voteresult[linkid][currentslice][futureslice],' ',
                              lkcheck1[linkid][currentslice][futureslice],
                              lkcheck2[linkid][currentslice][futureslice],
                              lkcheck3[linkid][currentslice][futureslice],
                              lklist[0][linkid][currentslice][futureslice],
                              lklist[1][linkid][currentslice][futureslice])
                        if lkcheck1[linkid][currentslice][futureslice] != lkcheck2[linkid][currentslice][futureslice]:
                            check1_2_diff+=1
                        if lkcheck1[linkid][currentslice][futureslice] != voteresult[linkid][currentslice][futureslice]:
                            check1_diff_vote+=1
                        if lkcheck2[linkid][currentslice][futureslice] != voteresult[linkid][currentslice][futureslice]:
                            check2_diff_vote += 1
                        if lkcheck3[linkid][currentslice][futureslice] != voteresult[linkid][currentslice][futureslice]:
                            check3_diff_vote += 1



    print('all diff:',countdiff)
    print('check1_2_diff:',check1_2_diff)
    print('check1_diff_vote:',check1_diff_vote)
    print('check2_diff_vote:',check2_diff_vote)
    print('check3_diff_vote:',check3_diff_vote)

if __name__=='__main__':
    # submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/最好的结果/49.csv'
    # linkinfo_49 = getsubmitdata(submitdatafilepath)
    # submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/最好的结果/4748.csv'
    # linkinfo_4748 = getsubmitdata(submitdatafilepath)
    # submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/最好的结果/4747.csv'
    # linkinfo_4747 = getsubmitdata(submitdatafilepath)
    # submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/最好的结果/5012.csv'
    # linkinfo_5012 = getsubmitdata(submitdatafilepath)
    #
    # testdatafilepath = 'E:/My competitions/didi road condition/code/dataSelfcompleted/每行未缺失值小于等于3-自补全/20190801.txt'
    # linkoginfo_htall, linkoginfo_ht7 = gettestdataoriginhtlb(testdatafilepath)
    # testdatafilepath = 'E:/My competitions/didi road condition/code/dataSelfcompleted/ht-7-14缺失用traindata补全/20190801.txt'
    # linkscinfo_htall, linkscinfo_ht7 = gettestdataoriginhtlb(testdatafilepath)
    #
    # checkdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/check/result/embedding+focal+0.15，0.25，0.6.csv'
    # linkinfo_check1 = getsubmitdata(checkdatafilepath)
    # checkdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/check/result/focal+0.15，0.25，0.6.csv'
    # linkinfo_check2 = getsubmitdata(checkdatafilepath)
    # checkdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/check/result/focal+0.2，0.2，0.6.csv'
    # linkinfo_check3 = getsubmitdata(checkdatafilepath)
    #
    # print('0.49')
    # showresultCounter(linkinfo_49)
    # print('0.4748')
    # showresultCounter(linkinfo_4748)
    # print('0.4747')
    # showresultCounter(linkinfo_4747)
    # print('0.5012')
    # showresultCounter(linkinfo_5012)
    # print('check')
    # showresultCounter(linkinfo_check1)
    # showresultCounter(linkinfo_check2)
    # showresultCounter(linkinfo_check3)
    #
    # test_not_in_train_links = []
    # voteresult = getdifferentresult(linkinfo_4747, linkinfo_4748, linkinfo_49, linkinfo_5012,
    #                                 [linkoginfo_htall,
    #                                  # linkoginfo_ht7,
    #                                  linkscinfo_htall
    #                                  ],
    #                                 test_not_in_train_links,
    #                                 linkinfo_check1, linkinfo_check2,show=False)
    #
    # showdifferent_check(linkinfo_4747, linkinfo_4748, linkinfo_49, linkinfo_5012,voteresult,
    #                                 [linkoginfo_htall,
    #                                  linkscinfo_htall],
    #                                 linkinfo_check1,linkinfo_check2,linkinfo_check3)

    submitdatafilepath = 'E:/My competitions/didi road condition/code/predictresult/new/vote/47_48_49取众数_融合4FH.csv'
    linkinfo_49 = getsubmitdata(submitdatafilepath)

    showresultCounter(linkinfo_49)