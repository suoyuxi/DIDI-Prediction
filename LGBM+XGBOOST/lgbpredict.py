import os
from collections import Counter

from xgboostclassify.load_trainftdata import loadtrainftdata, loadtrainftdataby_date
import lightgbm as lgb
import joblib

from sklearn.metrics import classification_report
#输出每轮预测结果
def outputpredictresult(ans,y_test,trainindex):
    # 计算准确率
    cnt1 = 0
    cnt2 = 0
    for i in range(len(y_test)):
        if ans[i] == y_test[i]:
            cnt1 += 1
        else:
            cnt2 += 1
    print('model '+str(trainindex)+' finished:')
    print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))


    labelscore_report = classification_report(y_test,ans,target_names=['1','2','3'],output_dict=True)
    print(classification_report(y_test,ans,target_names=['1','2','3']))
    labelscore = labelscore_report['1']['f1-score']*0.2+labelscore_report['2']['f1-score']*0.2+labelscore_report['3']['f1-score']*0.6
    print('模型得分：'+str(labelscore))

    return labelscore
def predicttestdata(testX,testY,modelpath,saveresultpath,istest = 0):

    model = joblib.load(modelpath)
    resultlabel = model.predict(testX)
    print(Counter(resultlabel))
    result = getresultinfo(resultlabel,testX)
    if istest == 1:
        trainindex=0
        outputpredictresult(result, testY, trainindex)
    else:
        writeincsvresult(result,saveresultpath)

def getresultinfo(resultlabel,testX):
    result = []
    #测试数据整理
    # 读取文件内容
    for (line,label) in zip(testX,resultlabel):
        # headinfo = line.split(";")[0].split(' ')
        # linkid = headinfo[0]
        # currenslice = headinfo[2]
        # futureslice = headinfo[3]
        linetmp = list(line)[0:3]
        tmp = [int(linetmp[0]),int(linetmp[1]),int(linetmp[2]),int(label)]

        # tmp.extend([int(label)])
        result.append(tmp)
    return result

def takeFirst(elem):
    return int(elem[0])
#写入csv结果
def writeincsvresult(resultinfo,resultfilepath):
    resultinfo.sort(key=takeFirst)
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
def loadtrainftdatabyjson(testdatadir,testdate):
    links = os.listdir(testdatadir)
    return loadtrainftdataby_date(testdatadir, testdate, links)

if __name__ == '__main__':
    # testdatadir = 'E:/My competitions/didi road condition/code/processeddata/alltestlink724fts_lbs/'  # 731的数据做测试数据
    # istestcsv = '.csv'
    # testX, testY = loadtrainftdata(testdatadir, istestcsv)
    # modelpath = 'E:/My competitions/didi road condition/code/LgbSavedModels/lgb.pkl'
    # origintestfile = 'E:/My competitions/didi road condition/test/test/test.txt'
    # saveresultpath = '../predictresult/lgbresult_lgb63.csv'
    # predicttestdata(testX, testY, origintestfile, modelpath, saveresultpath,1)

    testdatadir = 'E:/My competitions/didi road condition/code/dataSelfcompleted/newfeatures4/extractedtestdatafts_ht-7-14缺失用traindata补全/'#731的数据做测试数据
    # istestcsv = '.csv'
    testdate=[1]
    testX, testY = loadtrainftdatabyjson(testdatadir,testdate)
    modelpath = 'E:/My competitions/didi road condition/code/LgbSavedModels/2/2_0_0.6667609636470291_lgb.pkl'
    # origintestfile = 'E:/My competitions/didi road condition/test/20190731-reordered.txt'
    saveresultpath = '../predictresult/new/lgboostresult_lgb_ht-7-14缺失用traindata补全_newfeatures4_2.csv'
    predicttestdata(testX,testY,modelpath,saveresultpath,istest=0)
