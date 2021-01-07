from xgboostclassify.load_trainftdata import loadtrainftdata
import xgboost as xgb


from sklearn.metrics import classification_report

def predicttestdata(testX,testY,origintestfile,modelpath,saveresultpath):
    model = xgb.XGBClassifier()
    model.load_model(modelpath)
    resultlabel = model.predict(testX)

    result = getresultinfo(resultlabel,origintestfile)
    writeincsvresult(result,saveresultpath)

def getresultinfo(resultlabel,origintestfile):
    result = []
    #测试数据整理
    # 读取文件内容
    f = open(origintestfile, "r")
    lines = f.readlines()
    for (line,label) in zip(lines,resultlabel):
        headinfo = line.split(";")[0].split(' ')
        linkid = headinfo[0]
        currenslice = headinfo[2]
        futureslice = headinfo[3]
        result.append([linkid,currenslice,futureslice,int(label)])
    f.close()
    return result

def takeFirst(elem):
    return int(elem[0])
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

if __name__ == '__main__':

    testdatadir = 'E:/My competitions/didi road condition/code/processeddata/extractedtestdata801fts/'#731的数据做测试数据
    # istestcsv = '.csv'
    testX, testY = loadtrainftdata(testdatadir)
    modelpath = 'E:/My competitions/didi road condition/code/LgbSavedModels/各阶段最好的模型结果保存/第三次实验/0.577/1_2_0.5753547636321078_lgb.pkl'
    # origintestfile = 'E:/My competitions/didi road condition/test/20190731-reordered.txt'
    saveresultpath = '../predictresult/lgboostresult_lgb_new.csv'
    predicttestdata(testX,testY,modelpath,saveresultpath)
