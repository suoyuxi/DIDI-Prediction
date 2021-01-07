import csv
import math
import os
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, classification_report
from xgboostclassify.load_trainftdata import loadtrainftdata, loadtrainftdatabynum, loadtrainftdataby_lbproportion
import pandas as pd
maxlabelscore = 0

#每次训练后用模型对测试集进行预测
def predicttestdata(testX,testY):
    model = xgb.XGBClassifier()
    model.load_model('E:/My competitions/didi road condition/code/SavedModels/xgb.model')
    result = model.predict(testX)
    labelscore_report = classification_report(testY, result, target_names=['1', '2', '3'], output_dict=True)
    print(classification_report(testY, result, target_names=['1', '2', '3']))
    labelscore = labelscore_report['1']['f1-score'] * 0.2 + labelscore_report['2']['f1-score'] * 0.2 + \
                 labelscore_report['3']['f1-score'] * 0.6
    print('模型得分：' + str(labelscore))
    # savepredictedresult(result)

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
#自定义评估函数
def f1_score_eval(valid_df,preds):
    labels = np.argmax(valid_df.reshape(3, -1), axis=0)
    preds = preds.get_label()
    scores = f1_score(y_true=labels, y_pred=preds, average=None)
    scores = scores[0]*0.2+scores[1]*0.2+scores[2]*0.6
    return 'f1_score', scores

def trainxgboost1(X,Y,testX,testY,trainindex,Feature):
    modeldir = 'E:/My competitions/didi road condition/code/SavedModels/'

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=52)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = Feature

    if trainindex == 1:
        # 训练模型
        model = xgb.XGBClassifier(max_depth=None, learning_rate=0.1, n_estimators=5000,objective='multi:softmax')
        print('第一次训练，创建模型')
    else:
        model = xgb.XGBClassifier()
        model.load_model('E:/My competitions/didi road condition/code/SavedModels/xgb.model')

    # K折交叉验证
    folds = KFold(n_splits=5, shuffle=True, random_state=2020)
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
        print('the {} training start ...'.format(fold))
        lgb_train_X = X_train[trn_idx]
        lgb_train_y = y_train[trn_idx]
        lgb_val_X = X_train[val_idx]
        lgb_val_y = y_train[val_idx]
        model.fit(lgb_train_X, lgb_train_y, eval_set=[(lgb_val_X, lgb_val_y)],
                  eval_metric=lambda y_true,y_pred: [f1_score_eval(y_true,y_pred)], early_stopping_rounds=100,verbose=100)

        fold_importance_df[f'fold_{fold}_imp'] = model.feature_importances_
    five_folds = [f'fold_{f}_imp' for f in range(0, 5)]
    fold_importance_df['avg_imp'] = fold_importance_df[five_folds].mean(axis=1)
    fold_importance_df.sort_values(by='avg_imp', ascending=False, inplace=True)
    print(fold_importance_df[['Feature', 'avg_imp']].head(20))


    # 对测试集进行预测
    ans = model.predict(X_test)

    result = model.predict(testX)
    # savepredictedresult(result)


    labelscore_train = outputpredictresult(ans,y_test,trainindex)#输出训练集中分出的测试集结果
    labelscore_test = outputpredictresult(result,testY,trainindex)#输出以730日数据为测试集的结果
    global maxlabelscore#全局变量用于保存在测试集上预测效果最好的分模型

    #仅保存在测试集上预测效果最好的分模型
    if labelscore_test > maxlabelscore:
        maxlabelscore= labelscore_test
        savefolder = os.path.join(modeldir,str(trainindex))
        if not os.path.exists(savefolder):  # 判断是否存在文件夹如果不存在则创建为文件夹
             os.makedirs(savefolder)  # makedirs 创建文件时如果路径不存在会创建这个路径
        model.save_model(os.path.join(savefolder,str(trainindex)+'_'+str(labelscore_test)+'_'+'xgb.model'))
        # 显示重要特征
        plot_importance(model)
        plt.savefig(os.path.join(savefolder, str(trainindex) + 'ftimportance.jpg'))
        plt.close()

    #总模型保存用于下一次训练
    model.save_model(os.path.join(modeldir,'xgb.model'))


def savepredictedresult(result):
    #测试数据整理
    # 读取文件内容
    f = open('E:/My competitions/didi road condition/test/test/test.txt', "r")
    lines = f.readlines()
    # 分离出①道路的id 当前时间片 待预测时间片 label ②recent_feature③history_feature
    # ①头部数据

    resultpath = '../predictresult/xgboostresult_all.csv'
    resultfile = open(resultpath, 'w', newline='')
    csv_writer = csv.writer(resultfile, dialect='excel', lineterminator='\n')
    csv_writer.writerow(['link','current_slice_id','future_slice_id','label'])
    for (line,label) in zip(lines,result):
        headinfo = line.split(";")[0].split(' ')
        linkid = headinfo[0]
        currenslice = headinfo[2]
        futureslice = headinfo[3]
        csv_writer.writerow([linkid,currenslice,futureslice,int(label)])
    f.close()
    resultfile.close()
#列表等分割
def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]
#训练集太大，分开训练
def aparttrain(traindatadir,istraincsv,testdatadir,istestcsv,oneturndatanum,Feature,labelproportion=[3,1,1]):
    #加载测试数据
    testX, testY = loadtrainftdata(testdatadir, istestcsv)
    # testX=[]
    # testY = []
    print('test data successfully loaded')
    #数据按比例抽样后可不用分批次训练
    if oneturndatanum == 30:

        links = os.listdir(traindatadir)
        # 测试用
        links = chunks(links, oneturndatanum)

        trainindex = 0#仅训练1次
        for linklist in links:
            X, Y = loadtrainftdataby_lbproportion(traindatadir, istraincsv,linklist,labelproportion)
            print('train data successfully loaded ' + str(trainindex))
            # 数据打乱
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]

            print(str(len(X))+' '+str(len(Y))+' data shuffled start training xgboost!')
            trainxgboost1(X, Y, testX, testY, trainindex,Feature)

    elif oneturndatanum > 1:
        #分批加载训练数据并训练
        links = os.listdir(traindatadir)
        linksapart = chunks(links,oneturndatanum)
        trainindex = 0
        for linklist in linksapart:
            X, Y = loadtrainftdatabynum(traindatadir, istraincsv,linklist)
            trainindex =  trainindex+1
            print('train data successfully loaded '+str(trainindex))
            #数据打乱
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]
            print('data shuffled, start training xgboost!')
            trainxgboost1(X, Y, testX,testY, trainindex)

if __name__ == '__main__':
    traindatadir = 'E:/My competitions/didi road condition/code/processeddata/extractedfeatures/'#提取特征后的训练数据，按linkid分成文件
    istraincsv = ''#文件尾标是否为.csv
    # testdatadir = 'E:/My competitions/didi road condition/code/processeddata/alltestlinkfts_lbs/'#731的数据做测试数据
    testdatadir = 'E:/My competitions/didi road condition/code/processeddata/alltestlink730fts_lbs/'  # 730的数据做测试数据
    istestcsv = '.csv'
    oneturndatanum = 30#每次提取500条道路的信息进行训练
    Feature = ['linktime1', 'linktime2', 'linktime3', 'linktime4', 'linktime5',
               'linkattr1', 'linkattr2', 'linkattr3', 'linkattr4', 'linkattr5', 'linkattr6', 'linkattr7',
               'recentft1', 'recentft2', 'recentft3', 'recentft4', 'recentft5', 'recentft6', 'recentft7', 'recentft8','recentft9',
               'history1ft1', 'history1ft2', 'history1ft3', 'history1ft4', 'history1ft5', 'history1ft6', 'history1ft7','history1ft8', 'history1ft9',
               'history2ft1', 'history2ft2', 'history2ft3', 'history2ft4', 'history2ft5', 'history2ft6', 'history2ft7','history2ft8', 'history2ft9',
               'history3ft1', 'history3ft2', 'history3ft3', 'history3ft4', 'history3ft5', 'history3ft6', 'history3ft7','history3ft8', 'history3ft9',
               'history4ft1', 'history4ft2', 'history4ft3', 'history4ft4', 'history4ft5', 'history4ft6', 'history4ft7','history4ft8', 'history4ft9',
               'historyallft1', 'historyallft2', 'historyallft3', 'historyallft4', 'historyallft5', 'historyallft6','historyallft7', 'historyallft8', 'historyallft9']

    aparttrain(traindatadir,istraincsv,testdatadir,istestcsv,oneturndatanum,Feature)

    # #仅预测测试数据
    # testX, testY = loadtrainftdata(testdatadir, istestcsv)
    # predicttestdata(testX,testY)