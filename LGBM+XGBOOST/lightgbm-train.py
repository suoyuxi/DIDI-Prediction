import csv
import math
import os
import numpy as np
import lightgbm as lgb
from lightgbm import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, classification_report
from xgboostclassify.load_trainftdata import loadtrainftdata, loadtrainftdatabynum, loadtrainftdataby_lbproportion, \
    loadtrainftdataby_date, loadtrainftdataby_lbpp_date, loadtrainftdataby_date_lbnum
import joblib
import pandas as pd
maxlabelscore = 0

#每次训练后用模型对测试集进行预测
def predicttestdata(testX,testY):
    model = lgb.LGBMClassifier()
    model.load_model('E:/My competitions/didi road condition/code/LgbSavedModels/lgb.model')
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
    labels = valid_df
    preds = np.argmax(preds.reshape(3, -1), axis=0)
    scores = f1_score(y_true=labels, y_pred=preds, average=None)
    scores = scores[0]*0.2+scores[1]*0.2+scores[2]*0.6
    return 'f1_score', scores, True

def trainlightgbm(X,Y,testX,testY,trainindex,Feature,nsplits=5):
    modeldir = 'E:/My competitions/didi road condition/code/LgbSavedModels/'

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=52)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = Feature

    if trainindex == 1:
        # 训练模型
        model = lgb.LGBMClassifier(
            learning_rate=0.1,#梯度下降的步长。默认设置成0.1,我们一般设置成0.05-0.2之间
            num_iterations=5000,#迭代次数
            max_depth=35,#限制树模型的最大深度. 在 #data 小的情况下防止过拟合. 树仍然可以通过 leaf-wise 生长.< 0 意味着没有限制.
            num_leaves=40,#一棵树上的叶子数
            feature_fraction=0.8,#如果 feature_fraction 小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征
            bagging_fraction=0.8,#类似于 feature_fraction, 但是它将在不进行重采样的情况下随机选择部分数据
            bagging_freq=5,#bagging 的频率, 0 意味着禁用 bagging. k 意味着每 k 次迭代执行bagging
            reg_alpha=0.001,#表示的是L1正则化
            reg_lambda=8,#表示的是L2正则化
            cat_smooth=10,#可以降低噪声在分类特征中的影响, 尤其是对数据很少的类别
            boosting_type='gbdt',#传统的梯度提升决策树
            objective='malticlass',#任务目标，multiclass, softmax 目标函数, 应该设置好 num_class
            num_class=3,
            # silent=True,#为1时模型运行不输出
            # is_unbalance=True,
            n_estimators=20,
            min_child_samples=21,#一个叶子上数据的最小数量。可以用来处理过拟合。
            min_child_weight=0.001#若是基学习器切分后得到的叶节点中样本权重和低于该阈值则不会进一步切分，该值越大模型的学习约保守，同样用于防止模型过拟合
            # metric=None#度量函数
        )
        print('第一次训练，创建模型')
    else:
        model = joblib.load('E:/My competitions/didi road condition/code/LgbSavedModels/lgb.pkl')

    # labelscore_test = 0
    global maxlabelscore  # 全局变量用于保存在测试集上预测效果最好的分模型
    #K折交叉验证
    folds = KFold(n_splits=nsplits, shuffle=True, random_state=2020)
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
        print('the {} training start ...'.format(fold))
        lgb_train_X = X_train[trn_idx]
        lgb_train_y = y_train[trn_idx]
        lgb_val_X = X_train[val_idx]
        lgb_val_y = y_train[val_idx]
        model.fit(lgb_train_X, lgb_train_y,eval_set=[(lgb_val_X, lgb_val_y)],
                  eval_metric=lambda y_true, y_pred: [f1_score_eval(y_true,y_pred)],early_stopping_rounds=200,verbose=200)

        fold_importance_df[f'fold_{fold}_imp'] = model.feature_importances_
        result = model.predict(testX)
        labelscore_test = outputpredictresult(result, testY, trainindex)  # 输出以730日数据为测试集的结果

        # 仅保存在测试集上预测效果最好的分模型
        if labelscore_test > maxlabelscore:
            maxlabelscore = labelscore_test
            savefolder = os.path.join(modeldir, str(trainindex))
            if not os.path.exists(savefolder):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(savefolder)  # makedirs 创建文件时如果路径不存在会创建这个路径
            joblib.dump(model, os.path.join(savefolder, str(trainindex)+'_'+str(fold) + '_' + str(labelscore_test) + '_' + 'lgb.pkl'))
            # 显示重要特征
            plot_importance(model)
            plt.savefig(os.path.join(savefolder, str(trainindex)+'_'+str(fold) + '_ftimportance.jpg'))
            plt.close()

    five_folds = [f'fold_{f}_imp' for f in range(0, nsplits)]
    fold_importance_df['avg_imp'] = fold_importance_df[five_folds].mean(axis=1)
    fold_importance_df.sort_values(by='avg_imp', ascending=False, inplace=True)
    print(fold_importance_df[['Feature', 'avg_imp']].head(20))

    # 对测试集进行预测
    # ans = model.predict(X_test)
    #
    # result = model.predict(testX)
    # savepredictedresult(result)


    # labelscore_train = outputpredictresult(ans,y_test,trainindex)#输出训练集中分出的测试集结果
    # labelscore_test = outputpredictresult(result,testY,trainindex)#输出以730日数据为测试集的结果

    #总模型保存用于下一次训练
    joblib.dump(model,os.path.join(modeldir,'lgb.pkl'))


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
def aparttrain(traindatadir,traindate,testdate,oneturndatanum,Feature,trainindex=0,labelproportion=[3,1,1]):
    links = os.listdir(traindatadir)
    testdatadir = 'E:/My competitions/didi road condition/code/dataSelfcompleted/newfeatures4/extractedtestdatafts_ht-7-14缺失用traindata补全/'

    testlinkslist = gettestlinkslist(links,testdatadir)

    print('开始加载测试数据-全部加载:',len(testlinkslist),' ',testdate,' ',[15000,400,200])
    testX, testY = loadtrainftdataby_date_lbnum(traindatadir,testdate,testlinkslist,[15000,400,200])
    # testX=[]
    # testY=[]
    print('test data successfully loaded:',str(len(testY)))
    #数据按比例抽样后可不用分批次训练
    if oneturndatanum == 1:
        # 测试用
        print('开始分批次训练：',str(oneturndatanum))
        links = chunks(links, oneturndatanum)

        trainindex = trainindex#仅训练1次
        for linklist in links:
            trainindex+=1
            # 加载测试数据训练数据按比例加载，测试数据全部加载
            print('开始按label比例加载训练数据')
            trainX, trainY = loadtrainftdataby_lbpp_date(traindatadir, traindate, linklist, labelproportion)
            # trainX, trainY = loadtrainftdataby_date(traindatadir, traindate, linklist)
            print('train data successfully loaded ',str(trainindex))
            # 数据打乱
            indices = np.arange(trainX.shape[0])
            np.random.shuffle(indices)
            trainX = trainX[indices]
            trainY = trainY[indices]

            print(str(len(trainX)) + ' ' + str(len(trainY)) + ' data shuffled start training lightgbm!')
            trainlightgbm(trainX, trainY, testX, testY, trainindex,Feature)

    elif oneturndatanum > 1:
        #分批加载训练数据并训练
        links = os.listdir(traindatadir)
        linksapart = chunks(links,oneturndatanum)
        trainindex = 0
        for linklist in linksapart:
            X, Y = loadtrainftdataby_date(traindatadir,traindate,links)
            trainindex =  trainindex+1
            print('train data successfully loaded '+str(trainindex))
            #数据打乱
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]
            print('data shuffled, start training xgboost!')
            trainlightgbm(X, Y, testX,testY, trainindex)

def gettestlinkslist(traindatalinks,testdatadir):
    traindatalinks = set(traindatalinks)
    testdatalinks = os.listdir(testdatadir)
    resultlinks = []
    for lk in testdatalinks:
        if lk in traindatalinks:
            resultlinks.append(lk)
    return resultlinks




if __name__ == '__main__':
    traindatadir = 'E:/My competitions/didi road condition/code/dataSelfcompleted/newfeatures4/extractedfeatures/'#提取特征后的训练数据，按linkid分成文件
    # traindatadir = 'E:/My competitions/didi road condition/code/processeddata/extractedfeatures/'
    # traindate = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30]
    # traindate = [25]
    # traindate = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 25, 26, 29,
    #              30]
    traindate = [15, 16, 17, 18, 19,
                 22, 23, 24, 25, 26,
                 29,30]
    # testdate = [1]
    testdate = [25, 26,
                 29,30]
    oneturndatanum = 1#每次提取500条道路的信息进行训练
    ChineseFt = ['linkid',

                  '当前时间片', '待预测时间片', '待预测时间片和当前时间片的差值的绝对值',

                  'link的限速', 'link的的功能等级', 'link的速度限制等级', 'link的车道数', 'link的level', 'link的长度', 'link的宽度',

                  'Rc畅通状态占比', 'Rc缓行状态占比', 'Rc拥堵状态占比', '主要状态', '主要状态占比', 'Rc平均路况速度', 'Rc平均eta速度', 'Rc平均路况速度/平均eta速度',
                  'Rc平均路况速度/link限速', 'Rc平均eta速度/link限速', 'Rc参与路况计算的车辆总数',

                  'Ht4r畅通状态占比', 'Ht4r缓行状态占比', 'Ht4r拥堵状态占比', 'Ht4主要状态', 'Ht4主要状态占比', 'Ht4r平均路况速度', 'Ht4r平均eta速度',
                  'Ht4r平均路况速度/平均eta速度', 'Ht4r平均路况速度/link限速', 'Ht4r平均eta速度/link限速', 'Ht4r参与路况计算的车辆总数',

                  'Ht3r畅通状态占比', 'Ht3r缓行状态占比', 'Ht3r拥堵状态占比', 'Ht3主要状态', 'Ht3主要状态占比', 'Ht3r平均路况速度', 'Ht3r平均eta速度',
                  'Ht3r平均路况速度/平均eta速度', 'Ht3r平均路况速度/link限速', 'Ht3r平均eta速度/link限速', 'Ht3r参与路况计算的车辆总数',

                  'Htall畅通状态占比', 'Htall缓行状态占比', 'Htall拥堵状态占比', 'Htall主要状态', 'Htall主要状态占比', 'Htall平均路况速度',
                  'Htall平均eta速度', 'Htall平均路况速度/平均eta速度', 'Htall平均路况速度/link限速', 'Htall平均eta速度/link限速',
                  'Htall参与路况计算的车辆总数']

    trainindexorigin = 0 #接着之前的轮数继续训练
    labelproportion = [10, 4, 1]
    trainnum = 5#模型在全部数据上多轮训练
    for trainnum in range(0,trainnum):#全部数据一共训练trainnum轮
        print('第',str(trainnum),'轮训练：')
        print('》》》》》》》》》》》》》》》》》》》》》》》》~^o^~《《《《《《《《《《《《《《《《《《《《《')
        trainindex = trainindexorigin + trainnum*oneturndatanum
        aparttrain(traindatadir,traindate,testdate,oneturndatanum,ChineseFt,trainindex,labelproportion)
