import joblib
from lightgbm import plot_importance
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd

Feature = ['linkid',
           'linktime1', 'linktime2', 'linktime3', 'linktime4', 'linktime5',
           'linkattr1', 'linkattr2', 'linkattr3', 'linkattr4', 'linkattr5', 'linkattr6', 'linkattr7',
           'recentft1', 'recentft2', 'recentft3', 'recentft4', 'recentft5', 'recentft6', 'recentft7', 'recentft8',
           'recentft9',
           'history1ft1', 'history1ft2', 'history1ft3', 'history1ft4', 'history1ft5', 'history1ft6', 'history1ft7',
           'history1ft8', 'history1ft9',
           'history2ft1', 'history2ft2', 'history2ft3', 'history2ft4', 'history2ft5', 'history2ft6', 'history2ft7',
           'history2ft8', 'history2ft9',
           'history3ft1', 'history3ft2', 'history3ft3', 'history3ft4', 'history3ft5', 'history3ft6', 'history3ft7',
           'history3ft8', 'history3ft9',
           'history4ft1', 'history4ft2', 'history4ft3', 'history4ft4', 'history4ft5', 'history4ft6', 'history4ft7',
           'history4ft8', 'history4ft9',
           'historyallft1', 'historyallft2', 'historyallft3', 'historyallft4', 'historyallft5', 'historyallft6',
           'historyallft7', 'historyallft8', 'historyallft9']
ChineseFt = ['linkid',

             '当前时间片','待预测时间片','待预测时间片和当前时间片的差值的绝对值','星期几','是否工作日',

             'link的的功能等级','link的速度限制等级','link的车道数','link的限速','link的level','link的长度','link的宽度',

             'Rc畅通状态占比','Rc缓行状态占比','Rc拥堵状态占比','Rc平均路况速度','Rc平均eta速度','Rc平均路况速度/平均eta速度','Rc平均路况速度/link限速','Rc平均eta速度/link限速','Rc参与路况计算的车辆总数',

             'Ht1畅通状态占比','Ht1缓行状态占比','Ht1拥堵状态占比','Ht1平均路况速度','Ht1平均eta速度','Ht1平均路况速度/平均eta速度','Ht1平均路况速度/link限速','Ht1平均eta速度/link限速','Ht1参与路况计算的车辆总数',

             'Ht2畅通状态占比', 'Ht2缓行状态占比', 'Ht2拥堵状态占比', 'Ht2平均路况速度', 'Ht2平均eta速度', 'Ht2平均路况速度/平均eta速度', 'Ht2平均路况速度/link限速','Ht2平均eta速度/link限速', 'Ht2参与路况计算的车辆总数',

             'Ht3畅通状态占比', 'Ht3缓行状态占比', 'Ht3拥堵状态占比', 'Ht3平均路况速度', 'Ht3平均eta速度', 'Ht3平均路况速度/平均eta速度', 'Ht3平均路况速度/link限速','Ht3平均eta速度/link限速', 'Ht3参与路况计算的车辆总数',

             'Ht4畅通状态占比', 'Ht4缓行状态占比', 'Ht4拥堵状态占比', 'Ht4平均路况速度', 'Ht4平均eta速度', 'Ht4平均路况速度/平均eta速度', 'Ht4平均路况速度/link限速','Ht4平均eta速度/link限速', 'Ht4参与路况计算的车辆总数',

             'Htall畅通状态占比', 'Htall缓行状态占比', 'Htall拥堵状态占比', 'Htall平均路况速度', 'Htall平均eta速度', 'Htall平均路况速度/平均eta速度', 'Htall平均路况速度/link限速','Htall平均eta速度/link限速', 'Htall参与路况计算的车辆总数']

Chinese2Ft = ['linkid',

             '当前时间片','待预测时间片','待预测时间片和当前时间片的差值的绝对值','星期几','是否工作日',

             'link的的功能等级','link的速度限制等级','link的车道数','link的限速','link的level','link的长度','link的宽度',

             'Rc畅通状态占比','Rc缓行状态占比','Rc拥堵状态占比','Rc平均路况速度','Rc平均eta速度','Rc平均路况速度/平均eta速度','Rc平均路况速度/link限速','Rc平均eta速度/link限速','Rc参与路况计算的车辆总数',

             'Ht4r畅通状态占比', 'Ht4r缓行状态占比', 'Ht4r拥堵状态占比', 'Ht4r平均路况速度', 'Ht4r平均eta速度', 'Ht4r平均路况速度/平均eta速度', 'Ht4r平均路况速度/link限速','Ht4r平均eta速度/link限速', 'Ht4r参与路况计算的车辆总数',

             'Htall畅通状态占比', 'Htall缓行状态占比', 'Htall拥堵状态占比', 'Htall平均路况速度', 'Htall平均eta速度', 'Htall平均路况速度/平均eta速度', 'Htall平均路况速度/link限速','Htall平均eta速度/link限速', 'Htall参与路况计算的车辆总数']

Chinese3Ft = ['linkid',

             '当前时间片','待预测时间片','待预测时间片和当前时间片的差值的绝对值',

             'link的限速','link的的功能等级','link的速度限制等级','link的车道数','link的level','link的长度','link的宽度',

             'Rc畅通状态占比','Rc缓行状态占比','Rc拥堵状态占比','主要状态','主要状态占比','Rc平均路况速度','Rc平均eta速度','Rc平均路况速度/平均eta速度','Rc平均路况速度/link限速','Rc平均eta速度/link限速','Rc参与路况计算的车辆总数',

             'Ht4r畅通状态占比', 'Ht4r缓行状态占比', 'Ht4r拥堵状态占比', 'Ht4主要状态', 'Ht4主要状态占比', 'Ht4r平均路况速度', 'Ht4r平均eta速度','Ht4r平均路况速度/平均eta速度', 'Ht4r平均路况速度/link限速', 'Ht4r平均eta速度/link限速', 'Ht4r参与路况计算的车辆总数',

             'Ht3r畅通状态占比', 'Ht3r缓行状态占比', 'Ht3r拥堵状态占比', 'Ht3主要状态', 'Ht3主要状态占比', 'Ht3r平均路况速度', 'Ht3r平均eta速度', 'Ht3r平均路况速度/平均eta速度', 'Ht3r平均路况速度/link限速','Ht3r平均eta速度/link限速', 'Ht3r参与路况计算的车辆总数',

             'Htall畅通状态占比', 'Htall缓行状态占比', 'Htall拥堵状态占比', 'Htall主要状态','Htall主要状态占比','Htall平均路况速度', 'Htall平均eta速度', 'Htall平均路况速度/平均eta速度', 'Htall平均路况速度/link限速','Htall平均eta速度/link限速', 'Htall参与路况计算的车辆总数']


model = joblib.load('E:/My competitions/didi road condition/code/LgbSavedModels/各阶段最好的模型结果保存/第四次实验/1/1_0_0.664591486644224_lgb.pkl')
ftimportance = model.feature_importances_

fold_importance_df = pd.DataFrame()
fold_importance_df["Feature"] = Chinese3Ft
fold_importance_df['avg_imp'] = ftimportance
fold_importance_df.sort_values(by='avg_imp', ascending=False, inplace=True)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_colwidth',500)
print(fold_importance_df[['Feature', 'avg_imp']].head(67))


