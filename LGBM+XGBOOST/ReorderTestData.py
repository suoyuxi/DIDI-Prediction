import os

from xgboostclassify.FeatureExtract import readjsontodict

#对测试集按之前处理成json的形式，重新排序
def reordertestdatabydict(datadir,saveftpath):
    for link in os.listdir(datadir):
        linkid = link.split('.')[0]
        filepath = datadir+link
        linkinfo = readjsontodict(filepath)
        reorderdatabydict(linkinfo, linkid, saveftpath)

def reorderdatabydict(linkinfo,linkid,savepath):
    features = []
    for onedayfts in linkinfo[linkid][:-1]:
        for ft in onedayfts:
            reorderedft = ';'.join(ft[0:6])
            features.append(reorderedft)
    savereordertestfts(savepath, features)

def savereordertestfts(savepath,features):
    # 创建txt文件
    with open(savepath, "a+") as f:  # 设置文件对象
        for i in features:  # 对于双层列表中的数据
            f.writelines(i)
            f.write('\n')  # 显示写入换行
    f.close()

if __name__ == '__main__':
    testdatadir = 'E:/My competitions/didi road condition/code/processeddata/TestData724ToDict/'
    saveftpath = '../processeddata/ReorderedTestData/test.txt'
    reordertestdatabydict(testdatadir,saveftpath)