#-*-coding:UTF-8-*-
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np
import time
# import matplotlib.pyplot as plt
import pandas as pd

def getbasis():
    #step 1: 加载数据
    print ("step 1: load data...")
    dataSet=[]#256
    filen = 1512486000
    fileIn=np.load('./traindata_taxi/traindata.npy')
    fileIn = fileIn.reshape((fileIn.shape[0],-1))
    # for i in range (1):
    #     print(i)
    #     dataSet1=[]
    #     fileIn = np.load('G:/125/out/DBSCANinout/inoutMatrix_%s.npy' % (str(filen)))
    #     # data = fileIn.tolist()
    #     # data = np.mat(data)
    #     # dataSet.append(data)
    #     filen+=3600
    dataSet = fileIn.tolist()
    print(len(dataSet))
    for k in range(9, 10):
        clf = KMeans(n_clusters=k)  # 设定k  ！！！！！！！！！！这里就是调用KMeans算法
        s = clf.fit(dataSet)  # 加载数据集合
        numSamples = len(dataSet)
        centroids = clf.labels_
        print(centroids, type(centroids))  # 显示中心点
        r1 = pd.Series(clf.labels_).value_counts()
        print(r1)
        print(clf.inertia_)  # 显示聚类效果
        np.save('clusterend%s.npy'%(str(k)),centroids)
        print(clf.cluster_centers_)
        np.save('clusterbasic%s.npy'%(str(k)),clf.cluster_centers_)
        print(type(clf.cluster_centers_))

    # data=np.array(dataSet)
    # for j in range(586756):
    #     data1=data[:,j]
    #     a=np.array(data1)
    #     if a==0:
    #         data=np.delete(data,j,axis=1)

    # for line in fileIn.readlines():
    #     lineArr = line.strip().split(' ')
    #     dataSet.append([float(lineArr[0]), float(lineArr[1])])

    # print(dataSet)
    #设定不同k值以运算
    # for k in range(5,6):
    #     clf = KMeans(n_clusters=k) #设定k  ！！！！！！！！！！这里就是调用KMeans算法
    #     s = clf.fit(dataSet) #加载数据集合
    #     numSamples = len(dataSet)
    #     centroids = clf.labels_
    #     print (centroids,type(centroids)) #显示中心点
    #     print (clf.inertia_)  #显示聚类效果
    #     mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        #画出所有样例点 属于同一分类的绘制同样的颜色
        # for i in range(numSamples):
        #     #markIndex = int(clusterAssment[i, 0])
        #     plt.plot(dataSet[i][0], dataSet[i][1], mark[clf.labels_[i]]) #mark[markIndex])
        # mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # # 画出质点，用特殊图型
        # centroids =  clf.cluster_centers_
        # for i in range(k):
        #     plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize = 12)
        #     #print centroids[i, 0], centroids[i, 1]
        # plt.show()

if __name__ == '__main__':
    getbasis()
