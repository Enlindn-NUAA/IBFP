#-*-coding:UTF-8-*-
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np
import time
# import matplotlib.pyplot as plt
import pandas as pd



if __name__ == '__main__':
    #step 1: 加载数据
    print ("step 1: load data...")
    # dataSet=[]#256
    # filen = 1522335600
    # df=pd.DataFrame
    # time=[]
    dataset=np.load('./subspace-clusterbasic9.npy')
    print(dataset.shape)
    achao
    sum=0
    martrix=[]
    for j in range (1260):
        martrix.append([])
        for k in range (9):
            martrix[j].append(0)
    for i in range(1260):
        try:
            j=dataset[i]
            martrix[i][j]+=1
        except:
            print(1)
    a=np.array(martrix)
    np.save('./subspace-W9.npy',martrix)
    # filen=1513432800
    # for i in range (300):
    #     filen += 3600
    #     print(i)
    #     dataSet1=[]
    #     try:
    #         fileIn = np.load('H:/1216/pro/out1/DBSCANinout1216/inoutMatrix_%s.npy' % (str(filen)))
    #     except:
    #         continue
    #     for j in range (766):
    #         martrix=fileIn[j]
    #         for k in range (766):
    #             dataSet1.append(martrix[k])
    #     dataSet.append(dataSet1)
    # print (sum)
    # print (len(time))
    # dataSet=np.array(dataSet)
    # dataSet1=pd.DataFrame(dataSet)
    # # dataSet1=dataSet1.replace(0,pd.np.nan).dropna(axis=1,how='all')
    # # dataSet1=dataSet1.replace(pd.np.nan,0)
    # dataSet =np.array(dataSet1)
    # time = np.array(time)
    # np.save('H:/1.31/newweek/week.npy', dataSet)
    # np.save('H:/1.31/newweek/listtime.npy', time)
    # print (dataSet1)
    # dataSet1.to_csv('H:/yangben.csv')
    # print(1)
    # dataSet1=np.array(dataSet1)#121522
    # np.save('H:/inout517.npy',dataSet1)
    # dataSet1=pd.DataFrame(dataSet1)
    # print(dataSet1)
    # for j in range(586756):
    #     data1=dataSet[:,j]
    #     a=np.sum(data1)
    #     if a==0:
    #         print(j)
    #         dataSet=np.delete(dataSet,j,axis=1)
    # print(1)
    # for line in fileIn.readlines():
    #     lineArr = line.strip().split(' ')
    #     dataSet.append([float(lineArr[0]), float(lineArr[1])])

    # print(dataSet)
    #设定不同k值以运算
    # from sklearn.decomposition import PCA
    # for k in range(10,11):
    # k=4
    # clf = KMeans(n_clusters=k) #设定k  ！！！！！！！！！！这里就是调用KMeans算法
    #     # pca = PCA(n_components=2)
    #     # dataset = pca.fit(dataSet)
    #     # print(dataset)
    #     # dataset1 = pca.fit(centroids)
    # s = clf.fit(dataSet1) #加载数据集合
    # numSamples = len(dataSet1)
    # print(numSamples)
    # centroids = clf.labels_
    # print (centroids,type(centroids)) #显示中心点
    # A=pd.DataFrame(centroids)
    # print(A)
    # print (clf.inertia_)  #显示聚类效果
    # showCluster(dataSet, k, centroids)
        # pca=PCA(n_components=2)
        # dataset=pca.fit(s)
        # dataset1=pca.fit(centroids)

    # mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    #     #画出所有样例点 属于同一分类的绘制同样的颜色
    # for i in range(numSamples):
    #     # plt.plot(dataSet1[i][0], dataSet1[i][1], mark[clf.labels_[i]])
    #     #mark[markIndex])
    #     plt.plot(dataSet1[i][i], dataSet1[i][i+1], mark[clf.labels_[i]])
    #     plt.plot(dataSet1[i][i], dataSet1[i][i + 2], mark[clf.labels_[i]])
    #     plt.plot(dataSet1[i][i], dataSet1[i][i + 3], mark[clf.labels_[i]])
    #     plt.plot(dataSet1[i][i], dataSet1[i][i + 4], mark[clf.labels_[i]])
    #     # plt.plot(dataSet1[i][0], dataSet1[i][2], mark[clf.labels_[i]])
    #     # plt.plot(dataSet1[i][0], dataSet1[i][3], mark[clf.labels_[i]])
    #     # plt.plot(dataSet1[i][0], dataSet1[i][4], mark[clf.labels_[i]])
    #     # plt.plot(dataSet1[i][0], dataSet1[i][5], mark[clf.labels_[i]])
    # mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    #     # 画出质点，用特殊图型
    # centroids =  clf.cluster_centers_
    # for i in range(k):
    #     plt.plot(dataSet1[i][0], dataSet1[i][1], mark[i], markersize = 12)
    #         #print centroids[i, 0], centroids[i, 1]
    # plt.show()