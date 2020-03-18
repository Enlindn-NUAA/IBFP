#coding=utf-8
import numpy as np
import scipy.io as sio
import math

'''
TODO
用A跑regulardata 生成result116和118，116重新生成
A1跑regulardata生成result121
以及wforecast[0,2,5].npy

'''

standard_time = 1475424000

def get_database(w, timestamp, weekd, noweekd):
	first_timestamp = timestamp[0]
	sample_num = len(timestamp)
	database = []
	for i in range(48):
		database.append([[],[]])
	dealed_sample = []
	for i in range(sample_num-1):
		hour = int((timestamp[i]-standard_time)/3600%24)
		week = int(((timestamp[i]-standard_time)/3600/24+1)%7)
		if timestamp[i+1]-timestamp[i]==3600:
			if week in weekd:
				database[hour][0].append(w[i])
				database[hour][1].append(w[i+1])
			if week in noweekd:
				database[hour+24][0].append(w[i])
				database[hour+24][1].append(w[i+1])
	database = np.array(database)
	return database

def get_Least_squares_answer(A, Y):
	return np.dot(np.dot((np.dot(A.T, A)).I, A.T), Y)

def del_col(dat,nozerocol):
	dat2 = np.zeros((dat.shape[0],len(nozerocol)))
	for i in range(dat.shape[0]):
		for j in range(len(nozerocol)):
			dat2[i][j] = dat[i][nozerocol[j]]

	return dat2

def train(database, weekd, noweekd, k):
	Tr = np.zeros((48,k,k))
	for i in range(0,48):
		dbarray = np.array(database[i][0])
		zerocol = []
		nozerocol = []
		for db1 in range(dbarray.shape[1]):
			shia = True
			for db2 in range(dbarray.shape[0]):
				if dbarray[db2][db1]!=0:
					shia = False
			if shia:
				zerocol.append(db1)
			else:
				nozerocol.append(db1)
		db = del_col(np.array(database[i][0]),nozerocol)
		database[i][1] = del_col(np.array(database[i][1]),nozerocol)
		count = 0
		for j in range(db.shape[1]):
			y = []
			for kl in range(database[i][1].shape[0]):
				y.append([database[i][1][kl][j]])
			y = np.mat(y)
			beta = get_Least_squares_answer(np.mat(db),y)
			for kl in range(db.shape[1]):
				Tr[i][kl][nozerocol[j]] = beta[kl][0]
	return Tr


def get_next_w(A,wone,tim,noweekd):
	hour = int((tim-standard_time)/3600%24)
	week = int(((tim-standard_time)/3600/24+1)%7)
	isnoweeked = 0
	if week in noweekd:
		isnoweeked = 24
	return np.dot(wone,A[hour+isnoweeked])

def get_MAE(MA,True_MA):
	MAE = 0
	count = 0
	for i in range(MA.shape[0]):
		if True_MA[i]!=0:
			count += 1
			MAE += abs(True_MA[i]-MA[i])
	MAE = MAE / count
	return MAE

def get_RMSE(MA,True_MA):
	RMSE = 0
	count = 0
	for i in range(MA.shape[0]):
		if True_MA[i]!=0:
			count += 1
			RMSE += math.pow(True_MA[i]-MA[i], 2)
	RMSE = math.sqrt(RMSE / count)
	return RMSE

def get_ER(MA,True_MA):
	ER = 0
	count = 0
	for i in range(MA.shape[0]):
		if True_MA[i]!=0:
			count += 1
			ER += abs(True_MA[i]-MA[i])
	ER = ER / np.sum(True_MA)
	return ER




def test(A,w,H,testset,testtime,wnew, weekd, noweekd, timestamp):
	forcast_ = np.zeros((testset.shape[0],testset.shape[1]))
	MAE = 0
	RMSE = 0
	ER = 0
	result = []
	for i in range(24):
		hour = int((testtime[i]-standard_time)/3600%24)
		week = int(((testtime[i]-standard_time)/3600/24+1)%7)
		isnoweeked = 0
		if week in noweekd:
			isnoweeked = 24
		realtime = testtime[i]
		while realtime not in timestamp:
			realtime = realtime - 3600
		timeres = (testtime[i] - realtime) / 3600
		Wlast = w[np.where(timestamp==realtime)]
		for j in range(1,int(timeres)+1):
			Wlast = get_next_w(A,Wlast,realtime+j*3600,noweekd)
		wnew.append(Wlast)
		WHlast = np.dot(Wlast,H)
		forcast_[i] = WHlast		
		result.append([get_MAE(forcast_[i],testset[i]),get_RMSE(forcast_[i],testset[i]),get_ER(forcast_[i],testset[i])])
		print('MAE:', result[-1][0], 'RMSE:', result[-1][1], 'ER:', result[-1][2])
	result = np.array(result)
	return wnew

if __name__ == '__main__':
	weekd = [1,2,3,4,5]
	noweekd = [6,0]
	timestamp = np.load("./traindata/traintime.npy")
	first_timestamp = timestamp[0]
	sample_num = len(timestamp)
	w = hdf5storage.loadmat('./W1.mat')
	H = hdf5storage.loadmat('./H1.mat')
	w=w['S']
	H=H['B']
	#
	H = H.reshape(H.shape[0],-1)
	w = w.reshape(-1,w.shape[1])
	A=np.load('./A1.npy')
	testset = np.load('./testdata_taxi/regulardata.npy')
	first_dim=testset.shape[0]
	testset=testset.reshape(first_dim,-1)
	testtime = np.load('./testdata_taxi/regulartime.npy')
	wnew = []
	test(A,w,testset,testtime)
	wnew = np.array(wnew)
	np.save('./testdata_taxi/wforecast5.npy',wnew)








