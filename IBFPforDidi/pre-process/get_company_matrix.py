#coding:utf-8
import numpy as np
import os
import time
from datetime import datetime

def getcompmatrix():
	holiday = [i for i in range(1506765600, 1507478400+3600, 3600)]
	weekd = [1,2,3,4,5]
	noweekd = [6,0]
	listtime = np.load('./traindata_taxi/traintime.npy')
	
	
	co = np.zeros((listtime.shape[0],listtime.shape[0]))
	print(co.shape)
	for i in range(co.shape[0]):
		tod = int(((listtime[i]-1427644800)/3600/24+1)%7)
		print(tod)
		for j in range(co.shape[1]):
			tod2 = int(((listtime[j]-1427644800)/3600/24+1)%7)
			if (tod in weekd) & (tod2 in weekd) & (abs(i-j)%24==0) & (listtime[i] not in holiday) & (listtime[j] not in holiday):
				co[i][j] = 1
			if ((tod in noweekd) | (listtime[i] in holiday)) & ((tod2 in noweekd) | (listtime[j] in holiday)) & (abs(i-j)%24==0):
				co[i][j] = 1
	for i in range(co.shape[0]):
		print(co[i].sum())
	print(co.sum())
	np.savetxt("./company_matrix",co)

if __name__ == '__main__':
	getcompmatrix()
