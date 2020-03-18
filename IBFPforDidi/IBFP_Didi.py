#coding:utf-8
import numpy as np
import Decomposition_model
import Learn_transition_matrix
import sys

if_pre_decomposition_flag = True
if_pre_train_flag = True
k = 10
if if_pre_train_flag==False:
	if if_pre_decomposition_flag==False:
		#input data
		V = np.load("./pre-process/traindata_taxi/traindata.npy")
		V = V.reshape(V.shape[0],-1)
		C = np.loadtxt("./pre-process/company_matrix")

		#parameters
		parameters = {'ks': [i for i in range(5,15)], 'lambda_s':[10.0 ** i for i in range(-10,-1)], 'sigma_as':[10.0], 'sigma_bs':[5], 'etas':[4], 'thetas':[10], 'sigmas':[8], 'gammas':[i for i in range(0.1, 1.0)], 'lmd_s':[i for i in range(0.1, 1.0)]}

		De_model=Decomposition_model.Decomposition()
		W, H = De_model.main(V, C, parameters)
		timestamp = np.load("./pre-process/traindata_taxi/traintime.npy")

	else:
		'''
		W = np.load('./pre-train/W.npy')
		H = np.load('./pre-train/H.npy')
		H = H.reshape((k,-1))
		W = W.reshape((-1,H.shape[0]))
		timestamp = np.load("./pre-train/listtime.npy")
		'''
		sys.exit()
	#learn A
	weekd = [1,2,3,4,5]
	noweekd = [6,7]
	H = H.reshape((H.shape[0],-1))
	print(W.min(),W.max(),W.mean(),W.std())
	print(H.min(),H.max(),H.mean(),H.std())
	print(W.shape, H.shape, timestamp.shape)
	database = Learn_transition_matrix.get_database(W, timestamp, weekd, noweekd)
	A = Learn_transition_matrix.train(database, weekd, noweekd, k)

else:
	'''
	weekd = [1,2,3,4,5]
	noweekd = [6,7]
	A = np.load('./pre-train/A.npy')
	W = np.load('./pre-train/W.npy')
	H = np.load('./pre-train/H.npy')
	H = H.reshape((k,-1))
	W = W.reshape((-1,H.shape[0]))
	timestamp = np.load("./pre-train/listtime.npy")
	'''
	sys.exit()

testset = np.load('./pre-process/testdata_taxi/regulardata.npy')
testtime = np.load('./pre-process/testdata_taxi/regulartime.npy')
wnew = []
Learn_transition_matrix.test(A, W, H, testset, testtime, wnew, weekd, noweekd, timestamp)
wnew = np.array(wnew)
np.save('./testdata/wforecast.npy',wnew)