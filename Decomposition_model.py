#coding:utf-8
from numpy.linalg import norm

import numpy as np

from collections import namedtuple

from scipy.optimize import minimize

from numpy.linalg import norm

from measure import Measure

from pandas import DataFrame

from time import time

import math

import scipy.io as sio


class Decomposition(object):

    def __init__(self):

        self.eps = 1e-7

        self._P = namedtuple('_Parameters',
                        'V C k lambda_ sigma sigma_a sigma_b eta theta gamma lmd I diagnol_col_idxes non_diagnol_col_idxes W_shape H_shape')

    def _reshape(self,W, H, parameters):

        return W.reshape(parameters.W_shape), H.reshape(parameters.H_shape)

    def mylog(self, mat):

        mat = np.where(mat > 0, mat, 1)

        return np.log(mat)

    def _f(self, W, H, parameters):
        '''
        Equation 12

        :param W: Coefficient matrix (N × K)
        :param H: base matrices (K × M)
        :param parameters:
        :return: loss funtion
        '''

        W, H = self._reshape(W, H, parameters)

        P = parameters

        WH_diagnol = np.dot(W, H[:, P.diagnol_col_idxes])

        WH = np.dot(W, H)

        # Equation 12
        Loss_function = -np.sum((P.V * np.log(WH + self.eps) - WH) * P.I) + norm(W) ** 2 / P.sigma_a ** 2

        Loss_function += -(np.sum(
            (P.eta - 1) * np.log(H[:, P.diagnol_col_idxes] + self.eps) - P.theta * H[:, P.diagnol_col_idxes]) + np.sum(
            (P.sigma - 1) * np.log(H[:, P.non_diagnol_col_idxes] + self.eps) - P.sigma_b * H[:, P.non_diagnol_col_idxes]))

        # Equation 6
        return Loss_function + P.lambda_ * np.dot(np.dot(W.T, P.C), W).trace() + P.gamma * norm(
            (P.V - np.dot(W, H)) * P.I) ** 2 + P.lmd * np.sum(np.absolute(W))

    def __grad_f_W(self,W, H, parameters):

        W, H = self._reshape(W, H, parameters)

        P = parameters

        WH = np.dot(W, H)

        grad = - ((P.V * 1. / (WH + self.eps) - 1) * P.I).dot(H.T) + P.lambda_ * np.dot(P.C + P.C.T,
                    W) + 1 / P.sigma_a ** 2 * W + 2 * P.gamma * np.dot((P.V - np.dot(W, H)) * P.I, -H.T)

        return grad.reshape(np.product(grad.shape))

    def __grad_f_H(self,W, H, parameters):

        W, H = self._reshape(W, H, parameters)

        P = parameters

        WH_diagnol = np.dot(W, H[:, P.diagnol_col_idxes])

        WH = np.dot(W, H)

        WH_non_diagnol = np.dot(W, H[:, P.non_diagnol_col_idxes])

        grad = np.zeros(H.shape)

        grad[:, P.diagnol_col_idxes] = - W.T.dot(
            (P.V[:, P.diagnol_col_idxes] / (WH_diagnol + self.eps) - 1) * P.I[:, P.diagnol_col_idxes]) \
                                       - (P.eta - 1) / H[:, P.diagnol_col_idxes] + P.theta - 2 * P.gamma * W.T.dot(
            (P.V[:, P.diagnol_col_idxes] - np.dot(W, H[:, P.diagnol_col_idxes])) * P.I[:, P.diagnol_col_idxes])
        grad[:, P.non_diagnol_col_idxes] = - W.T.dot(
            (P.V[:, P.non_diagnol_col_idxes] / (WH_non_diagnol + self.eps) - 1) * P.I[:, P.non_diagnol_col_idxes]) \
                                           - (P.sigma - 1) / H[:,
                                                             P.non_diagnol_col_idxes] + P.sigma_b - 2 * P.gamma * W.T.dot(
            (P.V[:, P.non_diagnol_col_idxes] - np.dot(W, H[:, P.non_diagnol_col_idxes])) * P.I[:, P.non_diagnol_col_idxes])

        return grad.reshape(np.product(grad.shape))


    def _pgd(self,W, H, parameters):

        '''
        Proximal Gradient Algorithm

        :return:  Coefficient matrix
        '''

        W, H = self._reshape(W, H, parameters)

        P = parameters

        f_value = self._f(W, H, parameters)

        last_f_value = f_value + 100

        iter_time = 0

        one_divide_L = 0.1 * (0.5)

        new_W = np.copy(W)

        while ((abs(last_f_value - f_value) > 10) & (iter_time < 50)):

            tree_count = [0, 0, 0]

            temp_grad = self.__grad_f_W(W, H, parameters)

            for i in range(P.V.shape[0]):

                for j in range(P.k):

                    z = W[i][j] - one_divide_L * temp_grad[i * P.k + j]

                    if P.lmd * one_divide_L < z:

                        tree_count[0] += 1

                        new_W[i][j] = z - P.lmd * one_divide_L

                    elif -(P.lmd * one_divide_L) > z:

                        new_W[i][j] = 0

                        tree_count[1] += 1

                    else:

                        new_W[i][j] = 0

                        tree_count[2] += 1

            W_temp = W

            W = new_W

            new_W = W_temp

            last_f_value = f_value

            f_value = self._f(W, H, parameters)

            iter_time += 1

            change_flag = True

            if (f_value > last_f_value) | (math.isnan(f_value)):

                W_temp = W

                W = new_W

                new_W = W_temp

                f_value_temp = f_value

                f_value = last_f_value

                last_f_value = f_value_temp

                change_flag = False

                print("the", iter_time, "time iteration: last_f_value:", f_value, "  new_f_value:", last_f_value,
                      "change_flag:", change_flag, "stat(up,down,zero):", tree_count)

                if one_divide_L > 1e-15:

                    one_divide_L = one_divide_L * 0.5

            if change_flag == True:

                print("the", iter_time, "time iteration: last_f_value:", last_f_value, "  new_f_value:", f_value,
                      "change_flag:", change_flag, "stat:", tree_count)

        return W.reshape(np.product(W.shape))


    def nmf6(self,V, C, k=9, lambda_=1, sigma=1, sigma_a=1e-2, sigma_b=1, eta=1, theta=1, gamma=1, lmd=1, max_iter=8, WInit=None,

             HInit=None):
        # print(V.shape)
        # print(C.shape)
        # initialize

        W = WInit if WInit is not None else np.random.uniform(0, 10, (V.shape[0], k))

        # H = HInit if HInit is not None else np.random.uniform(0, 10, (k, V.shape[1]))

        H = np.load('./pre-train/basis_from_subspace_clustering.npy').reshape(k,-1) # auto setting another dimention
        # print(H.shape)
        # return

        r = int(np.sqrt(V.shape[1]))

        diagnol_col_idxes = [i * r + i for i in range(r)]

        non_diagnol_col_idxes = np.setdiff1d(range(r ** 2), diagnol_col_idxes)

        parameters = self._P(V, C, k, lambda_, sigma, sigma_a, sigma_b, eta, theta, gamma, lmd, np.where(V > 0, 1, 0),
                        diagnol_col_idxes, non_diagnol_col_idxes, W.shape, H.shape)

        print('W shape (%d,%d)' % W.shape)

        print('H shape (%d,%d)' % H.shape)

        n_W = np.product(W.shape)

        n_H = np.product(H.shape)

        W = W.reshape(n_W)

        H = H.reshape(n_H)

        for iter_count in range(max_iter):

            print('*** in iter %d ***' % iter_count)

            W = self._pgd(W, H, parameters)

            np.save('./Wtemp%d' % iter_count, W)

            np.save('./Htemp%d' % iter_count, H)

        return list(self._reshape(W, H, parameters))

    def main(self, V, C, super_parameters):

        data = 'data2'

        ks = super_parameters['ks']

        lambda_s = super_parameters['lambda_s'] # trade-off parameter

        sigma_as = super_parameters['sigma_as']  #

        sigma_bs = super_parameters['sigma_bs']  #

        etas = super_parameters['etas']  # Parameter of Gamma distribution

        thetas = super_parameters['thetas']  # Parameter of Gamma distribution

        sigmas = super_parameters['sigmas']  # parameter of Gaussian distribution

        gammas = super_parameters['gammas']  # regularization parameter

        lmd_s = super_parameters['lmd_s']

        R_test = V

        result = DataFrame(
            columns=["Data Set", "lambda", "sigma", "sigma A", "sigma B", "eta", "theta", "gamma", "lmd", "k"]
                    + [i + str(j) for i in ["NDCG@", "Precision@", "Recall@"] for j in [3, 5, 10]])

        idx = 0

        start = time()

        for k in ks:

            WInit = np.random.uniform(1, 2, (V.shape[0], k))

            HInit = np.random.uniform(1, 2, (k, V.shape[1]))

            for lambda_ in lambda_s:

                for sigma_a in sigma_as:

                    for sigma_b in sigma_bs:

                        for eta in etas:

                            for gamma in gammas:

                                for theta in thetas:

                                    for sigma in sigmas:

                                        for lmd in lmd_s:
                                            W, H = self.nmf6(V, C, k, lambda_, sigma, sigma_a, sigma_b, eta, theta, gamma,
                                                        lmd, WInit=WInit, HInit=HInit)

                                            WH = W.dot(H)

                                            m = Measure(WH, R_test)  # Measure in code

                                            precision_recall = list(zip(*[m.precision_recall(i) for i in [3, 5, 10]]))

                                            result.loc[idx] = [data, lambda_, sigma, sigma_a, sigma_b, eta, theta,
                                                               gamma,
                                                               lmd, k, m.ndcg(3), m.ndcg(5), m.ndcg(10)] + list(
                                                precision_recall[0]) + list(precision_recall[1])

                                            print('W max %.2f min %.2f norm %.2f' % (W.max(), W.min(), norm(W)))

                                            print('H max %.2f min %.2f norm %.2f' % (H.max(), H.min(), norm(H)))

                                            np.save('./W%d' % idx, W)

                                            np.save('./H%d' % idx, H)

                                            idx += 1

                                            result.to_csv('./result.csv')

        print('\n%.2f seconds cost\n' % (time() - start))
        return W,H


if __name__ == '__main__':
    demo=Decomposition()
    # demo.main()


