# 获得设计矩阵
import pandas as pd
import numpy as np
import os

year = 2022
path = './data/' + str(year) + '/'
# 设计矩阵差额项
delta = 1e-07
# 读取空 dataframe
csv_file = path + "20220101.csv"
csv_data = pd.read_csv(csv_file, low_memory = False)#防止弹出警告
csv_df = pd.DataFrame(csv_data)
if csv_df.empty:
    # 若为dataframe，则循环继续
    continue

# 读取非空 dataframe
csv_file = path + "20220104.csv"
csv_data = pd.read_csv(csv_file, low_memory = False)#防止弹出警告
csv_df = pd.DataFrame(csv_data)
csv_df

a = csv_df['close']/(csv_df['open']+delta)
data1 = pd.DataFrame(np.array(a).reshape(1,-1))
data1.columns = csv_df['ts_code']
data1.loc[:, 'trade_date'] = csv_df['trade_date'][0]
data1 = pd.concat([data1['trade_date'], data1.drop(columns='trade_date')], axis=1)



csv_file = path + "20220105.csv"
csv_data = pd.read_csv(csv_file, low_memory = False)#防止弹出警告
csv_df = pd.DataFrame(csv_data)
csv_df

a = csv_df['close']/(csv_df['open']+delta)
data2 = pd.DataFrame(np.array(a).reshape(1,-1))
data2.columns = csv_df['ts_code']
data2.loc[:, 'trade_date'] = csv_df['trade_date'][0]
data2 = pd.concat([data2['trade_date'], data2.drop(columns='trade_date')], axis=1)


mid = pd.merge(data1,data2,how='outer')
pd.concat([data1,data2], axis=1)
mid.to_csv('./code/data/' + str(year) +'.csv')

time_start1 = time.time()  # 记录开始时间
matrix1 = pd.concat([data1,data2], ignore_index=True, sort=False)
time_end1 = time.time()  # 记录结束时间
time_sum1 = time_end1 - time_start1  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum1)



# mv_test

import numpy as np
import pandas as pd
import nonlinshrink as nls #nonlinear shrinkage
from sklearn.covariance import LedoitWolf #linear shrinkage
from sklearn.model_selection import TimeSeriesSplit

from cvxopt import matrix
from cvxopt import solvers
from tqdm import tqdm
import numba as nb
from spo.utils import nb_type_f

def design_train_clip(design_m):
    # 删除NA
    design_m = design_m.dropna(axis=1, how='any')
    # 获得指标
    mid = design_m.drop(columns='trade_date')
    index = mid.columns
    # 获得矩阵
    design_m = np.array(mid)
    # 筛选前2000个指标
    fillter_index = (np.mean(design_m, axis=0)>-np.sort(-np.mean(design_m, axis=0))[2000])
    index = mid.columns[fillter_index]
    result_mat = design_m[:,fillter_index]
    return(result_mat, index)


def design_test_clip(design_m):
    # 填充NA为0
    design_m = design_m.fillna(value=0)
    # 获得矩阵
    design_m = np.array(design_m)
    return(design_m)


def proj(w):
    d = w.shape[0]
    sort_w = -np.sort(-w, axis=None) # 用于获得降序
    tmp = (np.cumsum(sort_w) - 1) * (1.0 / np.arange(1, d + 1))
    rho = np.sum(sort_w > tmp) - 1
    w = np.maximum(w - tmp[rho], 0)
    return w

def mean_var_opt(Sigma, mu, lam_MV=0.0):
    d = Sigma.shape[0]

    eps = 1e-6
    # 浮点数中非负的最小数
    delta = np.finfo(np.float64).eps
    while True:
        try:
            P = matrix(lam_MV * Sigma, tc='d')
            q = matrix(-mu, tc='d')
            G = matrix(-np.eye(d), tc='d')
            h = matrix(np.zeros(d), tc='d')
            A = matrix(np.ones(d).reshape((1, -1)), tc='d')
            b = matrix(np.ones(1), tc='d')
            #sol = solvers.qp(P, q, G, h, A, b, options={'show_progress': True,
            #                                            'abstol': 1e-12, 'reltol': 1e-11,
            #                                            'maxiters': int(1e2), 'feastol': 1e-16})
            sol = solvers.qp(P, q, G, h, A, b, options={'show_progress': True,
                                                        'abstol': 1e-7, 'reltol': 1e-6,
                                                        'maxiters': int(1e2), 'feastol': 1e-7})
            # sol['x']为求解结果；转换为数组后，flatten用于拉直，输出为1*n维数组
            w = np.array(sol['x']).flatten()
            w[w <= delta] = 0.
            w = proj(w)
            break
        except:
            print('singular')
            Sigma = Sigma + np.identity(d) * eps
            eps *= 10

    return w


def cov(X, method):
    '''
    Parameters
    ----------
    X : np.array
        The sample matrix with size \(n, p\).
    method : str
        The method used to estimate the covariance.

    Returns
    ----------
    Cov : np.array
        The estimated covariance matrix.
    '''
    if method.startswith('MV-P'):
        return np.cov(X, rowvar=False)
    elif method.startswith('MV-LS'):
        cov = LedoitWolf(assume_centered=False).fit(X)
        return cov.covariance_
    elif method.startswith('MV-NLS'):
        return nls.shrink_cov(X)


def eval(X_train, X_val, method, lam_MV=0.0):
    Sigma = cov(X_train, method)
    mu = np.mean(X_train, axis=0) - 1
    w = mean_var_opt(Sigma, mu, lam_MV)
    return np.dot(X_val - 1, w), w



# 读取设计矩阵
design_M = pd.read_csv('./code/data/design_matrix.csv', low_memory=False)
design_M = design_M.drop(columns='Unnamed: 0')


# 用于索引为期半年的训练矩阵和为期三个月的测试矩阵
index_number = [20200000,20200100,20200200,20200300,20200400,20200500,20200600,20200700,20200800,20200900,20201000,20201100,20201200,
                20210100,20210200,20210300,20210400,20210500,20210600,20210700,20210800,20210900,20211000,20221100,20211200,
                20220100,20220200,20220300,20220400,20220500,20220600,20220700,20220800,20220900,20221000,20221100,20221200,20221300]
itrs = 8
# 模型固定参数设置
# 效应函数设定
func_names = ['LOG', 'EXP']
func=0
a = 1.
# mv方法设定
k = 0
method_list = ['MV-P', 'MV-LS', 'MV-NLS']
method = method_list[k]

n_lam_MV = 100
n_folds = 5



# 保存输出结果的路径
path_result = './code/result/'


# 获得用于训练的设计矩阵和用于测试的设计矩阵
i = 0
train_index = ((index_number[3 * i + 13] > design_M['trade_date']) & (index_number[3 * i + 7] < design_M['trade_date']))
test_index = ((index_number[3 * i + 16] > design_M['trade_date']) & (index_number[3 * i + 13] < design_M['trade_date']))
train_design_m = design_M.loc[train_index,]
train_design_m, index = design_train_clip(train_design_m)
test_design_m = design_M.loc[test_index, index]
test_design_m = design_test_clip(test_design_m)
# 首先获得 lambdas 用于随后的交叉检验
lam_MVs = np.logspace(np.log10(1e-3), np.log10(1e2), num=n_lam_MV)[::-1]

#### test

Sigma = cov(train_design_m, method)
mu = np.mean(train_design_m, axis=0) - 1

import time
time_start1 = time.time()  # 记录开始时间
w11 = mean_var_opt(Sigma, mu, lam_MVs[50])
time_end1 = time.time()  # 记录结束时间
time_sum1 = time_end1 - time_start1  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum1)
np.where(w11>0)
w11.shape


# GMV/MV方法权重筛选原则

import numpy as np

def proj(w):
    d = w.shape[0]
    sort_w = -np.sort(-w, axis=None) # 用于获得降序
    tmp = (np.cumsum(sort_w) - 1) * (1.0 / np.arange(1, d + 1))
    rho = np.sum(sort_w > tmp) - 1
    w = np.maximum(w - tmp[rho], 0)
    return w

w = np.linspace(start = 1, stop = 10, num = 10)
w = w/sum(w)











# 迭代充分性讨论
import numpy as np
import pandas as pd
import nonlinshrink as nls #nonlinear shrinkage
from sklearn.covariance import LedoitWolf #linear shrinkage

from cvxopt import matrix
from cvxopt import solvers


def design_train_clip(design_m):
    # 删除NA
    design_m = design_m.dropna(axis=1, how='any')
    # 获得指标
    mid = design_m.drop(columns='trade_date')
    index = mid.columns
    # 获得矩阵
    design_m = np.array(mid)
    # 筛选前2000个指标
    fillter_index = (np.mean(design_m, axis=0)>-np.sort(-np.mean(design_m, axis=0))[2000])
    index = mid.columns[fillter_index]
    result_mat = design_m[:,fillter_index]
    return(result_mat, index)


def design_test_clip(design_m):
    # 填充NA为0
    design_m = design_m.fillna(value=0)
    # 获得矩阵
    design_m = np.array(design_m)
    return(design_m)


def proj(w):
    d = w.shape[0]
    sort_w = -np.sort(-w, axis=None)
    tmp = (np.cumsum(sort_w) - 1) * (1.0/np.arange(1,d+1))
    rho = np.sum(sort_w > tmp) - 1
    w = np.maximum(w - tmp[rho], 0)
    return w


def mean_var_opt(Sigma):
    d = Sigma.shape[0]

    eps = 1e-6
    delta = np.finfo(np.float64).eps
    while True:
        try:
            P = matrix(Sigma, tc='d')
            q = matrix(np.zeros(d), tc='d')
            G = matrix(-np.eye(d), tc='d')
            h = matrix(np.zeros(d), tc='d')
            A = matrix(np.ones(d).reshape((1, -1)), tc='d')
            b = matrix(np.ones(1), tc='d')
            # sol = solvers.qp(P, q, G, h, A, b, options={'show_progress': True,
            #                                             'abstol': 1e-12, 'reltol': 1e-11,
            #                                             'maxiters': int(1e4), 'feastol': 1e-16})
            sol = solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
            print(sol['iterations'])
            w = np.array(sol['x']).flatten()
            w[w <= delta] = 0.
            w = proj(w)
            break
        except:
            print('singular')
            Sigma = Sigma + np.identity(d) * eps
            eps *= 10

    return w


def cov(X, method):
    '''
    Parameters
    ----------
    X : np.array
        The sample matrix with size \(n, p\).
    method : str
        The method used to estimate the covariance.

    Returns
    ----------
    Cov : np.array
        The estimated covariance matrix.
    '''
    if method.startswith('GMV-P'):
        return np.cov(X, rowvar = False)
    elif method.startswith('GMV-LS'):
        cov = LedoitWolf(assume_centered = False).fit(X)
        return cov.covariance_
    elif method.startswith('GMV-NLS'):
        return nls.shrink_cov(X)


def eval(X_train, X_val, method):
    Sigma = cov(X_train, method)
    w = mean_var_opt(Sigma)
    return np.dot(X_val, w) - 1, w


# 主体拟合函数
def fit(i, design_M, method):
    # 获得用于训练的设计矩阵和用于测试的设计矩阵
    train_index = ((index_number[3 * i + 13] > design_M['trade_date']) & (
                index_number[3 * i + 7] < design_M['trade_date']))
    test_index = ((index_number[3 * i + 16] > design_M['trade_date']) & (
                index_number[3 * i + 13] < design_M['trade_date']))
    train_design_m = design_M.loc[train_index,]
    train_design_m, index = design_train_clip(train_design_m)
    test_design_m = design_M.loc[test_index, index]
    test_design_m = design_test_clip(test_design_m)
    # 创建ws_test用于保存权重
    ws_test = np.empty((1, design_M.shape[1]))
    ws_test.fill(0)
    # 中间转换为pd的frames便于索引操作。
    ws_test = pd.DataFrame(ws_test)
    ws_test.columns = design_M.columns  # 添加索引
    # 获得权重和得分
    score_test, ws_test.loc[0, index] = eval(train_design_m, test_design_m, method)
    ws_test = np.array(ws_test)

    return score_test, ws_test


# 读取设计矩阵
design_M = pd.read_csv('./code/data/design_matrix.csv', low_memory=False)
design_M = design_M.drop(columns='Unnamed: 0')


# 用于索引为期半年的训练矩阵和为期三个月的测试矩阵
index_number = [20200000,20200100,20200200,20200300,20200400,20200500,20200600,20200700,20200800,20200900,20201000,20201100,20201200,
                20210100,20210200,20210300,20210400,20210500,20210600,20210700,20210800,20210900,20211000,20221100,20211200,
                20220100,20220200,20220300,20220400,20220500,20220600,20220700,20220800,20220900,20221000,20221100,20221200,20221300]
itrs = 8


# 保存输出结果的路径
path_result = './code/result/'


""" GMV-P 方法"""
# mv方法设定
k = 0
method_list = ['GMV-P', 'GMV-LS', 'GMV-NLS']
method = method_list[k]
# 程序运行主体
from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    print(method)

    out = parallel(delayed(fit)(i, design_M, method) for i in range(itrs))
    score_test_list, ws_test_list = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)

    np.savez('./code/result/res_%s.npz'%(method),
            score_test_list=score_test_list, ws_test_list=ws_test_list)

np.sum(ws_test_list>0)

if itrs == False:
    sol = solvers.qp(P, q, G, h, A, b, options={'show_progress': False,
                                                'abstol': 1e-12, 'reltol': 1e-11,
                                                'maxiters': maxiters, 'feastol': 1e-16})
else:
    sol = solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    itr = sol['iterations']