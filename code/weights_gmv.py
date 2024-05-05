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


def mean_var_opt(Sigma, maxiters = int(1e4)):
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
            sol = solvers.qp(P, q, G, h, A, b, options={'show_progress': False, 'maxiters': maxiters})
            itr = sol['iterations']
            sol = solvers.qp(P, q, G, h, A, b, options={'show_progress': False,
                                                        'abstol': 1e-12, 'reltol': 1e-11,
                                                        'maxiters': maxiters, 'feastol': 1e-16})
            w = np.array(sol['x']).flatten()
            w[w <= delta] = 0.
            w = proj(w)
            break
        except:
            print('singular')
            Sigma = Sigma + np.identity(d) * eps
            eps *= 10

    return w, itr


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


def eval(X_train, X_val, method, maxiters):
    Sigma = cov(X_train, method)
    w, itr = mean_var_opt(Sigma, maxiters)
    return np.dot(X_val, w) - 1, w, itr


# 主体拟合函数
def fit(i, design_M, method, maxiters):
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
    score_test, ws_test.loc[0, index], itr = eval(train_design_m, test_design_m, method, maxiters)
    ws_test = np.array(ws_test)

    return score_test, ws_test, itr


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
# 最大迭代次数
maxiters = int(1e4)


""" GMV-P 方法"""
# mv方法设定
k = 0
method_list = ['GMV-P', 'GMV-LS', 'GMV-NLS']
method = method_list[k]
# 程序运行主体
from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    print(method)

    out = parallel(delayed(fit)(i, design_M, method, maxiters) for i in range(itrs))
    score_test_list, ws_test_list, itr_test_list = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    itr_test_list = np.array(itr_test_list)

    np.savez('./code/result/res_%s.npz'%(method),
            score_test_list=score_test_list, ws_test_list=ws_test_list, itr_test_list = itr_test_list)


""" GMV-LS 方法"""
# mv方法设定
k = 1
method_list = ['GMV-P', 'GMV-LS', 'GMV-NLS']
method = method_list[k]
# 程序运行主体
from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    print(method)

    out = parallel(delayed(fit)(i, design_M, method, maxiters) for i in range(itrs))
    score_test_list, ws_test_list, itr_test_list = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    itr_test_list = np.array(itr_test_list)

    np.savez('./code/result/res_%s.npz'%(method),
            score_test_list=score_test_list, ws_test_list=ws_test_list, itr_test_list = itr_test_list)


""" GMV-NLS 方法"""
# mv方法设定
k = 2
method_list = ['GMV-P', 'GMV-LS', 'GMV-NLS']
method = method_list[k]
# 程序运行主体
from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    print(method)

    out = parallel(delayed(fit)(i, design_M, method, maxiters) for i in range(itrs))
    score_test_list, ws_test_list, itr_test_list = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    itr_test_list = np.array(itr_test_list)

    np.savez('./code/result/res_%s.npz'%(method),
            score_test_list=score_test_list, ws_test_list=ws_test_list, itr_test_list = itr_test_list)