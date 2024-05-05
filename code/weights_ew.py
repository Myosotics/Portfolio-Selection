import pandas as pd
import numpy as np


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


# 主体拟合函数
def fit(i, design_M):
    # 获得用于训练的设计矩阵和用于测试的设计矩阵
    train_index = ((index_number[3 * i + 13] > design_M['trade_date']) & (index_number[3 * i + 7] < design_M['trade_date']))
    test_index = ((index_number[3 * i + 16] > design_M['trade_date']) & (index_number[3 * i + 13] < design_M['trade_date']))
    train_design_m = design_M.loc[train_index,]
    train_design_m, index = design_train_clip(train_design_m)
    test_design_m = design_M.loc[test_index, index]
    test_design_m = design_test_clip(test_design_m)
    # 获得权重
    score_test = np.mean(test_design_m-1, axis=1)
    ws = 1. / len(index)
    # 创建ws_test用于保存权重
    ws_test = np.empty((1, design_M.shape[1]))
    ws_test.fill(0)
    # 中间转换为pd的frames便于索引操作。
    ws_test = pd.DataFrame(ws_test)
    ws_test.columns = design_M.columns # 添加索引
    # 保存权重
    ws_test.loc[0, index] = ws
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


# 超参数固定参数设定
n_lambdas = 100
n_folds = 5


# 保存输出结果的路径
path_result = './code/result/'


""" EW """
# 方法
method = 'ew'
# 程序运行主体
from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    out = parallel(delayed(fit)(i, design_M) for i in range(itrs))
    score_test_list, ws_test_list = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)

    np.savez('./code/result/res_%s.npz'%(method),
            score_test_list=score_test_list, ws_test_list=ws_test_list)