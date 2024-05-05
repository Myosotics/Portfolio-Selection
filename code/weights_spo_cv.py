import pandas as pd
import numpy as np
from tqdm import tqdm


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


# 交叉验证的函数
def cross_v_spo(n_folds, lambdas, train_design_m, func, a):
    # 加载必要的包
    from sklearn.model_selection import TimeSeriesSplit
    from spo import spo_l1_path

    # 提供训练 / 测试索引以拆分在训练 / 测试集中以固定时间间隔观察到的时间序列数据样本。
    KF = TimeSeriesSplit(n_splits=n_folds, test_size=None)
    # 初始化用于判断超参数对应模型得分的得分向量
    n_lambdas = len(lambdas)
    score_val = np.zeros((n_folds, n_lambdas))
    # 一次交叉验证，用时1:28:41
    for j, (train_Index, val_Index) in tqdm(enumerate(KF.split(train_design_m)), total = n_folds):
        X_train, X_val = train_design_m[train_Index], train_design_m[val_Index]

        ws, _, _, _, _ = spo_l1_path(X_train, func, a, lambdas, None,
                                     screen=True, max_iter=int(1e4), f=30, tol=1e-5, verbose=False)

        # by returns
        # score_val[j] = np.where(np.sum(ws>0, axis=0)>0, np.mean(X_val @ ws, axis=0), -np.inf)

        # by sharpe ratio
        ret = np.log((X_val-1) @ ws + 1)
        sd = np.std(ret, axis=0)
        # 此时 sharpe ration 的计算中默认： 无风险利率 = 0
        score_val[j] = np.divide(np.mean(ret, axis=0), sd, out=np.zeros_like(sd), where=sd != 0)
    # 获得最佳超参数的id
    id_lam = np.argmax(np.median(score_val, axis=0))
    return id_lam


# 主体拟合函数
def fit(i, design_M, func, a, n_folds):
    # 加载必要的包
    from spo import spo_l1_path, spo_nw_min

    # 获得用于训练的设计矩阵和用于测试的设计矩阵
    train_index = ((index_number[3 * i + 13] > design_M['trade_date']) & (index_number[3 * i + 7] < design_M['trade_date']))
    test_index = ((index_number[3 * i + 16] > design_M['trade_date']) & (index_number[3 * i + 13] < design_M['trade_date']))
    train_design_m = design_M.loc[train_index,]
    train_design_m, index = design_train_clip(train_design_m)
    test_design_m = design_M.loc[test_index, index]
    test_design_m = design_test_clip(test_design_m)
    # 首先获得 lambdas 用于随后的交叉检验
    delta = 2.
    _, lambdas, _, _, _ = spo_l1_path(train_design_m, func, a, None, n_lambdas, screen=True, delta=delta,
                                      max_iter=int(0), verbose=False)
    # 设计交叉验证实验用于获得超参数，此处设计为 5折交叉验证。
    id_lam = cross_v_spo(n_folds, lambdas, train_design_m, func, a)
    # 返回第一个权重并非全为0的lambda的指标
    lambdas = lambdas[id_lam:id_lam + 5]
    ws, _, _, _, _ = spo_nw_min(train_design_m, func, a, lambdas, None,
                                screen=True, max_iter=int(1e5), f=30, tol=1e-8, nw_min=1)
    id_lams = np.where(np.sum(ws > 0, axis=0) > 0)[0]
    # 创建ws_test用于保存权重
    ws_test = np.empty((1, design_M.shape[1]))
    ws_test.fill(0)
    # 中间转换为pd的frames便于索引操作。
    ws_test = pd.DataFrame(ws_test)
    ws_test.columns = design_M.columns # 添加索引
    # 保存权重
    if len(id_lams) > 0:
        w = ws[:, id_lams[0]]
        score_test = np.dot(test_design_m, w) - 1
        ws_test.loc[0, index] = w
        ws_test = np.array(ws_test)
    else:
        score_test = [0.]
        ws_test.loc[0, index] = 0.
    print(i, np.sum(ws > 0, axis=0), np.cumprod(score_test + 1.))
    return score_test, ws_test, lambdas


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


# 效应函数试验列表
func_names = ['LOG', 'EXP']


# 保存输出结果的路径
path_result = './code/result/'

""" LOG-1.00 """
# 效应函数设定
func = 0
a = 1.
# Joblib是一个可以将Python代码转换为并行计算模式的包
from joblib import Parallel, delayed
# n_jobs=-1: 使用所有的CPU执行并行计算
# verbose: 信息级别:如果非零，则打印进度消息。超过50，输出被发送到stdout。消息的频率随着信息级别的增加而增加。如果大于10，则报告所有迭代。
with Parallel(n_jobs=-1, verbose=100) as parallel:
    out = parallel(delayed(fit)(i, design_M, func, a, n_folds) for i in range(itrs))
    # result = zip(a, b): 对于我们的两个list，a和b，list(zip(a, b))生成了一个列表。在这个列表中，每个元素是一个tuple；对于第i个元组，它其中的内容是(a[i-1], b[i-1])
    # origin = zip(*result): 前面加*号，事实上*号也是一个特殊的运算符，叫解包运算符, 可以得到原来的a和b
    score_test_list, ws_test_list, lambdas = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    lambdas = np.concatenate(lambdas, axis=0)
    np.savez(path_result+'res_spo_%s_%.2f.npz'%(func_names[func],a),
                score_test_list=score_test_list,
                ws_test_list=ws_test_list,
                lambdas=lambdas
            )
    ret = np.log(score_test_list+1)
    print(func, a, np.nancumprod(score_test_list+1)[-1], np.mean(ret)/np.std(ret))


""" EXP-0.05 """
# 效应函数设定
func = 1
a_list = [5e-2,1e-1,5e-1,1.,1.5]
k = 1
a = a_list[k-1]
# Joblib是一个可以将Python代码转换为并行计算模式的包
from joblib import Parallel, delayed
# n_jobs=-1: 使用所有的CPU执行并行计算
# verbose: 信息级别:如果非零，则打印进度消息。超过50，输出被发送到stdout。消息的频率随着信息级别的增加而增加。如果大于10，则报告所有迭代。
with Parallel(n_jobs=-1, verbose=100) as parallel:
    out = parallel(delayed(fit)(i, design_M, func, a, n_folds) for i in range(itrs))
    # result = zip(a, b): 对于我们的两个list，a和b，list(zip(a, b))生成了一个列表。在这个列表中，每个元素是一个tuple；对于第i个元组，它其中的内容是(a[i-1], b[i-1])
    # origin = zip(*result): 前面加*号，事实上*号也是一个特殊的运算符，叫解包运算符, 可以得到原来的a和b
    score_test_list, ws_test_list, lambdas = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    lambdas = np.concatenate(lambdas, axis=0)
    np.savez(path_result+'res_spo_%s_%.2f.npz'%(func_names[func],a),
                score_test_list=score_test_list,
                ws_test_list=ws_test_list,
                lambdas=lambdas
            )
    ret = np.log(score_test_list+1)
    print(func, a, np.nancumprod(score_test_list+1)[-1], np.mean(ret)/np.std(ret))



""" EXP-0.10 """
# 效应函数设定
func = 1
a_list = [5e-2,1e-1,5e-1,1.,1.5]
k = 2
a = a_list[k-1]
# Joblib是一个可以将Python代码转换为并行计算模式的包
from joblib import Parallel, delayed
# n_jobs=-1: 使用所有的CPU执行并行计算
# verbose: 信息级别:如果非零，则打印进度消息。超过50，输出被发送到stdout。消息的频率随着信息级别的增加而增加。如果大于10，则报告所有迭代。
with Parallel(n_jobs=-1, verbose=100) as parallel:
    out = parallel(delayed(fit)(i, design_M, func, a, n_folds) for i in range(itrs))
    # result = zip(a, b): 对于我们的两个list，a和b，list(zip(a, b))生成了一个列表。在这个列表中，每个元素是一个tuple；对于第i个元组，它其中的内容是(a[i-1], b[i-1])
    # origin = zip(*result): 前面加*号，事实上*号也是一个特殊的运算符，叫解包运算符, 可以得到原来的a和b
    score_test_list, ws_test_list, lambdas = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    lambdas = np.concatenate(lambdas, axis=0)
    np.savez(path_result+'res_spo_%s_%.2f.npz'%(func_names[func],a),
                score_test_list=score_test_list,
                ws_test_list=ws_test_list,
                lambdas=lambdas
            )
    ret = np.log(score_test_list+1)
    print(func, a, np.nancumprod(score_test_list+1)[-1], np.mean(ret)/np.std(ret))


""" EXP-0.50 """
# 效应函数设定
func = 1
a_list = [5e-2,1e-1,5e-1,1.,1.5]
k = 3
a = a_list[k-1]
# Joblib是一个可以将Python代码转换为并行计算模式的包
from joblib import Parallel, delayed
# n_jobs=-1: 使用所有的CPU执行并行计算
# verbose: 信息级别:如果非零，则打印进度消息。超过50，输出被发送到stdout。消息的频率随着信息级别的增加而增加。如果大于10，则报告所有迭代。
with Parallel(n_jobs=-1, verbose=100) as parallel:
    out = parallel(delayed(fit)(i, design_M, func, a, n_folds) for i in range(itrs))
    # result = zip(a, b): 对于我们的两个list，a和b，list(zip(a, b))生成了一个列表。在这个列表中，每个元素是一个tuple；对于第i个元组，它其中的内容是(a[i-1], b[i-1])
    # origin = zip(*result): 前面加*号，事实上*号也是一个特殊的运算符，叫解包运算符, 可以得到原来的a和b
    score_test_list, ws_test_list, lambdas = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    lambdas = np.concatenate(lambdas, axis=0)
    np.savez(path_result+'res_spo_%s_%.2f.npz'%(func_names[func],a),
                score_test_list=score_test_list,
                ws_test_list=ws_test_list,
                lambdas=lambdas
            )
    ret = np.log(score_test_list+1)
    print(func, a, np.nancumprod(score_test_list+1)[-1], np.mean(ret)/np.std(ret))


""" EXP-1.00 """
# 效应函数设定
func = 1
a_list = [5e-2,1e-1,5e-1,1.,1.5]
k = 4
a = a_list[k-1]
# Joblib是一个可以将Python代码转换为并行计算模式的包
from joblib import Parallel, delayed
# n_jobs=-1: 使用所有的CPU执行并行计算
# verbose: 信息级别:如果非零，则打印进度消息。超过50，输出被发送到stdout。消息的频率随着信息级别的增加而增加。如果大于10，则报告所有迭代。
with Parallel(n_jobs=-1, verbose=100) as parallel:
    out = parallel(delayed(fit)(i, design_M, func, a, n_folds) for i in range(itrs))
    # result = zip(a, b): 对于我们的两个list，a和b，list(zip(a, b))生成了一个列表。在这个列表中，每个元素是一个tuple；对于第i个元组，它其中的内容是(a[i-1], b[i-1])
    # origin = zip(*result): 前面加*号，事实上*号也是一个特殊的运算符，叫解包运算符, 可以得到原来的a和b
    score_test_list, ws_test_list, lambdas = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    lambdas = np.concatenate(lambdas, axis=0)
    np.savez(path_result+'res_spo_%s_%.2f.npz'%(func_names[func],a),
                score_test_list=score_test_list,
                ws_test_list=ws_test_list,
                lambdas=lambdas
            )
    ret = np.log(score_test_list+1)
    print(func, a, np.nancumprod(score_test_list+1)[-1], np.mean(ret)/np.std(ret))


""" EXP-1.50 """
# 效应函数设定
func = 1
a_list = [5e-2,1e-1,5e-1,1.,1.5]
k = 5
a = a_list[k-1]
# Joblib是一个可以将Python代码转换为并行计算模式的包
from joblib import Parallel, delayed
# n_jobs=-1: 使用所有的CPU执行并行计算
# verbose: 信息级别:如果非零，则打印进度消息。超过50，输出被发送到stdout。消息的频率随着信息级别的增加而增加。如果大于10，则报告所有迭代。
with Parallel(n_jobs=-1, verbose=100) as parallel:
    out = parallel(delayed(fit)(i, design_M, func, a, n_folds) for i in range(itrs))
    # result = zip(a, b): 对于我们的两个list，a和b，list(zip(a, b))生成了一个列表。在这个列表中，每个元素是一个tuple；对于第i个元组，它其中的内容是(a[i-1], b[i-1])
    # origin = zip(*result): 前面加*号，事实上*号也是一个特殊的运算符，叫解包运算符, 可以得到原来的a和b
    score_test_list, ws_test_list, lambdas = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    lambdas = np.concatenate(lambdas, axis=0)
    np.savez(path_result+'res_spo_%s_%.2f.npz'%(func_names[func],a),
                score_test_list=score_test_list,
                ws_test_list=ws_test_list,
                lambdas=lambdas
            )
    ret = np.log(score_test_list+1)
    print(func, a, np.nancumprod(score_test_list+1)[-1], np.mean(ret)/np.std(ret))