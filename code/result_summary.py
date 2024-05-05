import numpy as np
import pandas as pd


# 载入数据
path = './code/result/'
# ew
res_ew = np.load(path+'res_ew'+'.npz')
# gmv
res_GMV_P = np.load(path+'res_GMV-P'+'.npz')
res_GMV_LS = np.load(path+'res_GMV-LS'+'.npz')
res_GMV_NLS = np.load(path+'res_GMV-NLS'+'.npz')
# mv
res_MV_P = np.load(path+'res_MV-P'+'.npz')
res_MV_LS = np.load(path+'res_MV-LS'+'.npz')
res_MV_NLS = np.load(path+'res_MV-NLS'+'.npz')
# spo
res_spo_LOG = np.load(path+'res_spo_LOG_1.00'+'.npz')
res_spo_EXP_05 = np.load(path+'res_spo_EXP_0.05'+'.npz')
res_spo_EXP_10 = np.load(path+'res_spo_EXP_0.10'+'.npz')
res_spo_EXP_50 = np.load(path+'res_spo_EXP_0.50'+'.npz')
res_spo_EXP_100 = np.load(path+'res_spo_EXP_1.00'+'.npz')
res_spo_EXP_150 = np.load(path+'res_spo_EXP_1.50'+'.npz')


# 获得得分数据
combined = np.concatenate(
    (res_ew['score_test_list'].reshape((-1,1)), res_GMV_P['score_test_list'].reshape((-1,1)),
           res_GMV_LS['score_test_list'].reshape((-1,1)), res_GMV_NLS['score_test_list'].reshape((-1,1)),
           res_MV_P['score_test_list'].reshape((-1,1)), res_MV_LS['score_test_list'].reshape((-1,1)),
           res_MV_NLS['score_test_list'].reshape((-1,1)), res_spo_LOG['score_test_list'].reshape((-1,1)),
           res_spo_EXP_05['score_test_list'].reshape((-1,1)), res_spo_EXP_10['score_test_list'].reshape((-1,1)),
           res_spo_EXP_50['score_test_list'].reshape((-1,1)), res_spo_EXP_100['score_test_list'].reshape((-1,1)),
           res_spo_EXP_150['score_test_list'].reshape((-1,1))), axis=1)
# 整理成列数据框
score = pd.DataFrame(combined, columns=['res_ew', 'res_GMV_P', 'res_GMV_LS', 'res_GMV_NLS',
                                        'res_MV_P', 'res_MV_LS', 'res_MV_NLS', 'res_spo_LOG',
                                        'res_spo_EXP_05', 'res_spo_EXP_10', 'res_spo_EXP_50',
                                        'res_spo_EXP_100', 'res_spo_EXP_150'])


# 获得权重数据
weights = [
    res_ew['ws_test_list'], res_GMV_P['ws_test_list'],
    res_GMV_LS['ws_test_list'], res_GMV_NLS['ws_test_list'],
    res_MV_P['ws_test_list'], res_MV_LS['ws_test_list'],
    res_MV_NLS['ws_test_list'], res_spo_LOG['ws_test_list'],
    res_spo_EXP_05['ws_test_list'], res_spo_EXP_10['ws_test_list'],
    res_spo_EXP_50['ws_test_list'], res_spo_EXP_100['ws_test_list'],
    res_spo_EXP_150['ws_test_list']
]


# 获得迭代次数数据
combined_itr = [
    res_GMV_P['itr_test_list'], res_GMV_LS['itr_test_list'],
    res_GMV_NLS['itr_test_list'], res_MV_P['itr_test_list'],
    res_MV_LS['itr_test_list'], res_MV_NLS['itr_test_list']]


# 迭代收敛性
itr_number = np.concatenate((
    combined_itr[0].reshape((-1,1)), combined_itr[1].reshape((-1,1)),
    combined_itr[2].reshape((-1,1)), combined_itr[3].reshape((-1,1)),
    combined_itr[4].reshape((-1,1)), combined_itr[5].reshape((-1,1))), axis=1)
itr_number = pd.DataFrame(itr_number, columns=['res_GMV_P', 'res_GMV_LS', 'res_GMV_NLS',
                                               'res_MV_P', 'res_MV_LS', 'res_MV_NLS'])


# 指标计算
# 获得累计收益
cumreturn = np.cumprod(score+1, axis=0)
cumreturn.iloc[-1,]
# 获得最大损失
max_draw = lambda x: np.max((np.maximum.accumulate(x) - x)/np.maximum.accumulate(x))
max_drawdown = cumreturn.apply(max_draw)
max_drawdown
# 获得夏普比率
sr = lambda x: np.mean(x)/np.std(x)
sharpe_ratio = score.apply(sr)
sharpe_ratio
# 获得资产数量
w_number = np.concatenate((
    np.sum(weights[0]>0, axis=1).reshape((-1,1)), np.sum(weights[1]>0, axis=1).reshape((-1,1)),
    np.sum(weights[2]>0, axis=1).reshape((-1,1)) ,np.sum(weights[3]>0, axis=1).reshape((-1,1)),
    np.sum(weights[4]>0, axis=1).reshape((-1,1)),np.sum(weights[5]>0, axis=1).reshape((-1,1)),
    np.sum(weights[6]>0, axis=1).reshape((-1,1)),np.sum(weights[7]>0, axis=1).reshape((-1,1)),
    np.sum(weights[8]>0, axis=1).reshape((-1,1)),np.sum(weights[9]>0, axis=1).reshape((-1,1)),
    np.sum(weights[10]>0, axis=1).reshape((-1,1)),np.sum(weights[11]>0, axis=1).reshape((-1,1)),
    np.sum(weights[12]>0, axis=1).reshape((-1,1))), axis=1)
w_number = pd.DataFrame(w_number, columns=['res_ew', 'res_GMV_P', 'res_GMV_LS', 'res_GMV_NLS',
                                        'res_MV_P', 'res_MV_LS', 'res_MV_NLS', 'res_spo_LOG',
                                        'res_spo_EXP_05', 'res_spo_EXP_10', 'res_spo_EXP_50',
                                        'res_spo_EXP_100', 'res_spo_EXP_150'])