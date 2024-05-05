# 获得设计矩阵
import pandas as pd
import numpy as np
import os
import time

# 2022年数据

year = 2022

def design_matrix(folder_path, delta = 1e-07):
    flag = 0
    # 遍历文件夹下所有的文件
    time_start = time.time()
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # 如果是csv文件，打开文件并存储为dataframe
            with open(os.path.join(folder_path, filename), 'r', newline='') as csvfile:
                csv_data = pd.read_csv(csvfile, low_memory=False)  # 防止弹出警告
                csv_df = pd.DataFrame(csv_data)
                if csv_df.empty:
                    # 若dataframe为空，则循环继续
                    continue
                else:
                    # 若非空，首先计算 price realtive。
                    value = csv_df['close'] / (csv_df['open'] + delta)
                    # 保存为数据框，实际为设计矩阵的每一行
                    df = pd.DataFrame(np.array(value).reshape(1, -1))
                    # 添加列名：即股票编码名字
                    df.columns = csv_df['ts_code']
                    # 添加trade_date列并将其设置为首列
                    df.loc[:, 'trade_date'] = csv_df['trade_date'][0]
                    df = pd.concat([df['trade_date'], df.drop(columns='trade_date')], axis=1)

                    # 进行连接合并数据
                    if flag==0:
                        result = df.copy()
                    else:
                        time_start_for = time.time()
                        result = pd.concat([result, df], ignore_index=True, sort=False)
                        time_end_for = time.time()
                        time_sum_for = time_end_for - time_start_for
                        print(time_sum_for)
                    flag += 1
                    print(flag)
    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)
    return result


folder_path = './data/' + str(year)

time_start1 = time.time()  # 记录开始时间
matrix = design_matrix(folder_path)
time_end1 = time.time()  # 记录结束时间
time_sum1 = time_end1 - time_start1  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum1)

matrix.to_csv('./code/data/' + str(year) +'.csv')


# 2021年数据
year = 2021

folder_path = './data/' + str(year)

time_start1 = time.time()  # 记录开始时间
matrix = design_matrix(folder_path)
time_end1 = time.time()  # 记录结束时间
time_sum1 = time_end1 - time_start1  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum1)

matrix.to_csv('./code/data/' + str(year) +'.csv')


# 2020年数据
year = 2020

folder_path = './data/' + str(year)

time_start1 = time.time()  # 记录开始时间
matrix = design_matrix(folder_path)
time_end1 = time.time()  # 记录结束时间
time_sum1 = time_end1 - time_start1  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum1)

matrix.to_csv('./code/data/' + str(year) +'.csv')


# 获得最终设计矩阵
matrix1 = pd.read_csv('./code/data/2020.csv', low_memory=False)
matrix2 = pd.read_csv('./code/data/2021.csv', low_memory=False)
matrix3 = pd.read_csv('./code/data/2022.csv', low_memory=False)
design_M = pd.concat([matrix1, matrix2,matrix3], ignore_index=True, sort=False)
design_M = design_M.drop(columns='Unnamed: 0')
design_M.to_csv('./code/data/design_matrix.csv')