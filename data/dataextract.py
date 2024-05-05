# 导入tushare
import tushare as ts

# 初始化pro接口
pro = ts.pro_api('20231121155429-e964e3b1-b328-4446-ab75-c98dea17f293')
pro._DataApi__http_url = 'http://tsapi.majors.ltd:7000'

# 常规接口
c1 = '20211231'
df1 = pro.daily(trade_date=c1)
df2 = pro.daily_basic(trade_date=c1, fields='ts_code,trade_date,close,dv_ratio,dv_ttm')
print(df1)
print(df2)
df1.columns
df1.to_csv("df1.csv")

df = pro.daily(trade_date='20180810')
df.to_csv("df.csv")


df3 = pro.daily_basic(ts_code='600230.SH', fields='ts_code,trade_date,close,dv_ratio,dv_ttm')
print(df3)
df3['close'] * df3['dv_ttm']
df3['close'] * df3['dv_ratio']
# 日期字符获得函数

# sub-function
def if_runnian(year):
    if year % 4 == 0 and year % 100 != 0 or year % 400 == 0:
        logis = 1
        Febdate = 29
        return logis,Febdate
    else:
        logis = 0
        Febdate = 28
        return logis, Febdate

def monthdate(year, month, monthnumber):
    result = []
    yearstr = str(year)
    if month < 10 :
        monthstring = '0'+ str(month)
    else:
        monthstring = str(month)
    numbers1 = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    numbers2 = list(range(10, 33))
    numbers2 = list(map(str, numbers2))
    numbers1.extend(numbers2)
    for i in range(monthnumber):
        string = yearstr + monthstring + numbers1[i]
        result.append(string)
    return result

# main function
def datestr(year):
    runyear = if_runnian(year)
    monthnumber = [31, runyear[1], 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    result = []
    for i in range(12):
        month = i+1
        result.extend(monthdate(year, month, monthnumber[i]))
    return result


# 提取2020年数据
year = 2020

dateindex = datestr(year)

for i in range(len(dateindex)):
    path = './data/' + str(year) + '/' + dateindex[i] +'.csv'
    df = pro.daily(trade_date=dateindex[i])
    df.to_csv(path)


# 提取2021年数据
year = 2021

dateindex = datestr(2021)

for i in range(len(dateindex)):
    path = './data/' + str(year) + '/' + dateindex[i] +'.csv'
    df = pro.daily(trade_date=dateindex[i])
    df.to_csv(path)


# 提取2022年数据
year = 2022

dateindex = datestr(year)

for i in range(len(dateindex)):
    path = './data/' + str(year) + '/' + dateindex[i] +'.csv'
    df = pro.daily(trade_date=dateindex[i])
    df.to_csv(path)

