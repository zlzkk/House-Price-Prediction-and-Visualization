import pandas as pd
import re

# 读取 CSV 文件
data = pd.read_csv('intro_job.csv')


total_rows = data.shape[0]
#---------------------------------------------------------数据清洗与预处理------------------------------------------------------------------
# 保留特定的列
selected_columns = ['id', 'category', 'city', 'salary', 'workYear', 'education']
data = data.loc[:, selected_columns]
salary_column = data['salary']
# 定义一个函数，将薪资区间转换为平均值
def process_salary(salary_range):
    try:
        salary_values = salary_range.split('-')
        low_salary = int(salary_values[0].replace('k', ''))
        high_salary = int(salary_values[1].replace('k', ''))
        return (low_salary + high_salary) / 2
    except Exception as e:
        print(f"Ignoring row due to error: {e}")
        return None

# # 将薪资区间转换为平均值
data['salary'] = salary_column.apply(process_salary)
# # 删除包含错误的整行
data = data.dropna()
# 定义一个函数，将年份区间转换为平均值
def process_workYear(workYear_range):
    try:
        workYear_values = workYear_range.split('-')
        low_workYear = int(workYear_values[0][0])
        high_workYear = int(workYear_values[1][0])
        return (low_workYear + high_workYear) / 2
    except:
        return None
# 将年份区间转换为平均值
data['workYear'] = data['workYear'].apply(process_workYear)

# 删除包含 NaN 值的行
data.dropna(subset=['workYear'], inplace=True)
data.to_csv('selected_intro_job.csv', index=False)
# 打印结果
print(data)

