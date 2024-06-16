import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math
import re


df = pd.read_csv('selected_intro_job.csv')
# 将相应字段数据保存至列表中
job_array = df['category'].values
education_array = df['education'].values
place_array = df['city'].values
salary_array = df['salary'].values
date_array = df['workYear'].values

print(job_array)
# ------------------------------------------------线性回归-------------------------------------------------------------------
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读取数据
df = pd.read_csv('selected_intro_job.csv')

# 提取特征和目标变量
X = df[['category', 'education', 'city', 'workYear']]
y = df['salary']

# 将分类变量编码为数值
label_encoders = {}
for col in ['category', 'education', 'city', 'workYear']:
    label_encoders[col] = LabelEncoder()
    # X[col] = label_encoders[col].fit_transform(X[col])
    X.loc[:, col] = label_encoders[col].fit_transform(X[col])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# 输入其他字段进行预测
job_input = 'Python'
education_input = '本科'
city_input = '北京'
workYear_input = 2.0

job_encoded = label_encoders['category'].transform([job_input])[0]
education_encoded = label_encoders['education'].transform([education_input])[0]
city_encoded = label_encoders['city'].transform([city_input])[0]
workYear_encoded = label_encoders['workYear'].transform([workYear_input])[0]

predicted_salary = model.predict([[job_encoded, education_encoded, city_encoded, workYear_encoded]])
# print('Predicted Salary: {:.2f}K'.format(predicted_salary[0]))
print(f'Predicted Salary: {predicted_salary[0]/12:.5f}K')

# ------------------------------------------------随机森林-------------------------------------------------------------------

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读取数据
df = pd.read_csv('selected_intro_job.csv')

# 提取特征和目标变量
X = df[['category', 'education', 'city', 'workYear']]
y = df['salary']

# 将分类变量编码为数值
label_encoders = {}
for col in ['category', 'education', 'city']:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# 输入其他字段进行预测
job_input = 'Python'
education_input = '本科'
city_input = '北京'
workYear_input = 2

job_encoded = label_encoders['category'].transform([job_input])[0]
education_encoded = label_encoders['education'].transform([education_input])[0]
city_encoded = label_encoders['city'].transform([city_input])[0]
workYear_encoded = workYear_input

predicted_salary = model.predict([[job_encoded, education_encoded, city_encoded, workYear_encoded]])
print(f'Predicted Salary: {predicted_salary[0]/12:.5f}K')

# ------------------------------------------------岭回归-------------------------------------------------------------------
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读取数据
df = pd.read_csv('selected_intro_job.csv')

# 提取特征和目标变量
X = df[['category', 'education', 'city', 'workYear']]
y = df['salary']

# 将分类变量编码为数值
label_encoders = {}
for col in ['category', 'education', 'city']:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练岭回归模型
model = Ridge()
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# 输入其他字段进行预测
job_input = 'Python'
education_input = '本科'
city_input = '北京'
workYear_input = 2.0

job_encoded = label_encoders['category'].transform([job_input])[0]
education_encoded = label_encoders['education'].transform([education_input])[0]
city_encoded = label_encoders['city'].transform([city_input])[0]
workYear_encoded = workYear_input

predicted_salary = model.predict([[job_encoded, education_encoded, city_encoded, workYear_encoded]])
print(f'Predicted Salary: {predicted_salary[0]/12:.5f}K')
# -----------------------------------------------支持向量-------------------------------------------------------------------

import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读取数据
df = pd.read_csv('selected_intro_job.csv')

# 提取特征和目标变量
X = df[['category', 'education', 'city', 'workYear']]
y = df['salary']

# 将分类变量编码为数值
label_encoders = {}
for col in ['category', 'education', 'city']:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练支持向量机回归模型
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# 输入其他字段进行预测
job_input = 'Python'
education_input = '本科'
city_input = '北京'
workYear_input = 2.0

job_encoded = label_encoders['category'].transform([job_input])[0]
education_encoded = label_encoders['education'].transform([education_input])[0]
city_encoded = label_encoders['city'].transform([city_input])[0]
workYear_encoded = workYear_input

predicted_salary = model.predict([[job_encoded, education_encoded, city_encoded, workYear_encoded]])
print(f'Predicted Salary: {predicted_salary[0]/12:.5f}K')

