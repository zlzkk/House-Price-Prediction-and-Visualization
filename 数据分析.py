import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LassoCV
import numpy as np
if __name__ == '__main__':
    df = pd.read_csv('cs_hours_data.csv', encoding='gbk')
    print(f'样本量共有 {df.shape[0]} 个')
    df.duplicated().sum()
    print(df.isnull().sum())
    print(df.dtypes)
    print(df['朝向'].unique())
    print(df['楼层'].unique())
    print(df['装修'].unique())
    print(df['产权性质'].unique())
    print(df['住宅类别'].unique())
    print(df['建筑结构'].unique())
    print(df['建筑类别'].unique())
    print(df['区域'].unique())
    print(df['建筑年代'].unique())
    df.replace('暂无', np.nan, inplace=True)
    df['建筑面积'] = df['建筑面积'].map(lambda x: x.replace('平米', '')).astype('float')
    df['单价'] = df['单价'].map(lambda x: x.replace('元/平米', '')).astype('float')


    def process_year(year):
        if year is not None:
            year = str(year)[:4]
        return year


    df['建筑年代'] = df['建筑年代'].map(process_year)
    floor = {'低楼层': '低', '中楼层': '中', '高楼层': '高', '低层': '低', '中层': '中', '高层': '高'}
    df['楼层'] = df['楼层'].map(floor)


    def process_area(area):
        if area != '新区':
            area = area.replace('区', '').replace('县', '')
        return area


    df['区域'] = df['区域'].map(process_area)
    df.replace('nan', np.nan, inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df.info())
    # 户型、朝向、楼层处理，缺失值数量不多，考虑直接删除即可
    df.dropna(subset=['户型', '朝向', '楼层'], inplace=True)
    # 建筑年代、建筑类别、建筑结构、住宅类别、产权性质、装修 这些离散型变量很难处理，得根据实际情况填充，为了得到更加真实的结果暂不处理
    # 电梯处理（底层无，高层有，中层随机处理）
    df.loc[(df['楼层'] == '高') & (df['电梯'].isnull()), '电梯'] = '有 '
    df.loc[(df['楼层'] == '低') & (df['电梯'].isnull()), '电梯'] = '无 '
    df.loc[(df['楼层'] == '中') & (df['电梯'].isnull()), '电梯'] = random.choice(['有 ', '无 '])
    df.reset_index(drop=True, inplace=True)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    df.boxplot(column=['建筑面积'], flierprops={'markeredgecolor': 'red', 'markersize': 4}, ax=ax[0],fontsize=20)

    df.boxplot(column=['总价'], flierprops={'markeredgecolor': 'red', 'markersize': 4}, ax=ax[1])
    plt.show()
    # print(df.describe())
    # df.drop(index=df[df['总价'] > 200].index, inplace=True)
    # fig, ax = plt.subplots(3, 1, figsize=(8, 18))
    # x = df['区域'].unique()
    # # 各区单价对比
    # y1 = round(df.groupby(by=['区域'])['单价'].mean().sort_values(ascending=False), 2)
    # sns.barplot(x, y1, ax=ax[0], palette='Blues_r')
    # ax[0].set_title('长沙各县\区房产平均单价对比')
    #
    # # 各区总价对比
    # y2 = round(df.groupby(by=['区域'])['总价'].mean().sort_values(ascending=False), 2)
    # sns.barplot(x, y2, ax=ax[1], palette='BuGn_r')
    # ax[1].set_title('长沙各县\区房产平均总价对比')
    #
    # # 各区房子数量对比
    # y3 = round(df.groupby(by=['区域']).size().sort_values(ascending=False), 2)
    # sns.barplot(x, y3, ax=ax[2], palette='Oranges_d')
    # ax[2].set_title('长沙各县\区房产数量对比')
    # ax[2].set_ylabel('数量')
    # # 删除所有缺失值
    # d1 = df.dropna().reset_index(drop=True)
    # plt.figure(figsize=(11, 10))
    # sns.scatterplot(x='建筑面积', y='总价', data=df, s=14)
    #
    # plt.figure(figsize=(10, 8))
    # my_order = df.groupby(by=["朝向"])["总价"].median().sort_values(ascending=False).index
    # sns.boxplot(x='朝向', y='总价', data=df, width=0.5, notch=True, order=my_order)
    # plt.figure(figsize=(10, 8))
    # my_order = df.groupby(by=["装修"])["总价"].median().sort_values(ascending=False).index
    # sns.boxplot(x='装修', y='总价', data=df, width=0.4, notch=True, order=my_order)
    # plt.figure(figsize=(10, 8))
    # my_order = df.groupby(by=["楼层"])["总价"].median().sort_values(ascending=False).index
    # sns.boxplot(x='楼层', y='总价', data=df, width=0.3, notch=True, order=my_order)
    # plt.figure(figsize=(10, 8))
    # my_order = df.groupby(by=["电梯"])["总价"].median().sort_values(ascending=False).index
    # sns.boxplot(x='电梯', y='总价', data=df, width=0.2, notch=True, order=my_order)
    # plt.figure(figsize=(10, 8))
    # my_order = df.groupby(by=["学校"])["总价"].median().sort_values(ascending=False).index
    # sns.boxplot(x='学校', y='总价', data=df, width=0.2, notch=True, order=my_order)
    # plt.figure(figsize=(16, 8))
    order = sorted(df['建筑年代'].value_counts().index)
    sns.countplot(x=df['建筑年代'], order=order)
    plt.figure(figsize=(13, 8))
    my_order = df.groupby(by=["建筑年代"])["总价"].size().sort_values(ascending=False).index[:20]
    sns.boxplot(x='建筑年代', y='总价', data=df, width=0.2, notch=True, order=my_order)
    plt.figure(figsize=(10, 8))
    my_order = df.groupby(by=["产权性质"])["总价"].median().sort_values(ascending=False).index
    sns.boxplot(x='产权性质', y='总价', data=df, width=0.2, order=my_order)
    plt.figure(figsize=(10, 8))
    my_order = df.groupby(by=["住宅类别"])["总价"].median().sort_values(ascending=False).index
    sns.boxplot(x='住宅类别', y='总价', data=df, width=0.2, order=my_order)
    plt.figure(figsize=(10, 8))
    my_order = df.groupby(by=["建筑结构"])["总价"].median().sort_values(ascending=False).index
    sns.boxplot(x='建筑结构', y='总价', data=df, width=0.2, order=my_order)

    plt.figure(figsize=(10, 8))
    my_order = df.groupby(by=["建筑类别"])["总价"].median().sort_values(ascending=False).index
    sns.boxplot(x='建筑类别', y='总价', data=df, width=0.2, order=my_order)
    plt.figure(figsize=(15, 15))
    my_order = df.groupby(by=["户型"])["总价"].median().sort_values().index
    sns.boxplot(y='户型', x='总价', data=df, width=0.2, order=my_order)
    plt.figure(figsize=(15, 15))

    order = df['户型'].value_counts(ascending=False).index
    sns.countplot(y=df['户型'], order=order)
    plt.show()
