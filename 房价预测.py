import pandas as pd
import matplotlib.pyplot as plt
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
# 读取CSV文件
df11 = pd.read_csv('cs_hours_data.csv',encoding='gbk')

# 随机选择100条数据
random_data = df11.sample(n=100, random_state=10)  # 设置随机种子以确保结果可重复



    # 对 apply 进行预测并输出结果
    # 然后在这里添加模型的预测和输出代码

    # 对 apply 进行预测并输出结果
    # 然后在这里添加模型的预测和输出代码

# 保留指定列名的数据

selected_data = random_data[['户型', '建筑面积', '朝向', '学校','总价']]
selected_data = random_data[random_data['总价'] <= 200]
selected_data.loc[:, '建筑面积'] = selected_data['建筑面积'].str.replace('平米', '').astype(float)
df11 = pd.DataFrame(selected_data)
#
# # 提取总价列作为y轴数据
# y = df['总价']
# print(y)
# # 创建x轴数据，即样本数量，范围从1到样本数量（在这里是5）
# x = range(1, len(df) + 1)
#
# # 创建折线图
# plt.plot(x, y)
#
# # 添加标题和标签
# plt.title('Total Price vs. Sample Number')
# plt.xlabel('Sample Number')
# plt.ylabel('Total Price')
#
# # 显示图形
# plt.show()

df = pd.read_csv('cs_hours_data.csv', encoding='gbk')
df.duplicated().sum()
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
# 户型、朝向、楼层处理，缺失值数量不多，考虑直接删除即可
df.dropna(subset=['户型', '朝向', '楼层'], inplace=True)
# 建筑年代、建筑类别、建筑结构、住宅类别、产权性质、装修 这些离散型变量很难处理，得根据实际情况填充，为了得到更加真实的结果暂不处理
# 电梯处理（底层无，高层有，中层随机处理）
df.loc[(df['楼层'] == '高') & (df['电梯'].isnull()), '电梯'] = '有 '
df.loc[(df['楼层'] == '低') & (df['电梯'].isnull()), '电梯'] = '无 '
df.loc[(df['楼层'] == '中') & (df['电梯'].isnull()), '电梯'] = random.choice(['有 ', '无 '])
df.reset_index(drop=True, inplace=True)
df.drop(index=df[df['总价'] > 200].index, inplace=True)
df.to_excel('cs_house.xlsx', index=False)
# 删除所有缺失值
d1 = df.dropna().reset_index(drop=True)


def apart_room(x):
    room = x.split('室')[0]
    return int(room)


def apart_hall(x):
    hall = x.split('厅')[0].split('室')[1]
    return int(hall)

    return int(wc)


d1['室'] = d1['户型'].map(apart_room)
d1['厅'] = d1['户型'].map(apart_hall)

# 删除楼层、户型、单价
d1.drop(columns=['户型', '楼层', '单价'], inplace=True)
# 编码-有序多分类（根据上面可视化的结果，按照对价格的影响程度排序，越大影响越高）
# 无序多分类无法直接引入，必须“哑元”化变量

# 等级变量（有序多分类）可以直接引入模型
map1 = {'南': 5, '南北': 6, '北': 1, '西南': 10, '东西': 4, '东': 2, '东北': 8, '东南': 9, '西': 3, '西北': 7}
d1['朝向'] = d1['朝向'].map(map1)
map2 = {'毛坯': 1, '简装修': 2, '精装修': 3, '中装修': 4, '豪华装修': 5}
d1['装修'] = d1['装修'].map(map2)
map3 = {'有 ': 1, '无 ': 0}
d1['电梯'] = d1['电梯'].map(map3)
map4 = {'商品房': 6, '个人产权': 5, '商品房(免税)': 7, '普通商品房': 4, '经济适用房': 2, '房改房': 3, '限价房': 8,
        '房本房': 1}
d1['产权性质'] = d1['产权性质'].map(map4)
map5 = {'普通住宅': 4, '经济适用房': 3, '公寓': 1, '商住楼': 2, '酒店式公寓': 5}
d1['住宅类别'] = d1['住宅类别'].map(map5)
map6 = {'平层': 4, '开间': 2, '跃层': 5, '错层': 1, '复式': 3}
d1['建筑结构'] = d1['建筑结构'].map(map6)
map7 = {'板楼': 4, '钢混': 5, '塔板结合': 3, '平房': 6, '砖混': 1, '塔楼': 7, '砖楼': 2}
d1['建筑类别'] = d1['建筑类别'].map(map7)
map8 = {'长沙': 9, '岳麓': 8, '雨花': 7, '望城': 6, '天心': 5, '宁乡': 4, '浏阳': 3, '开福': 2, '芙蓉': 1}
d1['区域'] = d1['区域'].map(map8)
# 删除超过2019年的房子，年代转变为房龄
d1['建筑年代'] = d1['建筑年代'].astype('int32')
d1.drop(index=d1[d1['建筑年代'] > 2019].index, inplace=True)
d1['房龄'] = d1['建筑年代'].map(lambda x: 2020 - x)
d1.drop(columns=['建筑年代'], inplace=True)
X = d1.drop(columns=['总价'])
y = d1['总价']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
poly = PolynomialFeatures(degree=2)
x_train = poly.fit_transform(X_train.values)
x_test = poly.fit_transform(X_test)
X = d1.drop(columns=['总价'])
y = d1['总价']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
poly = PolynomialFeatures(degree=2)
x_train = poly.fit_transform(X_train.values)
x_test = poly.fit_transform(X_test)
X = d1.drop(columns=['总价'])
y = d1['总价']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
poly = PolynomialFeatures(degree=2)
x_train = poly.fit_transform(X_train.values)
x_test = poly.fit_transform(X_test)


# class LassoRegression:
#     def __init__(self, lr=0.001, alpha=0.01, num_epochs=100):
#         self.lr = lr
#         self.alpha = alpha
#         self.num_epochs = num_epochs
#
#     def fit(self, X, y):
#         self.theta = np.zeros(X.shape[1])
#         self.loss_history = []
#
#         for epoch in range(self.num_epochs):
#             y_pred = np.dot(X, self.theta)
#             loss = np.mean((y_pred - y) ** 2) / 2 + self.alpha * np.sum(np.abs(self.theta))
#             self.loss_history.append(loss)
#
#             gradient = np.dot(X.T, (y_pred - y)) / len(y) + self.alpha * np.sign(self.theta)
#             self.theta -= self.lr * gradient
#
#             if epoch % 10 == 0:
#                 print(f"Epoch {epoch}, Loss: {loss}")
#
#     def predict(self, X):
#         return np.dot(X, self.theta)


# 训练模型
# model = LassoRegression(lr=0.00000001002, alpha=0.01, num_epochs=1000000)
# model.fit(x_train, y_train)

# 绘制损失曲线
# plt.plot(range(len(model.loss_history)), model.loss_history)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()
# from sklearn.metrics import r2_score

# 计算训练集和测试集的R²分数

la = Lasso(alpha=0.01, max_iter=10000)
la.fit(x_train, y_train)
kn = KNeighborsRegressor(n_neighbors=20)
kn.fit(x_train, y_train)
dt = DecisionTreeRegressor(max_depth=6)
dt.fit(x_train, y_train)
rf = RandomForestRegressor()
rf.fit(x_train, y_train)

# 映射朝向选项
direction_mapping = {
    '南': 5, '南北': 6, '北': 1, '西南': 10, '东西': 4,
    '东': 2, '东北': 8, '东南': 9, '西': 3, '西北': 7
}
selected_data['朝向'] = selected_data['朝向'].map(direction_mapping)

# 为了方便处理，我们假设 '户型' 列中的数据格式为 'x室x厅'，并分别提取出室和厅
selected_data[['室', '厅']] = selected_data['户型'].str.extract(r'(\d+)室(\d+)厅').astype(int)

# 迭代 selected_data 中的每一行，构建输入特征向量并进行预测
l1=[]
l2=[]
l3=[]
l4=[]
l5=[]
for index, row in selected_data.iterrows():
    # 获取建筑面积、学校、朝向、室和厅的值
    x1 = row['建筑面积']
    x2 = row['学校']
    x3 = row['朝向']
    x4 = row['室']
    x5 = row['厅']

    # 构建输入特征向量
    apply = np.array([x1, 10, 4, 0, 5, 4, 4, 5, 6, x2, x4, x5, x3]).reshape(1, -1)
    poly_apply = poly.fit_transform(apply)
    l1.append(round(la.predict(poly_apply)[0], 2))
    l2.append(round(rf.predict(poly_apply)[0], 2))
    l3.append(round(dt.predict(poly_apply)[0], 2))
    l4.append(round(kn.predict(poly_apply)[0], 2))
    l5.append(round(
    ((la.predict(poly_apply) + rf.predict(poly_apply) + dt.predict(poly_apply) + kn.predict(poly_apply)) / 4.0)[0],
    2))
# selected_data = random_data[['户型', '建筑面积', '朝向', '学校','总价']]
# selected_data = random_data[random_data['总价'] <= 200]
# selected_data.loc[:, '建筑面积'] = selected_data['建筑面积'].str.replace('平米', '').astype(float)
# df = pd.DataFrame(selected_data )

# 提取总价列作为y轴数据
y = df11['总价']


x = range(1, len(df11) + 1)
# 创建折线图
plt.plot(x, y, label='Actual Price')



# 添加第二条折线
plt.plot(x, l1,label='Lasso')
plt.legend()
# 添加标题和标签
plt.title('Total Price vs. Sample Number')
plt.xlabel('Sample Number')
plt.ylabel('Total Price')

plt.show()


plt.plot(x, y, label='Actual Price')



# 添加第二条折线
plt.plot(x, l2,label='Random Forest')
plt.legend()
# 添加标题和标签
plt.title('Total Price vs. Sample Number')
plt.xlabel('Sample Number')
plt.ylabel('Total Price')

plt.show()

plt.plot(x, y, label='Actual Price')



# 添加第二条折线
plt.plot(x, l3,label='Decision Tree')
plt.legend()
# 添加标题和标签
plt.title('Total Price vs. Sample Number')
plt.xlabel('Sample Number')
plt.ylabel('Total Price')

plt.show()


plt.plot(x, y, label='Actual Price')



# 添加第二条折线
plt.plot(x, l4,label='KNN')
plt.legend()
# 添加标题和标签
plt.title('Total Price vs. Sample Number')
plt.xlabel('Sample Number')
plt.ylabel('Total Price')

plt.show()
