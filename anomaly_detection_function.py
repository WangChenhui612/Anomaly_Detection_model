import pandas as pd
import numpy as np
from numpy import *

import matplotlib
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# from pyemma import msm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0, len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i] - 1]
        distance.set_value(i, np.linalg.norm(Xa - Xb))
    return distance

def gettTransitionMatrix(df):
    df = np.array(df)
    model = msm.estimate_markov_model(df, 1)
    return model.transition_matrix

def markovAmomaly(df, windows_size, threshold):
    transition_matrix = getDistanceByPoint(df)
    real_threshold = threshold ** windows_size
    df_anomaly = []
    for j in range(len(df)):
        if j < windows_size:
            df_anomaly.append(0)
        else:
            sequence = df[j - windows_size]
            sequence = sequence.reset_index(drop=True)
            df_anomaly.append(anomalyElement(sequence, real_threshold, transition_matrix))
    return df_anomaly


df = pd.read_csv("ambient_temperature_system_failure.csv")

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['value'] = (df['value'] - 32) * 5 / 9
# df.plot(x='timestamp', y='value')


# 判断是白天还是晚上 (7:00-22:00)
df['hours'] = df['timestamp'].dt.hour
df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

# 获取一周内的第几天(Monday=0, Sunday=6) and if it's a week end day or week day.
df['DayOfWeek'] = df['timestamp'].dt.dayofweek
df['WeekDay'] = (df['DayOfWeek'] < 5).astype(int)
outliers_fraction = 0.01

# time with int to plot easily
df['time_epoch'] = (df['timestamp'].astype(np.int64) / 100000000000).astype(np.int64)

# creation of 4 distinct categories that seem useful (week end/day week & night/day)
df['categories'] = df['WeekDay'] * 2 + df['daylight']

a = df.loc[df['categories'] == 0, 'value']
b = df.loc[df['categories'] == 1, 'value']
c = df.loc[df['categories'] == 2, 'value']
d = df.loc[df['categories'] == 3, 'value']

fig, ax = plt.subplots()
a_heights, a_bins = np.histogram(a)   # a_heights得到的是频数，即a_bins里面11个点区间之间的样本个数，a_bins得到11个均分的点，是将样本中的最大值减去最小值均等分
b_heights, b_bins = np.histogram(b, bins=a_bins)
c_heights, c_bins = np.histogram(c, bins=a_bins)
d_heights, d_bins = np.histogram(d, bins=a_bins)

width = (a_bins[1] - a_bins[0]) / 6
ax.bar(a_bins[: -1], a_heights * 100 / a.count(), width=width, facecolor='blue', label='WeekEndNight')
ax.bar(b_bins[: -1] + width, b_heights * 100 / b.count(), width=width, facecolor='green', label='WeekEndLight')
ax.bar(c_bins[: -1] + width * 2, c_heights * 100 / c.count(), width=width, facecolor='red', label='WeekDayNight')
ax.bar(d_bins[: -1] + width * 3, d_heights * 100 / d.count(), width=width, facecolor='black', label='WeekDayLight')
plt.legend()
plt.show()

#2.1建立聚类模型cluster
data = df[['value', 'hours', 'daylight', 'DayOfWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)

#降维至2个重要的维度
pca = PCA(n_components=2)
data = pca.fit_transform(data)

#标准化这两个new feature
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)

n_cluster = range(1, 20)
kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]
fig, ax = plt.subplots()
ax.plot(n_cluster, scores)
plt.show()

df['cluster'] = kmeans[3].predict(data)
df['principal_feature1'] = data[0]
df['principal_feature2'] = data[1]
df['cluster'].value_counts()

#画出不同的聚类
fig, ax = plt.subplots()
colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'pink', 4: 'black', 5: 'orange', 6: 'cyan', 7: 'yellow', 8: 'brown',
          9: 'purple', 10: 'white', 11: 'grey', 12: 'lightblue', 13: 'lightgreen', 14: 'darkgrey'}
ax.scatter(df['principal_feature1'], df['principal_feature2'], c=df['cluster'].apply(lambda x: colors[x]))
plt.show()

#计算点到最近聚类中心的距离，距离最大的点为异常点
distance = getDistanceByPoint(data, kmeans[14])
number_of_outliers = int(outliers_fraction * len(distance))
threshold = distance.nlargest(number_of_outliers).min()
df['animaly21'] = (distance >= threshold).astype(int)

#异常点可视化
fig, ax = plt.subplots()
colors = {0: 'blue', 1: 'red'}
ax.scatter(df['principal_feature1'], df['principal_feature2'], c=df['animaly21'].apply(lambda x: colors[x]))
plt.show()

#时间序列的异常可视化
fig, ax = plt.subplots()
a = df.loc[df['animaly21'] == 1, ['time_epoch', 'value']]
ax.plot(df['time_epoch'], df['value'], color='blue')
ax.scatter(a['time_epoch'], a['value'], color='red')
plt.show()

#温度异常分布可视化
a = df.loc[df['animaly21'] == 0, 'value']
b = df.loc[df['animaly21'] == 1, 'value']

fig, axs = plt.subplots()
axs.hist([a, b], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
plt.legend()
plt.show()
#

#2.2高斯分类模型
#据目录建立4个不同的数据集合
df_class0 = df.loc[df['categories'] == 0, 'value']
df_class1 = df.loc[df['categories'] == 1, 'value']
df_class2 = df.loc[df['categories'] == 2, 'value']
df_class3 = df.loc[df['categories'] == 3, 'value']

#画出温度分布
fig, axs = plt.subplots(2, 2)
df_class0.hist(ax=axs[0, 0], bins=32)
df_class1.hist(ax=axs[0, 1], bins=32)
df_class2.hist(ax=axs[1, 0], bins=32)
df_class3.hist(ax=axs[1, 1], bins=32)

#对每个分类使用高斯分布
envelope = EllipticEnvelope(contamination=outliers_fraction)
X_train = df_class0.values.reshape(-1, 1)
envelope.fit(X_train)
df_class0 = pd.DataFrame(df_class0)
df_class0['deviation'] = envelope.decision_function(X_train)
df_class0['anomaly'] = envelope.predict(X_train)

envelope = EllipticEnvelope(contamination=outliers_fraction)
X_train = df_class1.values.reshape(-1, 1)
envelope.fit(X_train)
df_class1 = pd.DataFrame(df_class1)
df_class1['deviation'] = envelope.decision_function(X_train)
df_class1['anomaly'] = envelope.predict(X_train)

envelope = EllipticEnvelope(contamination=outliers_fraction)
X_train = df_class2.values.reshape(-1, 1)
envelope.fit(X_train)
df_class2 = pd.DataFrame(df_class2)
df_class2['deviation'] = envelope.decision_function(X_train)
df_class2['anomaly'] = envelope.predict(X_train)

envelope = EllipticEnvelope(contamination=outliers_fraction)
X_train = df_class3.values.reshape(-1, 1)
envelope.fit(X_train)
df_class3 = pd.DataFrame(df_class3)
df_class3['deviation'] = envelope.decision_function(X_train)
df_class3['anomaly'] = envelope.predict(X_train)

#画出温度异常分布
a0 = df_class0.loc[df_class0['anomaly'] == 1, 'value']
b0 = df_class0.loc[df_class0['anomaly'] == -1, 'value']

a1 = df_class1.loc[df_class1['anomaly'] == 1, 'value']
b1 = df_class1.loc[df_class1['anomaly'] == -1, 'value']

a2 = df_class2.loc[df_class2['anomaly'] == 1, 'value']
b2 = df_class2.loc[df_class2['anomaly'] == -1, 'value']

a3 = df_class3.loc[df_class3['anomaly'] == 1, 'value']
b3 = df_class3.loc[df_class3['anomaly'] == -1, 'value']

fig, axs = plt.subplots(2, 2)
axs[0, 0].hist([a0, b0], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
axs[0, 1].hist([a1, b1], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
axs[1, 0].hist([a2, b2], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
axs[1, 1].hist([a3, b3], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])

axs[0, 0].set_title("WeekEndNight")
axs[0, 1].set_title("WeekEndLight")
axs[1, 0].set_title("WeekDayNight")
axs[1, 1].set_title("WeekDayLight")
plt.legend()
plt.show()


df_class = pd.concat([df_class0, df_class1, df_class2, df_class3])
df['anomaly22'] = df_class['anomaly']
df['anomaly22'] = np.array(df['anomaly22'] == -1).astype(int)

fig, ax = plt.subplots()
a = df.loc[df['anomaly22'] == 1, ('time_epoch', 'value')]
ax.plot(df['time_epoch'], df['value'], color='blue')
ax.scatter(a['time_epoch'], a['value'], color='red')
plt.show()

#2.3高斯聚类模型
#跟上面的方法类似，利用聚类来分离不同组里面的数据

#2.4马尔科夫链
x1 = (df['value'] <= 18).astype(int)
x2 = ((df['value'] > 18) & (df['value'] <= 21)).astype(int)
x3 = ((df['value'] > 21) & (df['value'] <= 24)).astype(int)
x4 = ((df['value'] > 24) & (df['value'] <= 27)).astype(int)
x5 = (df['value'] > 27).astype(int)
df_mm = x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5


#2.4孤立森林算法
data = df[['value', 'hours', 'daylight', 'DayOfWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)
#训练孤立森林模型
model = IsolationForest(contamination=outliers_fraction)
model.fit(data)
#增加数据到主函数里面
df['anomaly25'] = pd.Series(model.predict(data))
df['anomaly25'] = df['anomaly25'].map({1: 0, -1: 1})
print(df['anomaly25'].value_counts())

#时间序列异常点可视化
fig, ax = plt.subplots()
a = df.loc[df['anomaly25'] == 1, ['time_epoch', 'value']]
ax.plot(df['time_epoch'], df['value'],  color='blue')
ax.scatter(a['time_epoch'], a['value'], color='red')
plt.show()

#温度分布柱状图异常值可视化
a = df.loc[df['anomaly25'] == 0, 'value']
b = df.loc[df['anomaly25'] == 1, 'value']

fig, axs = plt.subplots()
axs.hist([a, b], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
plt.legend()
plt.show()


#2.6支持向量机
data = df[['value', 'hours', 'daylight', 'DayOfWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
#训练SVM
model = OneClassSVM(nu=0.95 * outliers_fraction)
data = pd.DataFrame(np_scaled)
model.fit(data)
#添加数据
df['anomaly26'] = pd.Series(model.predict(data))
df['anomaly26'] = df['anomaly26'].map({1: 0, -1: 1})
print(df['anomaly26'].value_counts())

#时间序列异常点可视化
fig, ax = plt.subplots()
a = df.loc[df['anomaly26'] == 1, ['time_epoch', 'value']]
ax.plot(df['time_epoch'], df['value'],  color='blue')
ax.scatter(a['time_epoch'], a['value'], color='red')
plt.show()

#温度分布柱状图异常值可视化
a = df.loc[df['anomaly26'] == 0, 'value']
b = df.loc[df['anomaly26'] == 1, 'value']

fig, axs = plt.subplots()
axs.hist([a, b], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
plt.legend()
plt.show()


#2.7循环神经网络
data_n = df[['value', 'hours', 'daylight', 'DayOfWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data_n)
data_n = pd.DataFrame(np_scaled)

#训练和测试模型的参数
prediction_time = 1
testdatasize = 1000
unroll_length = 50
testdatacut = testdatasize + unroll_length + 1

#训练数据
x_train = data_n[0: -prediction_time - testdatacut].as_matrix()
y_train = data_n[prediction_time: -testdatacut][0].as_matrix()

#测试数据
x_test = data_n[0 - testdatacut: -prediction_time].as_matrix() #将表格转化为矩阵
y_test = data_n[prediction_time - testdatacut:][0].as_matrix()


def unroll(data, sequence_length=24):
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)

#根据序列数据形状对数据集进行重构
x_train = unroll(x_train, unroll_length)
x_test = unroll(x_test, unroll_length)
y_train = y_train[-x_train.shape[0]:]
y_test = y_test[-x_test.shape[0]:]

#查看数据集的维数
print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)


from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
from keras.models import model_from_json
import sys

#建立模型
model = Sequential()

model.add(LSTM(input_dim=x_train.shape[-1], output_dim=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.add(Activation('linear'))
start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('compilation time:{}'.format(time.time() - start))

model.fit(x_train, y_train, batch_size=3028, nb_epoch=30, validation_split=0.1)

loaded_model = model
diff = []
ratio = []
p = loaded_model.predict(x_test)
for u in range(len(y_test)):
    pr = p[u][0]
    ratio.append((y_test[u] / pr) - 1)
    diff.append(abs(y_test[u] - pr))

# 画出真实和预测数据的曲线
fig, axs = plt.subplots()
axs.plot(p, color='red', label='prediction')
axs.plot(y_test, color='blue', label='y_test')
plt.legend(loc='upper left')
plt.show()

diff = pd.Series(diff)
number_of_outliers = int(outliers_fraction * len(diff))
threshold = diff.nlargest(number_of_outliers).min()  # 选取最大的前10个值
test = (diff >= threshold).astype(int)
complement = pd.Series(0, index=np.arange(len(data_n) - testdatasize))
df['anomaly27'] = complement.append(test, ignore_index='Ture')
print(df['anomaly27'].value_counts())


#时间序列异常点可视化
fig, ax = plt.subplots()
a = df.loc[df['anomaly27'] == 1, ['time_epoch', 'value']]
ax.plot(df['time_epoch'], df['value'],  color='blue')
ax.scatter(a['time_epoch'], a['value'], color='red')
plt.axis([1.370 * 1e7, 1.405 * 1e7, 15, 30])
plt.show()

#温度分布柱状图异常值可视化
a = df.loc[df['anomaly27'] == 0, 'value']
b = df.loc[df['anomaly27'] == 1, 'value']

fig, axs = plt.subplots()
axs.hist([a, b], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
plt.legend()
plt.show()


