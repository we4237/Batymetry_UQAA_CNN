import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from osgeo import gdal, osr
from scipy import stats
from statistics import mean
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import explained_variance_score, r2_score, median_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize

input =  r'D:\SHOU\NSG(0717)_1021\gf6_ll.dat'
TrainPoints = []
prefix = 'D:\\SHOU\\NSG(0717)_1021\\points\\'
suffix = '.txt'
size = [f'{x}' for x in range(1000,10100,500) ]
for x in size:
    path = prefix+str(x)+suffix
    TrainPoints.append(np.loadtxt(path))


# RMSE_UQAA = []
# MRE_UQAA = []
# for i in range(len(size)):
#
#     ds_sample = gdal.Open(input)  # type: gdal.Datasetr
#     sample_geotrans = ds_sample.GetGeoTransform()
#     sample_proj = ds_sample.GetProjection()
#     sample_width = ds_sample.RasterXSize
#     sample_height = ds_sample.RasterYSize
#     sample_bands = ds_sample.RasterCount
#     sample_data = ds_sample.ReadAsArray(0, 0, sample_width, sample_height)
#     gdal.AllRegister()
#     im_data = np.nan_to_num(sample_data)
#     im_data = np.vstack(
#                     (im_data[0:3],
#                      )
#                     )
#     ag440 = pd.read_csv(r'D:\SHOU\NSG(0717)_1021\ag440.csv', header = 0)
#     ag440 = ag440.values
#     ag440 = np.expand_dims(ag440,axis=0)
#     c = pd.read_csv(r'D:\SHOU\NSG(0717)_1021\c.csv', header = 0)
#     c = c.values
#     c = np.expand_dims(c,axis=0)
#     im_b = np.vstack((im_data,ag440,c))
#     # im_b = im_data
#     # 读取样本点数据
#     points= TrainPoints[i]
#     lon = points[:, 1]  # 经度
#     lat = points[:, 0]  # 纬度
#     depth = points[:,2]
#     points = (np.array([lon, lat])).T  # (经度,纬度)
#
#     def getSRSPair(dataset):
#         '''
#         获得给定数据的投影参考系和地理参考系
#         :param dataset: GDAL地理数据
#         :return: 投影参考系和地理参考系
#         '''
#         prosrs = osr.SpatialReference()
#         prosrs.ImportFromWkt(dataset.GetProjection())
#         geosrs = prosrs.CloneGeogCS()
#         return prosrs, geosrs
#     def lonlat2geo(dataset, points):
#         '''
#         将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
#         :param dataset: GDAL地理数据
#         :param lon: 地理坐标lon经度
#         :param lat: 地理坐标lat纬度
#         :return: 经纬度坐标(lon, lat)对应的投影坐标
#         '''
#         prosrs, geosrs = getSRSPair(dataset)
#         ct = osr.CoordinateTransformation(geosrs, prosrs)
#         coords = ct.TransformPoints(points)
#         return coords
#     def geo2imagexy(dataset, x, y):
#         '''
#         根据GDAL的六参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
#         :param dataset: GDAL地理数据
#         :param x: 投影或地理坐标x
#         :param y: 投影或地理坐标y
#         :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
#         '''
#         trans = dataset.GetGeoTransform()
#         a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
#         b = np.array([x - trans[0], y - trans[3]])
#         rowandcol = np.linalg.solve(a, b)
#         return rowandcol # 使用numpy的linalg.solve进行二元一次方程的求解
#     def points2bands(points):
#         #经纬度 -> 投影坐标：
#         coords = lonlat2geo(ds_sample, points)
#         coords = np.delete(coords, 2, axis=1)  # 删除第3列，第3列是z没用
#         #投影坐标 -> 图上坐标:
#         rowandcol = geo2imagexy(ds_sample, coords[:,0], coords[:,1])
#
#         row = np.array(rowandcol[1, :]).astype(np.int32)  # 行号转换为整数
#         col = np.array(rowandcol[0, :]).astype(np.int32)  # 列号转换为整数
#         b_dn = (im_b[:, row, col]).T
#
#         return b_dn
#
#     b_dn = points2bands(points)
#     Xtrain,Xtest,Ytrain,Ytest = train_test_split(b_dn,depth,test_size=0.1,random_state=10)
#
#     # regrRF = RandomForestRegressor(min_samples_split=2)
#     # regrRF.fit(Xtrain, Ytrain)
#     # parameters = {'max_depth' : [50,100,200],'n_estimators':[100,500,1000,2000],'random_state':[0,10,20,30,40,50]}
#     # grid_search = GridSearchCV(regrRF,parameters,cv=3)
#     # grid_search.fit(Xtest,Ytest)
#     # print(grid_search.best_params_)
#
#     # 随机森林
#     print('='*10,'RF:',str(i),'='*10)
#     regrRF = RandomForestRegressor(n_estimators=1000, max_depth=50, min_samples_split=2, random_state=0)
#     regrRF.fit(Xtrain, Ytrain)
#     Yhat_rf = regrRF.predict(Xtrain)
#     Y_rftest = regrRF.predict(Xtest)
#     print('训练集R^2: %f' % regrRF.score(Xtrain,Ytrain))
#     print('测试集R^2: %f' % regrRF.score(Xtest,Ytest))
#     print('RMSE: %f' % mean_squared_error(Y_rftest,Ytest) ** 0.5)
#     RMSE_UQAA.append(mean_squared_error(Y_rftest,Ytest) ** 0.5)
#     print('MAPE: %f' % (mean_absolute_percentage_error(Y_rftest,Ytest)*100))
#     MRE_UQAA.append((mean_absolute_percentage_error(Y_rftest,Ytest)*100))


# RMSE_noUQAA = []
# MRE_noUQAA = []
# for i in range(len(size)):
#
#     ds_sample = gdal.Open(input)  # type: gdal.Datasetr
#     sample_geotrans = ds_sample.GetGeoTransform()
#     sample_proj = ds_sample.GetProjection()
#     sample_width = ds_sample.RasterXSize
#     sample_height = ds_sample.RasterYSize
#     sample_bands = ds_sample.RasterCount
#     sample_data = ds_sample.ReadAsArray(0, 0, sample_width, sample_height)
#     gdal.AllRegister()
#     im_data = np.nan_to_num(sample_data)
#     im_data = np.vstack(
#                     (im_data[0:3],
#                      )
#                     )
#
#     im_b = im_data
#     # 读取样本点数据
#     points= TrainPoints[i]
#     lon = points[:, 1]  # 经度
#     lat = points[:, 0]  # 纬度
#     depth = points[:,2]
#     points = (np.array([lon, lat])).T  # (经度,纬度)
#
#     def getSRSPair(dataset):
#         '''
#         获得给定数据的投影参考系和地理参考系
#         :param dataset: GDAL地理数据
#         :return: 投影参考系和地理参考系
#         '''
#         prosrs = osr.SpatialReference()
#         prosrs.ImportFromWkt(dataset.GetProjection())
#         geosrs = prosrs.CloneGeogCS()
#         return prosrs, geosrs
#     def lonlat2geo(dataset, points):
#         '''
#         将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
#         :param dataset: GDAL地理数据
#         :param lon: 地理坐标lon经度
#         :param lat: 地理坐标lat纬度
#         :return: 经纬度坐标(lon, lat)对应的投影坐标
#         '''
#         prosrs, geosrs = getSRSPair(dataset)
#         ct = osr.CoordinateTransformation(geosrs, prosrs)
#         coords = ct.TransformPoints(points)
#         return coords
#     def geo2imagexy(dataset, x, y):
#         '''
#         根据GDAL的六参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
#         :param dataset: GDAL地理数据
#         :param x: 投影或地理坐标x
#         :param y: 投影或地理坐标y
#         :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
#         '''
#         trans = dataset.GetGeoTransform()
#         a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
#         b = np.array([x - trans[0], y - trans[3]])
#         rowandcol = np.linalg.solve(a, b)
#         return rowandcol # 使用numpy的linalg.solve进行二元一次方程的求解
#     def points2bands(points):
#         #经纬度 -> 投影坐标：
#         coords = lonlat2geo(ds_sample, points)
#         coords = np.delete(coords, 2, axis=1)  # 删除第3列，第3列是z没用
#         #投影坐标 -> 图上坐标:
#         rowandcol = geo2imagexy(ds_sample, coords[:,0], coords[:,1])
#
#         row = np.array(rowandcol[1, :]).astype(np.int32)  # 行号转换为整数
#         col = np.array(rowandcol[0, :]).astype(np.int32)  # 列号转换为整数
#         b_dn = (im_b[:, row, col]).T
#
#         return b_dn
#
#     b_dn = points2bands(points)
#     Xtrain,Xtest,Ytrain,Ytest = train_test_split(b_dn,depth,test_size=0.1,random_state=10)
#
#     # regrRF = RandomForestRegressor(min_samples_split=2)
#     # regrRF.fit(Xtrain, Ytrain)
#     # parameters = {'max_depth' : [50,100,200],'n_estimators':[100,500,1000,2000],'random_state':[0,10,20,30,40,50]}
#     # grid_search = GridSearchCV(regrRF,parameters,cv=3)
#     # grid_search.fit(Xtest,Ytest)
#     # print(grid_search.best_params_)
#
#     # 随机森林
#     print('='*10,'RF:',str(i),'='*10)
#     regrRF = RandomForestRegressor(n_estimators=500, max_depth=50, min_samples_split=2, random_state=10)
#     regrRF.fit(Xtrain, Ytrain)
#     Yhat_rf = regrRF.predict(Xtrain)
#     Y_rftest = regrRF.predict(Xtest)
#     print('训练集R^2: %f' % regrRF.score(Xtrain,Ytrain))
#     print('测试集R^2: %f' % regrRF.score(Xtest,Ytest))
#     print('RMSE: %f' % mean_squared_error(Y_rftest,Ytest) ** 0.5)
#     RMSE_noUQAA.append(mean_squared_error(Y_rftest,Ytest) ** 0.5)
#     print('MAPE: %f' % (mean_absolute_percentage_error(Y_rftest,Ytest)*100))
#     MRE_noUQAA.append((mean_absolute_percentage_error(Y_rftest,Ytest)*100))

# RMSE_UQAA = np.array(RMSE_UQAA)
# np.save('RMSE_rf_UQAA.npy',RMSE_UQAA)
# RMSE_npUQAA = np.array(RMSE_noUQAA)
# np.save('RMSE_rf_noUQAA.npy',RMSE_npUQAA)
RMSE_UQAA = np.load('RMSE_rf_UQAA.npy')
RMSE_noUQAA = np.load('RMSE_rf_noUQAA.npy')
# RMSE折线图
plt.figure(figsize=(6, 4), dpi=80)
plt.plot(size, RMSE_UQAA, color='blue', label='RF with UQAA')
plt.plot(size, RMSE_noUQAA, color='orange', label='RF without UQAA')
font1 = {'family':'Times New Roman','weight':'heavy','size': 14,}
plt.legend(loc='upper right',prop = font1)  # 显示图例
plt.ylabel("RMSE(m)", family='Times New Roman',weight = 'heavy',fontsize=20)
plt.xticks(range(0,len(size),2),fontproperties='Times New Roman',fontsize=14)
plt.yticks(fontproperties='Times New Roman',fontsize=14)
plt.ylim(0.5, 1)  # 设置y坐标轴的显示范围

plt.savefig('RF_RMSE' + '.svg', bbox_inches='tight', dpi=1000)
plt.show()

# MRE_UQAA = np.array(MRE_UQAA)
# np.save('MRE_rf_UQAA.npy',MRE_UQAA)
# MRE_npUQAA = np.array(MRE_noUQAA)
# np.save('MRE_rf_noUQAA.npy',MRE_npUQAA)
MRE_UQAA = np.load('MRE_rf_UQAA.npy')
MRE_noUQAA = np.load('MRE_rf_noUQAA.npy')
# MRE折线图
plt.figure(figsize=(6, 4), dpi=80)
plt.plot(size, MRE_UQAA, color='blue', label='RF with UQAA')
plt.plot(size, MRE_noUQAA, color='orange', label='RF without UQAA')
font1 = {'family':'Times New Roman','weight':'heavy','size': 14,}
plt.legend(loc='upper right',prop = font1)  # 显示图例
plt.ylabel("MRE(%)", family='Times New Roman',weight = 'heavy',fontsize=20)
plt.xticks(range(0,len(size),2),fontproperties='Times New Roman',fontsize=14)
plt.yticks(fontproperties='Times New Roman',fontsize=14)
plt.ylim(5, 15)  # 设置y坐标轴的显示范围

plt.savefig('RF_MRE' + '.svg', bbox_inches='tight', dpi=1000)
plt.show()