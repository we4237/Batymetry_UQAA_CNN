import numpy as np
import pandas as pd
from osgeo import gdal, osr
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import explained_variance_score,r2_score,median_absolute_error,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize


input =  r'D:\SHOU\NSG(0717)_1021\gf6_ll.dat'
TrainPoints=np.loadtxt(r'D:\SHOU\NSG(0717)_1021\points\5000.txt')

ds_sample = gdal.Open(input)  # type: gdal.Datasetr
sample_geotrans = ds_sample.GetGeoTransform()
sample_proj = ds_sample.GetProjection()
sample_width = ds_sample.RasterXSize
sample_height = ds_sample.RasterYSize
sample_bands = ds_sample.RasterCount
sample_data = ds_sample.ReadAsArray(0, 0, sample_width, sample_height)

gdal.AllRegister()
im_data = np.nan_to_num(sample_data)
# im_data = im_data.astype(float)
# im_data[im_data==0]=0.00001
im_data = np.vstack(
                (im_data[0:3],
                 # np.expand_dims((im_data[0] / im_data[1]),axis=0),
                 # np.expand_dims((im_data[0] / im_data[2]),axis=0),
                 # np.expand_dims((im_data[1] / im_data[2]),axis=0),
                 # np.expand_dims((im_data[1] / im_data[3]), axis=0),
                 )
                )

ag440 = pd.read_csv(r'D:\SHOU\NSG(0717)_1021\ag440.csv', header = 0)
ag440 = ag440.values
ag440 = np.expand_dims(ag440,axis=0)
c = pd.read_csv(r'D:\SHOU\NSG(0717)_1021\c.csv', header = 0)
c = c.values
c = np.expand_dims(c,axis=0)

## UQAA
im_b = np.vstack((im_data,ag440,c))
# im_b = im_data
# 读取样本点数据
points= TrainPoints
lon = points[:, 1]  # 经度
lat = points[:, 0]  # 纬度
depth = points[:,2]
points = (np.array([lon, lat])).T  # (经度,纬度)

def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs

def lonlat2geo(dataset, points):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoints(points)
    return coords

def geo2imagexy(dataset, x, y):
    '''
    根据GDAL的六参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    rowandcol = np.linalg.solve(a, b)
    return rowandcol # 使用numpy的linalg.solve进行二元一次方程的求解
def points2bands(points):
    #经纬度 -> 投影坐标：
    coords = lonlat2geo(ds_sample, points)
    coords = np.delete(coords, 2, axis=1)  # 删除第3列，第3列是z没用
    #投影坐标 -> 图上坐标:
    rowandcol = geo2imagexy(ds_sample, coords[:,0], coords[:,1])

    row = np.array(rowandcol[1, :]).astype(np.int32)  # 行号转换为整数
    col = np.array(rowandcol[0, :]).astype(np.int32)  # 列号转换为整数
    b_dn = (im_b[:, row, col]).T
    return b_dn

b_dn = points2bands(points)
# 标准化
# scaler = StandardScaler()
# b_dn = scaler.fit_transform(b_dn)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(b_dn, depth, test_size=0.1, random_state=10)
dataset = np.vstack((Xtest[:,0], Xtest[:,1], Xtest[:,2],Xtest[:,3], Xtest[:,4], Ytest)).T
depth0_2 = dataset[dataset[:,5] <= 2]

depth2_4 = dataset[dataset[:,5] > 2]
depth2_4 = depth2_4[depth2_4[:,5] <= 4]

depth4_6 = dataset[dataset[:,5] > 4]
depth4_6 = depth4_6[depth4_6[:,5] <= 6]

depth6_8 = dataset[dataset[:,5] > 6]
depth6_8 = depth6_8[depth6_8[:,5] <= 8]

depth8_10 = dataset[dataset[:,5] > 8]
depth8_10 = depth8_10[depth8_10[:,5] <= 10]

depth10_ = dataset[dataset[:,5] > 10]

np.save('500_depth0_2_UQAA.npy',depth0_2)
np.save('500_depth2_4_UQAA.npy',depth2_4)
np.save('500_depth4_6_UQAA.npy',depth4_6)
np.save('500_depth6_8_UQAA.npy',depth6_8)
np.save('500_depth8_10_UQAA.npy',depth8_10)
np.save('500_depth10__UQAA.npy',depth10_ )


# stan
# im_b = im_data
# # 读取样本点数据
# points= TrainPoints
# lon = points[:, 1]  # 经度
# lat = points[:, 0]  # 纬度
# depth = points[:,2]
# points = (np.array([lon, lat])).T  # (经度,纬度)
#
# def getSRSPair(dataset):
#     '''
#     获得给定数据的投影参考系和地理参考系
#     :param dataset: GDAL地理数据
#     :return: 投影参考系和地理参考系
#     '''
#     prosrs = osr.SpatialReference()
#     prosrs.ImportFromWkt(dataset.GetProjection())
#     geosrs = prosrs.CloneGeogCS()
#     return prosrs, geosrs
#
# def lonlat2geo(dataset, points):
#     '''
#     将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
#     :param dataset: GDAL地理数据
#     :param lon: 地理坐标lon经度
#     :param lat: 地理坐标lat纬度
#     :return: 经纬度坐标(lon, lat)对应的投影坐标
#     '''
#     prosrs, geosrs = getSRSPair(dataset)
#     ct = osr.CoordinateTransformation(geosrs, prosrs)
#     coords = ct.TransformPoints(points)
#     return coords
#
# def geo2imagexy(dataset, x, y):
#     '''
#     根据GDAL的六参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
#     :param dataset: GDAL地理数据
#     :param x: 投影或地理坐标x
#     :param y: 投影或地理坐标y
#     :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
#     '''
#     trans = dataset.GetGeoTransform()
#     a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
#     b = np.array([x - trans[0], y - trans[3]])
#     rowandcol = np.linalg.solve(a, b)
#     return rowandcol # 使用numpy的linalg.solve进行二元一次方程的求解
# def points2bands(points):
#     #经纬度 -> 投影坐标：
#     coords = lonlat2geo(ds_sample, points)
#     coords = np.delete(coords, 2, axis=1)  # 删除第3列，第3列是z没用
#     #投影坐标 -> 图上坐标:
#     rowandcol = geo2imagexy(ds_sample, coords[:,0], coords[:,1])
#
#     row = np.array(rowandcol[1, :]).astype(np.int32)  # 行号转换为整数
#     col = np.array(rowandcol[0, :]).astype(np.int32)  # 列号转换为整数
#     b_dn = (im_b[:, row, col]).T
#     return b_dn
#
# b_dn = points2bands(points)
# # 标准化
# scaler = StandardScaler()
# b_dn = scaler.fit_transform(b_dn)
#
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(b_dn, depth, test_size=0.1, random_state=10)
# dataset = np.vstack((Xtest[:,0], Xtest[:,1], Xtest[:,2], Ytest)).T
# depth0_3 = dataset[dataset[:,3] <= 3]
# depth3_6 = dataset[dataset[:,3] > 3]
# depth3_6 = depth3_6[depth3_6[:,3] <= 6]
# depth6_9 = dataset[dataset[:,3] > 6]
# depth6_9 = depth6_9[depth6_9[:,3] <= 9]
# depth9_ = dataset[dataset[:,3] > 9]
#
# np.save('500_depth0_3_noUQAA_stan.npy',depth0_3)
# np.save('500_depth3_6_noUQAA_stan.npy',depth3_6)
# np.save('500_depth6_9_noUQAA_stan.npy',depth6_9)
# np.save('500_depth9__noUQAA_stan.npy',depth9_ )