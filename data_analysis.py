Skip to content
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@DanTaXS 
DanTaXS
/
House_price
1
00
 Code Issues 0 Pull requests 0 Actions Projects 0 Wiki Security Insights Settings
House_price/data_analysis.py /
@DanTaXS DanTaXS Add files via upload
06cc727 8 minutes ago
101 lines (89 sloc)  4.18 KB
  
Code navigation is available!
Navigate your code with ease. Click on function and method calls to jump to their definitions or references in the same repository. Learn more

#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

original_data = pd.read_csv(r'c:/users/41547/test_program/house_price/train.csv')
train_data = original_data.drop('Id',axis = 1)
'''初步观察
missing_data columns:LotFrontage、Alley*、MasVnrType、MasVnrArea、BsmtQual、BsmtCond、BsmtExposure、
BsmtFinType2、Electrical、FireplaceQu*、GaragaeType、GaragBlt、GarageFinish、GarageQual、GarageCond、PoolQC***、Fence***、MiscFeature***
Target column:SalePrice'''

'''观察关系'''
#分析Sale_price
train_data.SalePrice.describe()
sns.distplot(train_data.SalePrice,fit = norm)
plt.title('SalePrice_distribution')
print('Skewness:%f' % train_data.SalePrice.skew())
print('Kurtosiss:%f' % train_data.SalePrice.kurt())
#Sale_price 趋近正态分布，其偏度与峰度不大，存在部分Outliers

#各个变量的相关性程度:
corrmat = train_data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax = 0.9,square = True)
plt.title('variables\' correlation')

#r>0.5为中度相关，寻找中度相关的变量
highcor_var = corrmat.loc[corrmat.SalePrice.abs()>0.5].sort_values(by = 'SalePrice',ascending = False).index
''''SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea','TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt','YearRemodAdd'''
data = train_data.loc[:,highcor_var]
fig,ax = plt.subplots(figsize = (9,9))
fig = sns.heatmap(data.corr(),annot = True,square = True)
plt.title('the highest variables')
'''
1.GrLivArea与TotRmsAbuGrd高度相关,增加拟合冗余，因此舍弃相关度较低的TotRmsAbvGrd
2.GarageCars与GarageArea高度相关，其代表的是车库容量，可用相关度较高的GarageCar代表
3.TotalBsmtSF与1stFlrSF高度相关，其代表的是一楼的‘院子’大小，可用相关度较高的TotalBsmtSF代表
筛选出'OverallQual','GrLivArea', 'GarageCars', 'TotalBsmtSF, 'FullBath','YearBuilt','YearRemodAdd'''
data.drop(labels = ['TotRmsAbvGrd','GarageArea','1stFlrSF'],axis = 1,inplace = True)
y_data = data.SalePrice
#分析以上7个variables对Target的影响：
var1 = data.OverallQual
fig,ax = plt.subplots(figsize=(12,9))
fig = sns.boxplot(x=var1,y=y_data)
fig.axis(ymin = 0,ymax = 800000)
plt.title('OverallQual and SalePrice relation[box]')

var2 = data.YearBuilt
fig,ax = plt.subplots(figsize=(12,9))
fig = sns.boxplot(x=var2,y=y_data)
fig.axis(ymin = 0,ymax = 800000)
plt.xticks(rotation = 90)
plt.title('YearBuilt and SalePrice relation[box]')

var3 = data.YearRemodAdd
fig,ax = plt.subplots(figsize=(12,9))
fig = sns.boxplot(x=var3,y=y_data)
fig.axis(ymin = 0,ymax = 800000)
plt.xticks(rotation = 90)
plt.title('YearRemodAdd and SalePrice relation[box]')

var4 = data.TotalBsmtSF #可进一步分类？
fig,ax = plt.subplots(figsize=(12,9))
fig = sns.scatterplot(x=var4,y = y_data)#呈强烈正相关，可能存在指数相关
fig.axis(ymin = 0,ymax = 800000)
plt.title('TotalBsmtSf and SalePrice relation[scatter]')

bins = list(range(0,2401,50))
bins.append(var4.max())
var4_bins = pd.cut(var4,bins,include_lowest = True)
fig,ax = plt.subplots(figsize=(16,8))
fig = sns.boxplot(x = var4_bins,y=np.log10(y_data))
plt.xticks(rotation = 90)
plt.title('TotalBsmtSf and SalePrice relation[box]')#呈指数相关

var6 = data.GrLivArea #可进一步分类
fig,ax = plt.subplots(figsize=(10,7))
fig = sns.scatterplot(x=var6,y=y_data)
fig.axis(ymin = 0,ymax = 800000)
plt.title('GrLivArea  and SalePrice relation[scatter]')

bins = list(range(0,3000,50))
bins.append(var6.max())
var6_bins = pd.cut(var6,bins)
fig,ax = plt.subplots(figsize=(16,8))
fig = sns.boxplot(x = var6_bins,y=np.log10(y_data))
plt.xticks(rotation = 90)
plt.title('GrLivArea and SalePrice relation[box]')#呈指数相关

var7 = data.FullBath #实际意义不明？但存在一定相关性
fig,ax = plt.subplots(figsize = (10,7))
fig = sns.boxplot(x = var7,y= y_data)
plt.title('FullBath and SalePrice relation[box]')

var8 = data.GarageCars #车库容量大小
fig,ax = plt.subplots(figsize = (10,7))
fig = sns.boxplot(x = var8,y = y_data)
plt.title('GarageCars and SalePrice relation[box]')
© 2020 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
