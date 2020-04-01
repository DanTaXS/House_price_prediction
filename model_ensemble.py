#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

#筛选后的数据导入
test_data = pd.read_csv(r'c:/users/41547/test_program/house_price/test.csv')
train = pd.read_csv(r'c:/users/41547/test_program/house_price/x_train_finished.csv')
y_train = pd.read_csv(r'c:/users/41547/test_program/house_price/y_train_finished.csv')
test = pd.read_csv(r'c:/users/41547/test_program/house_price/test_finished.csv')

n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train.values, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
#LASSO Regression：
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
#Elastic Net Regression
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
#Kernel Ridge Regression :
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
#Gradient Boosting Regression：
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,max_depth=4, max_features='sqrt',min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =5)
#XGBoost：
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200,reg_alpha=0.4640, reg_lambda=0.8571,subsample=0.5213, silent=1,random_state =7, nthread = -1)
#LightGBM:
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,learning_rate=0.05, n_estimators=720,max_bin = 55, bagging_fraction = 0.8,bagging_freq = 5, feature_fraction = 0.2319,feature_fraction_seed=9, bagging_seed=9,min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    def fit(self,X,y):
        self.base_models_ = [list() for x in self.base_models] #生成一个list，内容为x个空list
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits = self.n_folds,shuffle = True,random_state= 156 ) #return a 分组类实例

        #生成一个空数组，dtypes为数组，用于存放每个models训练的预测集
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y): #生成一个生成器
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index],y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index,i] = y_pred
        
        #stacking过程，将预测值作为一个new feature作为input和实际的y进行训练，获取一个final classifier
        self.meta_model_.fit(out_of_fold_predictions,y)

        return self #返回类的实例
    
    def predict(self,X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ 
        ])#生成一个由n个model预测出的集合，作为meta_model的输入
        return self.meta_model_.predict(meta_features) #通过meta_model进行筛选，筛选出最优解

#预测模型为：ENet、GBoost、KRR，利用Lasso作为final_classifier
stacked_averaged_models = StackingAveragedModels(
    base_models = (ENet, GBoost, KRR),
    meta_model = lasso
)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

stacked_averaged_models.fit(train.values, y_train.values)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

model_xgb.fit(train.values, y_train.values)
xgb_train_pred = model_xgb.predict(train.values)
xgb_pred = np.expm1(model_xgb.predict(test.values))

model_lgb.fit(train.values, y_train.values)
lgb_train_pred = model_lgb.predict(train.values)
lgb_pred = np.expm1(model_lgb.predict(test.values))

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +xgb_train_pred*0.15 + lgb_train_pred*0.15 ))

ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
final_pred = pd.DataFrame({'Id':test_data.Id,'SalePrice':ensemble}) 
final_pred.to_csv(r'c:/users/41547/test_program/house_price/predictions.csv' ,index = False)