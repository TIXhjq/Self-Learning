# _*_ coding:utf-8 _*_
'''=================================
@Author :tix_hjq
@Date   :19-11-15 上午11:59
================================='''
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score, r2_score
from hyperopt import fmin, tpe, hp, partial
from numpy.random import random, shuffle
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tqdm import tqdm
# from PIL import Image
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import networkx as nx
import pandas as pd
import numpy as np
import warnings
import datetime
# import cv2
import sys
import os
import gc
import re

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

print(os.getcwd())
#----------------------------------------------------
from model.base_model import base_model
data_folder='../../data/'
origin_data_folder=data_folder+'origin_data/'
submit_data_folder=data_folder+'submit/'
eda_data_folder=data_folder+'eda_data/'
result_fea_folder=submit_data_folder+'use_feature/'

class feature_tool(object):
    def __init__(self,save_folder):
        print('feature tool is backend')
        self.model = base_model(save_folder=submit_data_folder)
        self.save_folder=save_folder

    def null_feature(self,df,fea_cols): #row and col
        '''
        cal null feature
        '''
        #col is null
        is_null_fea=[]
        for col in fea_cols:
            is_null_col=col+'_is_null'
            df[is_null_col]=df[col].isnull().astype(int)
            is_null_fea.append(is_null_col)

        #row null count
        count_data=DataFrame(df[fea_cols].count(axis=1)).reset_index()
        df['num_null']=([len(fea_cols)]*count_data.shape[0])-(count_data[0])
        df['per_num_null']=df['num_null']/(df['num_null'].sum())

        null_fea=is_null_fea+['num_null','per_num_null']
        return df,null_fea

    def cal_cross_fea(df:DataFrame,first_fea:list,second_fea:list,third_fea:list=None,cross_rank=2):
        '''
        cross cate_fea
        '''
        cross_fea=[]
        if cross_rank==2:
            for f_1 in tqdm(first_fea,desc='rank_'+str(cross_rank)+' cross fea'):
                for f_2 in second_fea:
                    if f_1!=f_2:
                        cross_col=str(f_1)+'_cross_'+str(f_2)
                        df[cross_col]=(df[f_1].astype('str')+"_"+df[f_2].astype('str')).astype('category')
                        df[cross_col] = LabelEncoder().fit_transform(df[cross_col])
                        df[cross_col]=df[cross_col].astype('category')
                        cross_fea.append(cross_col)
        else:
            for f_1 in tqdm(first_fea,desc='rank_'+str(cross_rank)+' cross fea'):
                for f_2 in second_fea:
                    if f_1!=f_2:
                        for f_3 in third_fea:
                            if f_1!=f_3:
                                if f_2!=f_3:
                                    cross_col = str(f_1) + '_cross_' + str(f_2)+'_cross_'+str(f_3)
                                    df[cross_col] = (df[f_1].astype('str') + "_" + df[f_2].astype('str')+"_"+df[f_3].astype('str')).astype('category')
                                    df[cross_col]=LabelEncoder().fit_transform(df[cross_col])
                                    df[cross_col] = df[cross_col].astype('category')
                                    cross_fea.append(cross_col)
        return df, cross_fea

    def count_col(self,df:DataFrame,cate_fea:list):
        '''
        cate_fea count feature
        '''
        count_fea=[]
        for fea in tqdm(cate_fea,desc='cate fea count'):
            count_data=DataFrame(df[cate_fea].value_counts()).reset_index().rename(columns={fea:fea+'_count','index':fea})
            count_fea.append(fea+'_count')
            df=df.merge(count_data,'left',fea)

        return df,count_fea

    def stat_fea(self,df:DataFrame,cate_fea_list:list,num_fea_list:list,agg_param=['mean','sum','std']):
        cate_len=len(cate_fea_list)
        stat_fea_list=[]

        for cate_fea in tqdm(cate_fea_list,desc='by cate stat'):
            cate_len-=1
            by_agg_data=df.groupby(cate_fea)[num_fea_list].agg(agg_param)
            for num_fea in tqdm(num_fea_list,desc=cate_fea+'_stat_num_fea'+' rest:'+str(cate_len)):
                agg_cols=['by_'+cate_fea+'_'+cate_fea+'_'+agg_operator for agg_operator in agg_param]

                agg_data_=by_agg_data[num_fea]
                agg_=DataFrame(data=agg_data_.values,columns=agg_cols)
                agg_[cate_fea]=agg_data_.index.tolist()
                df=df.merge(agg_,'left',on=[cate_fea])
                stat_fea_list+=agg_cols

        return df,stat_fea_list

    def filter_feature(self,df:DataFrame,fea_cols,label,cate_fea,file_name):
        model_important,zero_fea=self.model.fit_transform(train_data=df,is_pred=False,cate_cols=cate_fea,label_col=label,use_cols=fea_cols)
        model_important.to_csv(self.save_folder+file_name+'_feature_important.csv',index=None)
        np.save(self.save_folder+file_name+'_zero_import.csv',zero_fea)

    def test_distribute(self,train:DataFrame,test:DataFrame,cate_fea:list=None):
        train['is_train']=1
        test['is_train']=0
        df=pd.concat([train,test],ignore_index=True)
        cols=df.columns.tolist()
        cols.remove('is_train')

        self.model.fit_transform(train_data=df, use_cols=cols, cate_cols=cate_fea,label_col='is_train',is_pred=False)

