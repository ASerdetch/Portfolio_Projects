#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:33:55 2020

@author: anna
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import warnings
import datetime
warnings.filterwarnings('ignore')

filepath=os.getcwd()+'/'


from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin


def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df



#initial eda funct
    
def sum_stats(df):
    for x in df.columns: 
        print(x)
        temp=df[x]
        print('Num nans: ', temp.isnull().sum())
        print('% nan: ', (temp.isnull().sum()/len(temp))*100)
        print('n unique: ', temp.nunique())
        #print('% nan: '(df[x].isnull().sum()))
        if x!='timestamp':
            try:
                
                fig, (ax1, ax2)=plt.subplots(2,1)
                fig.subplots_adjust(hspace=0.5)
                ax1.hist(temp)
                ax1.set_title(x)
                ax2.hist(np.log1p(temp))
                ax2.set_title('log '+ x)
                fig.show()
            except:
                print('cannot plot')
            if (temp.dtype in [int,float])| (str(temp.dtype)[:3] in ['flo','int']): 
                print('max: ', temp.max())
                print('min: ', temp.min())
                print('mean: ', temp.mean())
                print('median: ',temp.median())
            else: 
                print('val_counts: ',temp.value_counts() )



# feature engineering 

def feat_eng(df, df_weather, build_df, dummy=False,log=True,doy=False): 


    # build_df['floor_count'][build_df['floor_count'].isnull()]=0
    # build_df['year_built'][build_df['year_built'].isnull()]=np.nanmedian(build_df['year_built'])
       
    build_df=build_df.drop(['floor_count','year_built'],axis=1)
    
    if dummy == True: 
        dummy_use=pd.get_dummies(build_df['primary_use']).astype(int)
        build_df=pd.concat([build_df.drop('primary_use',axis=1),dummy_use],axis=1)
        dummy_use=pd.get_dummies(df['meter']).astype(int)
        df=pd.concat([df.drop('meter',axis=1), dummy_use],axis=1)
        
    print('preprocess weather data')       
    cols=['site_id',  'air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']  
    
    df_weather[cols]=df_weather[cols].astype(float)
    df_weather.cloud_coverage[df_weather.cloud_coverage.isnull()]=0
    
    #remove negative values
    df_weather.precip_depth_1_hr[df_weather['precip_depth_1_hr']<0]=0
    #will also change nans here to zero
    df_weather.precip_depth_1_hr[df_weather['precip_depth_1_hr'].isnull()]=0
    
    for x in df_weather.site_id.unique():
        try:
            df_weather[df_weather.site_id==x]=df_weather[df_weather.site_id==x].fillna(method='ffill')
        except: 
            for y in df_weather.columns:
                try:
                    df_weather[y][df_weather.site_id==x]=df_weather[y][df_weather.site_id==x].fillna(method='ffill')
                except:
                    print('cannot fill ', y, ' on site ', x)
        
    #fill in for for region with no data 
    df_weather.sea_level_pressure[df_weather.sea_level_pressure.isnull()]=0
    
    
    
    print('merging datasets')
    
    df=df.merge(build_df, how='left', on='building_id')
    df_final=df.merge(df_weather, how='left', on=['site_id','timestamp'])
    
    print('extract data from datetime')
    
    df_final['timestamp']=df_final['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df_final['month']=df_final['timestamp'].apply(lambda x: x.month)
    df_final['weekday']=df_final['timestamp'].apply(lambda x: x.weekday())
    df_final['hour']=df_final['timestamp'].apply(lambda x: x.hour)
   # df_final['weekofyear']=df_final['timestamp'].apply(lambda x: x.weekofyear)
    if doy==True: 
        df_final['dayofyear']=df_final['timestamp'].apply(lambda x: x.dayofyear)
    

    df_final.columns=[x.replace(' ','_') for x in df_final.columns]
    
    
    print('filling missing data')
    
    for x in df_final.site_id.unique():
        df_final[df_final.site_id==x]=df_final[df_final.site_id==x].fillna(method='ffill').fillna(method='bfill')
    
    df_final=df_final.drop(['timestamp'],axis=1)
    
    if log ==True: 
        #using natural log bc there are 0 values 
        df_final['meter_reading']=np.log1p(df_final.meter_reading)
        df_final['square_feet']=np.log1p(df_final.square_feet)
    return df_final


def temp_vis(df,split_by, types='month'):
    temp=df[df.meter_reading!=0]
    fig, ax=plt.subplots(4,4, sharex=True,sharey=True,figsize=(25,18))
    
    temp1=temp.groupby([split_by,types]).median().reset_index()
    if types=='month':
        temp2=temp.groupby([split_by,'dayofyear']).median().reset_index()
        
    ax=ax.ravel()
    for count, x in enumerate(df[split_by].unique()): 
        temp3=temp1[temp1[split_by]==x]
        ax[count].plot(temp3[types],temp3.meter_reading, c='cornflowerblue',label=types)
        if types=='month':
            temp4=temp2[temp2[split_by]==x]
            axb=ax[count].twiny()
            axb.plot(temp4.dayofyear,temp4.meter_reading, c='lightcoral', label='dayofyear')
            if count ==0: 
                ax[count].legend()
                axb.legend()
        ax[count].set_title(x)
        fig.show()



#predivtive model 

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import gc


param = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'reg_lambda' : 2,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}

catz=["building_id", "site_id", "meter", "primary_use"]

def pred(df_final_samp, valid=True, num_it=1000,cat=catz):

    x=df_final_samp.drop('meter_reading', axis=1)

    y=df_final_samp.meter_reading
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    kf.get_n_splits(x)
    
    if valid==True: 
        x,x_valid,y,y_valid=train_test_split(x,y,test_size=0.2)
    
    #if i wanted to keep a seperate set for testing, i would have to splot before kfold and use that  to evaliate metrics within each fold 
    #otherwise, collect results (annd idx valds)and then do eval after fold 
    
    pred=[]
    val=[]
    modelz=[]
    for train_index, test_index in  kf.split(x):
        #print(test_index)
        #print('spit',counts)
        #print('idx range: ', train_index.min(), train_index.max())
        
        print('new loop')
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    
        lgb_train = lgb.Dataset(x_train, label=y_train)
        lgb_test = lgb.Dataset(x_test, label=y_test)
    
        model = lgb.train(param, lgb_train, num_it, valid_sets=[lgb_train,lgb_test], 
                      early_stopping_rounds=100, verbose_eval=50, categorical_feature=cat)
    
        modelz.append(model)
        
        data_valid=x_test
        if valid==True:
            data_valid=x_valid
        model_pred= model.predict(data_valid, num_iteration=model.best_iteration)
        pred.append(model_pred)
        if valid==False: 
            val.append(y_test) 
            
        del x_train, y_train, x_test,y_test, lgb_train, lgb_test, model 
        gc.collect()
        
    if valid==True: 
        test=pd.DataFrame(pred)
        test=np.mean(test,axis=0)
        y_valid=y_valid.reset_index(drop=True)
    elif valid==False: 
        test=pd.Series(np.concatenate(pred))
        y_valid=pd.Series(np.concatenate(val))
    test=abs(test)
    #getting size errors, therefore converting to float 32 vs float 16
    test=test.astype(np.float32)
    y_valid=y_valid.astype(np.float32)
    results=np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(test)))
    test=pd.concat([test,y_valid], axis=1)
    test.columns=['pred', 'valid']
    
    return modelz, test, results


