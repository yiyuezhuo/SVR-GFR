# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 20:58:35 2019

@author: yiyuezhuo
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os

from empirical_formulas import apply_formulas

def checkpoint(model, name):
    #name = '{}-{}-{}-{}'.format(model_name, scale_method, sex_method, 'norm' if normalize else 'nonorm')
    os.makedirs("models", exist_ok=True)
    path = os.path.join('models',name)
    with open(path, 'wb') as ff:
        pickle.dump(model, ff)

def capture_parameter(model):
    if isinstance(model, LinearRegression):
        return dict(coef = model.coef_.copy(),
                    intercept = model.intercept_.copy())
    elif isinstance(model, SVR):
        return dict(support_vectors = model.support_vectors_.copy(),
                    dual_coef = model.dual_coef_.copy(),
                    intercept = model._intercept_.copy())
    raise NotImplementedError


def fit_predict(model, X, y, random_state=51):
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = random_state)
    
    #model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    model.fit(X, y)
    y_pred = model.predict(X)
    
    return dict(y_train_pred = y_train_pred,  # those value may be np.exp(.)
                y_test_pred = y_test_pred, 
                y_pred = y_pred,
                y_train = y_train, 
                y_test = y_test,
                y = y)

def experiment(model, df, scale_method, sex_method, random_state=51,
               normalize = False, normalize_y = False, model_name='model_name',
               cache = False):
    '''
    scale_method: ['linear_scale', 'log_scale']
    sex_method: ['no_sex', 'dummy', 'sex2']
    '''
    if scale_method == 'linear_scale':
        y = df['rGFR']
        if sex_method == 'dummy':
            X = df[['age', 'Scr', 'Cys', 'sex']]
        else:
            X = df[['age', 'Scr', 'Cys']]
        
    else:
        #df = df.loc[:df.shape[0]-1]
        y = np.log(df['rGFR'])
        if sex_method == 'dummy':
            X = np.stack([df['age'], np.log(df['Scr']), np.log(df['Cys']), df['sex']], 1)
        else:
            X = np.stack([df['age'], np.log(df['Scr']), np.log(df['Cys'])], 1)
            
    if normalize:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean)/std
    if normalize_y:
        raise NotImplementedError
            
    if sex_method in ['no_sex', 'dummy']:
        res = fit_predict(model, X, y, random_state = random_state)
        param = capture_parameter(model)
        if cache:
            name = '{}-{}-{}-{}'.format(model_name, scale_method, sex_method, 'norm' if normalize else 'nonorm')
            checkpoint(model, name)

    else:
        wmask = df['sex'] == 2
        X_women = X[wmask]
        y_women = y[wmask]
        X_men = X[~wmask]
        y_men = y[~wmask]
        
        res_women = fit_predict(model, X_women, y_women, random_state = random_state)
        param_women = capture_parameter(model)
        if cache:
            name = '{}-{}-{}-{}'.format(model_name, scale_method, 'women', 'norm' if normalize else 'nonorm')
            checkpoint(model, name)
        
        res_men = fit_predict(model, X_men, y_men, random_state = random_state)
        param_men = capture_parameter(model)
        if cache:
            name = '{}-{}-{}-{}'.format(model_name, scale_method, 'men', 'norm' if normalize else 'nonorm')
            checkpoint(model, name)
    
        param = [param_women, param_men]
        res = {key: np.concatenate([res_women[key],res_men[key]]) for key in res_women.keys()}
    
    #print(scale_method)
    if scale_method == 'log_scale':
        #print('apply exp')
        res = {key:np.exp(value) for key,value in res.items()}
        
    train_mse = np.mean((res['y_train_pred'] - res['y_train'])**2)
    test_mse = np.mean((res['y_test_pred'] - res['y_test'])**2)
    mse = np.mean((res['y_pred'] - res['y'])**2)
    
    R = np.corrcoef(res['y_pred'],res['y'])[0,1]
    R_train = np.corrcoef(res['y_train_pred'],res['y_train'])[0,1]
    R_test = np.corrcoef(res['y_test_pred'],res['y_test'])[0,1]
    
    return dict(train_mse = train_mse, test_mse = test_mse, mse = mse,
                res = res, param=param, X = X, normalize =normalize,
                R = R, R_train=R_train, R_test = R_test)

'''
df = pd.read_excel('GFR-SVR数据.xls')
df.columns = ['id', 'age', 'sex', 'rGFR', 'Scr', 'Cys']
df['Cys'] = pd.to_numeric(df['Cys'], errors='coerce')
df=df.dropna()
df = df[(18<df['age']) & (df['age']<=100) & (1<=df['sex']) & (df['sex']<=2) & (5 <= df['rGFR']) & (df['rGFR'] <= 150) & \
       (0.0 <= df['Scr']) & (df['Scr'] <= 3000) & (0.2 <= df['Cys']) & (df['Cys'] < 5.0)]
'''
from preprocessing import df


model_map = {'linear_model': LinearRegression(),
             #'SVR': SVR(kernel='rbf', gamma = 1e-05, C = 400, epsilon = 0.025)}
             #'SVR': SVR(kernel='rbf', C=500, gamma=0.006, epsilon=0.1) # best
             #'SVR': SVR(kernel='rbf', C=500, gamma=0.012, epsilon=0.1)
             #'SVR': SVR(kernel='rbf', C=500, gamma=0.003, epsilon=0.1)
             #'SVR': SVR(kernel='rbf', C=500, gamma=0.024, epsilon=0.1)
             #'SVR': SVR(kernel='rbf', C=10.0, gamma='auto', epsilon=0.1)
             'SVR': SVR(kernel='rbf', C=100.0, gamma=0.1, epsilon=0.05)
             #'SVR': SVR(kernel='rbf', C=500.0, gamma='auto', epsilon=0.1)
             #'SVR': SVR(kernel='rbf', C=500, gamma=0.018, epsilon=0.1)
} 

def desc_format(model_name, scale_method, sex_method, normalize, disable_norm = False):
    model_name = {'linear_model':'lr','SVR':'SVR'}[model_name]
    scale_method = {'linear_scale':'lin', 'log_scale':'log'}[scale_method]
    if not disable_norm:
        desc = '{}+{}+{}+{}'.format(model_name, scale_method, sex_method, 'norm' if normalize else 'nonorm')
    else:
        desc = '{}+{}+{}'.format(model_name, scale_method, sex_method)
    return desc

def line_format(model_name, scale_method, sex_method,normalize, train_mse, test_mse, mse, R, R_train,R_test, disable_norm=False):
    desc = desc_format(model_name, scale_method, sex_method, normalize, disable_norm = disable_norm)
    line = '{},{},{},{},{},{},{}\n'.format(desc, mse, train_mse, test_mse,R,R_train,R_test)
    return line

if __name__ == '__main__':
    random_state = 8964
    disable_norm = True
    
    with open('log.csv', 'w') as f:
        for model_name in ['linear_model', 'SVR']:
            model = model_map[model_name]
            for scale_method in ['linear_scale', 'log_scale']:
                for sex_method in ['no_sex', 'dummy', 'sex2']:
                    for normalize in [True, False]:
                        #res = experiment(model, df, scale_method, sex_method)
                        res = experiment(model, df, scale_method, sex_method, 
                                         normalize=normalize, random_state = random_state,
                                         model_name = model_name, cache=True)
                        print(model_name ,scale_method, sex_method,res['train_mse'],res['test_mse'],res['mse'],res['R'],res['R_train'],res['R_test'])
                        if disable_norm and not normalize:
                            continue
                        f.write(line_format(model_name ,scale_method, sex_method,
                            normalize,res['train_mse'],res['test_mse'],res['mse'],res['R'],res['R_train'],res['R_test'],
                            disable_norm = disable_norm))
                        
                        #name = '{}-{}-{}'.format(model_name, scale_method, sex_method)
                        '''
                        name = '{}-{}-{}-{}'.format(model_name, scale_method, sex_method, 'norm' if normalize else 'nonorm')
                        path = os.path.join('models',name)
                        with open(path, 'wb') as ff:
                            pickle.dump(model, ff)
                        '''
    
    apply_formulas(df)
    gt = df['rGFR']
    emp_res = df['CKD_EPI_Cr_Cys']
    lr_res = experiment(model_map['linear_model'], df, 'log_scale','dummy')['res']
    svr_res = experiment(model_map['SVR'], df, 'log_scale','dummy', normalize=True)['res']
    
    plt.figure(figsize=(16,9))
    plt.plot(np.sort(gt), np.sort(gt),color='black')
    plt.plot(gt,emp_res,'o',alpha=0.2,label='CKD_EPI_Cr_Cys',color='r')
    plt.plot(lr_res['y'],lr_res['y_pred'],'o',alpha=0.2,label='lr',color='g')
    plt.plot(svr_res['y'],svr_res['y_pred'],'o',alpha=0.2,label='SVR',color='b')
    plt.legend()
    plt.show()
    
    '''
    #y = df['rGFR']
    #X = df[['age', 'Scr', 'Cys', 'sex']]
    dfp = df.loc[:df.shape[0]-1]
    X = np.stack([dfp['age'], np.log(dfp['Scr']), np.log(dfp['Cys']), dfp['sex']], 1)
    y = np.log(dfp['rGFR'])
    
    svr = model_map['SVR']
    gamma = svr.gamma # used in rbf
    def rbf(X,Y):
        # X: (num_samples1, num_features)
        # Y: (num_samples2, num_features)
        # return: (num_samples1, num_samples2)
        
        
        X_ = np.expand_dims(X, 1) # unsqueeze -> (num_samples1, 1, num_features)
        Y_ = np.expand_dims(Y, 0) # unsqueeze -> (1, num_samples2, num_features)
        dm = X_ - Y_ # (num_samples1, num_samples2, num_features)
        
        norm = np.sum(dm**2, axis=2)
        return np.exp(-gamma * norm)
        
    support_vectors_ = svr.support_vectors_
    dual_coef_ = svr.dual_coef_
    _intercept_ = svr._intercept_
    
    kernel_mat = rbf(X, support_vectors_)
    pred = kernel_mat @ svr.dual_coef_.transpose() + _intercept_
    
    plt.plot(np.exp(y), np.exp(pred), 'o')
    plt.plot(np.exp(y), np.exp(y))
    plt.show()
    
    print('mathced?',np.all(np.abs(np.exp(pred).flatten() - svr_res['y_pred']) < 1e-6))
    
    import pickle
    
    
    '''