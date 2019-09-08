# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:49:46 2019

@author: yiyuezhuo
"""
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need
import pickle
import os
import matplotlib.pyplot as plt

from test_batch import df
from format_equations import equation_format as _equation_format
from convert_latex_into_docx import LaTeX, generate_table
from test_batch import desc_format


def mse_loss(X, Y):
    return np.mean((X - Y)**2)

def gp_clean(model_name, scale_method, sex_method, g, p, normalize=False):
    s = lambda x:x.squeeze()
    
    if sex_method == 'sex2':
        g = [s(g[0]),s(g[1])]
        p = [s(p[0]),s(p[1])]
    else:
        g = s(g)
        p = s(p)
    
    if normalize:
        '''
        _equation_format take "direct" linear or log regression result,
        so if normalize is applied, we will need to recover it to
        "direct" format from standard form.
        
        y-y0 = b1 (x1'-x1'0) + b2 (x2'-x2'0)
        y = y0 - b1 x1'0 - b2 x2'0 + b1 x1' + b2 x2' 
        ->     
            (The above one is done before calling this function)
            p = y0 - b1 x1'0 - b2 x2'0
            g = (b1, b2)
        ->
        y = p + b1 x1' + b2 x2'     (standard form)
        y = p + b1 (x1-m1)/s1 + b2 (x2-m2)/s2 
        y = p - b1/s1 m1 - s2/s2 m2 + b1/s1 x1 + b2/s2 x2   
        ->     
            (The above one should be done before calling _equation_format)
        ->
        log k = b0 - b1/s1 m1 - b2/s2 m2 + b1/s1 log z1 + b2/s2 log z2
        ->
        k = exp(b0 - b1/s1 m1 - m2/s2 m2) + exp(b1/s1) z1 + exp(b2/s2) z2
        '''
        if sex_method == 'dummy':
            X = df[['age', 'Scr', 'Cys', 'sex']]
        else:
            X = df[['age', 'Scr', 'Cys']]
        
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        if sex_method in ('dummy', 'no_sex', 'men', 'women'):
            g /= std
            p -= np.sum((g/std)*mean)
        else:
            g[0] /= std
            g[1] /= std
            p[0] -= np.sum((g[0]/std)*mean)
            p[1] -= np.sum((g[1]/std)*mean)
        
    return g,p

def equation_format(model_name, scale_method, sex_method, g, p, normalize=False):
    fake_model_name = 'linear_model'
    g,p = gp_clean(model_name, scale_method, sex_method, g, p, normalize = normalize)
        
    if sex_method == 'sex2':
        fake_res = {'param':({'coef': g[0], 'intercept': p[0]},
                             {'coef': g[1], 'intercept': p[1]})
        }
    else:
        fake_res = {'param': {'coef': g, 'intercept': p}}
    return _equation_format(fake_model_name, scale_method, sex_method, fake_res)

def rbf(X,Y, gamma):
    # X: (num_samples1, num_features)
    # Y: (num_samples2, num_features)
    # return: (num_samples1, num_samples2)
    
    
    X_ = np.expand_dims(X, 1) # unsqueeze -> (num_samples1, 1, num_features)
    Y_ = np.expand_dims(Y, 0) # unsqueeze -> (1, num_samples2, num_features)
    dm = X_ - Y_ # (num_samples1, num_samples2, num_features)
    
    norm = np.sum(dm**2, axis=2)
    return np.exp(-gamma * norm)

def predict(X, support_vectors, dual_coef, intercept, gamma=0.006):
    kernel_mat = rbf(X, support_vectors, gamma)
    pred = kernel_mat @ dual_coef.transpose() + intercept
    return pred


#dfp = df.loc[:df.shape[0]-1]
#dfp = df[:df.shape[0]-1]
dfp = df.copy()
Xp = np.stack([dfp['age'], np.log(dfp['Scr']), np.log(dfp['Cys']), dfp['sex']], 1)
#yp = np.log(dfp['rGFR'])

X = df[['age', 'Scr', 'Cys', 'sex']].values

mean_Xp = Xp.mean(0, keepdims=True)
mean_X = X.mean(0, keepdims=True)
is_women = df['sex'] == 2

X_map_unnormalized = {
    'linear_scale':{
        'no_sex': X[:,:-1],
        'dummy': X,
        #'sex2': (X[is_women,:-1], X[~is_women,:-1]), 
        'women': X[is_women,:-1],
        'men': X[~is_women,:-1]
    },
    'log_scale':{
        'no_sex': Xp[:,:-1],
        'dummy': Xp,
        #'sex2':(Xp[is_women,:-1], Xp[~is_women,:-1])
        'women': Xp[is_women,:-1],
        'men': Xp[~is_women,:-1]
    }
}
    
X_map = {}
for scale_method in ['linear_scale', 'log_scale']:
    X_map[scale_method] = {}
    for sex_method in ['no_sex', 'dummy', 'women', 'men']:
        xmu = X_map_unnormalized[scale_method][sex_method]
        X_map[scale_method][sex_method] = {
            'norm': (xmu - xmu.mean(axis=0))/xmu.std(axis=0),
            'nonorm': xmu,
        }
        
X_mean_map = {}
X_std_map = {}
for scale_method in ['linear_scale', 'log_scale']:
    X_mean_map[scale_method] = {}
    X_std_map[scale_method] = {}
    for sex_method in ['no_sex', 'dummy', 'women', 'men']:
        xm = X_map[scale_method][sex_method]
        X_mean_map[scale_method][sex_method] = {
            'norm': xm['norm'].mean(axis=0, keepdims=True),
            'nonorm': xm['nonorm'].mean(axis=0, keepdims=True),
        }
        X_std_map[scale_method][sex_method] = {
            'norm': xm['norm'].std(axis=0, keepdims=True),
            'nonorm': xm['nonorm'].std(axis=0, keepdims=True),
        }



if __name__ == '__main__':
    
    equation_list = []
    df_list = []
    disable_norm = True
    
    with open('equations_taylor.csv', 'w') as f:
        for model_name in ['SVR']:
            #model = model_map[model_name]
            for scale_method in ['linear_scale', 'log_scale']:
                for sex_method in ['no_sex', 'dummy', 'men', 'women']:
                    for normalize in [True, False]:
                        name = '{}-{}-{}-{}'.format(model_name, scale_method, sex_method, 'norm' if normalize else 'nonorm')
                        path = os.path.join('models', name)
                        with open(path, 'rb') as ff:
                            model = pickle.load(ff)
                        _predict = lambda X:predict(X, model.support_vectors_, model.dual_coef_, model._intercept_, gamma = model.gamma)
                        d_predict = grad(_predict)
                        
                        if sex_method == 'sex2':
                            _X = X_mean_map[scale_method]['women']['norm' if normalize else 'nonorm']
                            g1 = d_predict(_X)
                            p1 = _predict(_X) - (_X * g1).sum()
                            print(name+' women',p1, g1, _predict(_X), - (_X * g1).sum())

                            _X = X_mean_map[scale_method]['men']['norm' if normalize else 'nonorm']
                            g2 = d_predict(_X)
                            p2 = _predict(_X) - (_X * g2).sum()
                            g,p = (g1,g2),(p1,p2)
                            print(name+' men',p2, g2, _predict(_X), -(_X * g2).sum())

                        else:
                            _X = X_mean_map[scale_method][sex_method]['norm' if normalize else 'nonorm']
                            g = d_predict(_X)
                            p = _predict(_X) - (_X * g).sum()
                            print(name,p,g, _predict(_X), - (_X * g).sum())

                        content = equation_format(model_name, scale_method, 
                                    sex_method, g, p, normalize = normalize)
                        f.write(content)
                        
                        equation_list.extend(content.split('\n')[:-1])
                        df_list.append(np.prod(model.dual_coef_.shape) +\
                            np.prod(model.support_vectors_.shape) + 1)
                    
    scale_method = 'log_scale'
    sex_method = 'dummy'
    normalize = True
    
    name = '{}-{}-{}-{}'.format(model_name, scale_method, sex_method, 'norm' if normalize else 'nonorm')
    path = os.path.join('models', name)
    with open(path, 'rb') as ff:
        model = pickle.load(ff)
    _predict = lambda X:predict(X, model.support_vectors_, model.dual_coef_, 
                                model._intercept_, gamma = model.gamma)
    d_predict = grad(_predict)
    
    parames = ['intercept','age', 'Scr', 'Cys', 'sex']
    exp_list = [True, True, False, False, True]
    
    
    #fig, axs = plt.subplots(4,5 ,figsize=(18,9))
    fig, axs = plt.subplots(4,5 ,figsize=(18,9))
    
    # Unnormalized log-scale value
    for i, vl in enumerate([np.linspace(30.0, 70.0, 100),
                             np.linspace(-1.0, 2.5, 100),
                             np.linspace(-1.0, 2.0, 100),
                             np.linspace(1.0, 2.0, 100)]):
        # Ha.. Sex have 0 values... (75) Shocked.. So my model are all broken.
        mat = []
        
        vl_origin = vl.copy()
        if normalize:
            _mean = X_mean_map[scale_method][sex_method]['nonorm'][0,i]
            _std =  X_std_map[scale_method][sex_method]['nonorm'][0,i]
            vl = (vl - _mean)/_std
            
        for v in vl: # age
            _X = X_mean_map[scale_method][sex_method]['norm' if normalize else 'nonorm'].copy()
            _X[0,i] = v
            g = d_predict(_X)
            p = _predict(_X) - (_X * g).sum()
            g,p = gp_clean(model_name, scale_method, sex_method, g, p, normalize=normalize)
            arr = np.concatenate([[p],g])
            mat.append(arr)
        mat = np.array(mat)
        
        _X = X_mean_map[scale_method][sex_method]['norm' if normalize else 'nonorm'].copy()
        _X_origin = X_mean_map[scale_method][sex_method]['nonorm'].copy()
        g = d_predict(_X)
        p = _predict(_X) - (_X * g).sum()
        g,p = gp_clean(model_name, scale_method, sex_method, g, p, normalize=normalize)
        base = np.concatenate([[p],g])
        
        for j in range(5):
            '''
            if exp_list[j]:
                y  = np.exp(mat[:,j])
                yb = np.exp([base[j]])
                
            else:
                y  = mat[:,j]
                yb = [base[j]]
            
            if i in [1,2]:
                x  = np.exp(vl)
                xb = np.exp([_X[0][i]])
            else:
                x  = vl
                xb = [_X[0][i]]
            '''
            if exp_list[j]:
                y  = np.exp(mat[:,j])
                yb = np.exp([base[j]])
                
            else:
                y  = mat[:,j]
                yb = [base[j]]
            
            if i in [1,2]:
                x  = np.exp(vl_origin)
                xb = np.exp([_X_origin[0][i]])
            else:
                x  = vl_origin
                xb = [_X_origin[0][i]]

                
            axs[i][j].plot(x,y)
            axs[i][j].plot(xb,yb,'o')
            #axs[i,j].get_xaxis().get_major_formatter().set_scientific(False)
            axs[i,j].ticklabel_format(useOffset=False, style='plain')
            # disable scientific notation and "offset", see:
            # https://stackoverflow.com/questions/28371674/prevent-scientific-notation-in-matplotlib-pyplot

    
    cols = ['intercept','age', 'Scr', 'Cys', 'sex']
    rows = ['age', 'Scr', 'Cys', 'sex']
    
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    
    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(row, rotation=0, size='large')
    
    fig.tight_layout()
    plt.show()

    # verify
    print('verify mse_loss: {}'.format(mse_loss(np.exp(_predict(X_map[scale_method][sex_method]['norm'])).squeeze(), df['rGFR'])))
    
    age = df['age']
    Scr = df['Scr']
    Cys = df['Cys']
    sex = df['sex']
    
    pred_approx = np.exp(p) * np.exp(g[0])**age * Scr **g[1] * Cys**g[2] * np.exp(g[3])**sex # come from base
    print('approx mse_loss: {}'.format(mse_loss(pred_approx, df['rGFR'])))
    
    #raise Exception
    '''
    X_input = X_map[scale_method][sex_method]['norm' if normalize else 'nonorm']
    if sex_method in ['women','men']:
        _y = df[is_women if sex_method == 'women' else ~is_women]['rGFR']
    else:
        _y = df['rGFR']
    
    y_pred = p + X_input @ g[:,np.newaxis]
    y_pred = np.exp(y_pred)
    print('approx mse_loss2', mse_loss(y_pred.squeeze(), df['rGFR']))
    '''
    
    predictor_loss_list = []
    approx_loss_list = []
    name_list = []
    
    #raise Exception
    # calculating approx loss
    with open('log_taylor.csv', 'w') as f:
        for model_name in ['SVR']:
            #model = model_map[model_name]
            for scale_method in ['linear_scale', 'log_scale']:
                for sex_method in ['no_sex', 'dummy', 'women', 'men']:
                    for normalize in [True, False]:
                        name = '{}-{}-{}-{}'.format(model_name, scale_method, sex_method, 'norm' if normalize else 'nonorm')
                        path = os.path.join('models', name)
                        with open(path, 'rb') as ff:
                            model = pickle.load(ff)
                        _predict = lambda X:predict(X, model.support_vectors_, model.dual_coef_, model._intercept_, gamma = model.gamma)
                        d_predict = grad(_predict)
                        

                        if sex_method in ['women','men']:
                            _y = df[is_women if sex_method == 'women' else ~is_women]['rGFR']
                        else:
                            _y = df['rGFR']
                            
                        # origin nonlinear loss
                        X_input = X_map[scale_method][sex_method]['norm' if normalize else 'nonorm']

                        y_pred = _predict(X_input)
                        if scale_method == 'log_scale':
                            y_pred = np.exp(y_pred)
                        print(name, 'SVR mse', mse_loss(y_pred.squeeze(), _y))
                        predictor_loss_list.append(mse_loss(y_pred.squeeze(), _y))
                        
                        # approx loss
                        _X = X_mean_map[scale_method][sex_method]['norm' if normalize else 'nonorm'].copy()
                        g = d_predict(_X)
                        p = _predict(_X) - (_X * g).sum()
                        g,p = gp_clean(model_name, scale_method, sex_method, g, p, normalize=normalize)
                        
                        Xi = X_map[scale_method][sex_method]['nonorm']
                        #y_pred = p + X_input @ g.squeeze()[:,np.newaxis]
                        y_pred = p + Xi @ g[:,np.newaxis]
                        # mse_loss(np.exp(p + Xi @ g[:,np.newaxis]), df['rGFR'])
                        if scale_method == 'log_scale':
                            y_pred = np.exp(y_pred)
                        
                        print(name, 'SVR approx mse', mse_loss(y_pred.squeeze(), _y))
                        approx_loss_list.append(mse_loss(y_pred.squeeze(), _y))
                        #name_list.append(name)
                        name_list.append(desc_format(model_name, scale_method, 
                            sex_method, normalize, disable_norm = disable_norm))
                        
                        #raise Exception
                        
    mat = [['model','pMSE','aMSE','df','approx']]
    f = lambda s:'{:.2f}'.format(s)
    for i, (name, pmse, amse, deg_f, equations) in enumerate(zip(name_list, predictor_loss_list,
                                approx_loss_list, df_list, equation_list)):
        if disable_norm and i % 2 ==1:
            continue # skip non-norm result
        row = [name, f(pmse), f(amse), str(deg_f), LaTeX(equations)]
        mat.append(row)
    generate_table(mat, 'svr_table.docx')
