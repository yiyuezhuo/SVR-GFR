# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:07:43 2019

@author: yiyuezhuo
"""

from test_batch import df,experiment
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import numpy as np

from convert_latex_into_docx import LaTeX, generate_table
from test_batch import desc_format



model_map = {'linear_model': LinearRegression(),
             #'SVR': SVR(kernel='rbf', C=500, gamma=0.006, epsilon=0.1)
             #'SVR': SVR(kernel='rbf', C=500, gamma=0.018, epsilon=0.1)
}

def cut(d):
    return '{:.4f}'.format(d)

def linear_format(scale_method, sex_method, param):
    
    coef = param['coef']
    intercept = param['intercept']
    
    log_template = r'rGFR = {} \times {{{}}}^{{age}} \times Scr^{{{}}} \times Cys^{{{}}}'
    if sex_method == 'dummy':
        log_template += r' \times {{{}}}^{{sex}}'
    
    #linear_template = 'rGFR = {} + {} age + {} Scr + {} Cys'
    linear_template = 'rGFR = {} {} age {} Scr {} Cys'
    if sex_method == 'dummy':
        linear_template += '{} sex'
    
    d_list = [intercept.tolist()] + coef.tolist()
    p_list = []
    if scale_method == 'linear_scale':
        for i,d in enumerate(d_list):
            if d >=0 and i>0:
                p_list.append('+'+cut(d))
            else:
                p_list.append(cut(d))
    else:
        for i,d in enumerate(d_list):
            
            if i in [0,1,4]: # inter, age,Scr, Cys, sex
                d = np.exp(d)
            
            if d <0 and i==1:
            #assert d>= 0
            #if i == 1:
                p_list.append('(' + cut(d) + ')')
            else:
                p_list.append(cut(d))
    
    template = log_template if scale_method == 'log_scale' else linear_template
    
    return template.format(*p_list)
    

def equation_format(model_name, scale_method, sex_method, res):
    '''
    linear_model(dummy):
        linear scale:
            rGFR = inter + c[0] age + c[1] Scr + c[2] Cys + c[3] sex
        log scale:
            log rGFR = inter + c[0] age + c[1] log Scr + c[2] log Cys + c[3] sex
            rGFR = exp(inter) * exp(c[0])^age * Scr^c[1] * Cys^c[2] * exp(c[3])^sex
    '''
    if model_name == 'linear_model':
        if sex_method == 'sex2':
            equ_women = linear_format(scale_method, sex_method, res['param'][0])
            equ_men = linear_format(scale_method, sex_method, res['param'][1])
            equation = equ_women + '\n' + equ_men + '\n'
        else:
            equ = linear_format(scale_method, sex_method, res['param'])
            equation = equ + '\n'
        return equation
    else:
        return 'SVR model is too complex!\n'
    
if __name__ == '__main__':
    equation_list = []
    name_list = []
    
    with open('equations.csv', 'w') as f:
        #for model_name in ['linear_model', 'SVR']:
        for model_name in ['linear_model']:
            model = model_map[model_name]
            for scale_method in ['linear_scale', 'log_scale']:
                for sex_method in ['no_sex', 'dummy', 'sex2']:
                    res = experiment(model, df, scale_method, sex_method)
                    print(model_name ,scale_method, sex_method,res['train_mse'],res['test_mse'],res['mse'])
                    #f.write(line_format(model_name ,scale_method, sex_method,res['train_mse'],res['test_mse'],res['mse']))
                    content = equation_format(model_name, scale_method, sex_method, res)
                    f.write(content)
                    
                    equation_list.extend(content.split('\n')[:-1])
                    if sex_method != 'sex2':
                        name_list.append(desc_format(model_name, scale_method,
                            sex_method,None, disable_norm=True))
                    else:
                        name_list.append(desc_format(model_name,scale_method,
                            'women',None, disable_norm=True))
                        name_list.append(desc_format(model_name,scale_method,
                            'men',None, disable_norm=True))
    
    mat = [['model', 'equation']]
    for name, equation in zip(name_list, equation_list):
        mat.append([name, LaTeX(equation)])
    generate_table(mat, 'lr_table.docx')