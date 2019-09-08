# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:48:02 2019

@author: yiyuezhuo

Usage Example:
    df.apply(CKD_EPI_Cr, axis=1)

"""

def CKD_EPI_Cr(r):
    sex, Scr, age = r['sex'],r['Scr'],r['age']
    if sex == 2:
        if Scr <= 0.7:
            return 144*(Scr/0.7)**(-0.329) * 0.993**age
        else:
            return 144*(Scr/0.7)**(-1.209) * 0.993**age
    else:
        if Scr <= 0.9:
            return 141*(Scr/0.9)**(-0.411) * 0.993**age
        else:
            return 141*(Scr/0.9)**(-1.209) * 0.993**age

def CKD_EPI_Cys(r):
    sex, Cys, age = r['sex'],r['Cys'],r['age']
    if sex == 2:
        if Cys <= 0.8:
            return 133*(Cys/0.8)**(-0.499) * 0.996**age * 0.932
        else:
            return 133*(Cys/0.8)**(-1.328) * 0.996**age * 0.932
    else:
        if Cys <= 0.8:
            return  133*(Cys/0.8)**(-0.499) * 0.996**age
        else:
            return  133*(Cys/0.8)**(-1.328) * 0.996**age

def CKD_EPI_Cr_Cys(r):
    sex, Scr, Cys, age = r['sex'], r['Scr'], r['Cys'],r['age']
    if sex ==2:
        if Scr <= 0.7:
            if Cys <= 0.8:
                return 130*(Scr/0.7)**(-0.248) * (Cys/0.8)**(-0.375) * 0.995**age
            else:
                return 130*(Scr/0.7)**(-0.248) * (Cys/0.8)**(-0.711) * 0.995**age
        else:
            if Cys <= 0.8:
                return 130*(Scr/0.7)**(-0.601) * (Cys/0.8)**(-0.375) * 0.995**age
            else:
                return 130*(Scr/0.7)**(-0.601) * (Cys/0.8)**(-0.711) * 0.995**age
    else:
        if Scr <= 0.9:
            if Cys <= 0.8:
                return 135*(Scr/0.9)**(-0.207) * (Cys/0.8)**(-0.375) * 0.995**age
            else:
                return 135*(Scr/0.9)**(-0.207) * (Cys/0.8)**(-0.711) * 0.995**age
        else:
            if Cys <= 0.8:
                return 135*(Scr/0.9)**(-0.601) * (Cys/0.8)**(-0.375) * 0.995**age
            else:
                return 135*(Scr/0.9)**(-0.601) * (Cys/0.8)**(-0.711) * 0.995**age
    
def FAS_cr(r):
    Scr, age = r['Scr'],r['age']
    O_Cr = 0.7 if r['sex'] == 2 else 0.9
    t = 1. if age <= 40 else 0.988**(age-40)
    return 107.3/(Scr/O_Cr) * t

def FAS_Cys(r):
    Cys, age = r['Cys'],r['age']
    O_Cys = 0.82 if r['age'] <= 70 else 0.95
    t = 1.0 if age <= 40 else 0.988**(age-40)
    return 107.3/(Cys/O_Cys) * t

def FAS_Cr_Cys(r):
    Scr, Cys, age = r['Scr'],r['Cys'],r['age']
    O_Cr = 0.7 if r['sex'] == 2 else 0.9
    O_Cys = 0.82 if r['age'] <= 70 else 0.95
    t = 1.0 if age <= 40 else 0.988**(age-40)
    return 107.3/(0.5*(Scr/O_Cr)+0.5*(Cys/O_Cys)) * t

func_name_list = ['CKD_EPI_Cr', 'CKD_EPI_Cys', 'CKD_EPI_Cr_Cys',
             'FAS_cr', 'FAS_Cys', 'FAS_Cr_Cys']
func_map = {name: globals()[name] for name in func_name_list}

def apply_formulas(df):
    for func_name in func_name_list:
        df[func_name] = df.apply(func_map[func_name], axis=1)
    '''
    df['CKD_EPI_Cr'] = df.apply(CKD_EPI_Cr, axis=1)
    df['CKD_EPI_Cys'] = df.apply(CKD_EPI_Cys, axis=1)
    df['CKD_EPI_Cr_Cys'] = df.apply(CKD_EPI_Cr_Cys, axis=1)
    df['FAS_cr'] = df.apply(FAS_cr, axis=1)
    df['FAS_Cys'] = df.apply(FAS_Cys, axis=1)
    df['FAS_Cr_Cys'] = df.apply(FAS_Cr_Cys, axis=1)
    '''
    