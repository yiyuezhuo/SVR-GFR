import pandas as pd

df = pd.read_excel('数据1：共1329人，以SVR开发新模型.xlsx')
df.columns = ['sex', 'age', 'Cys', 'Scr', 'rGFR'] + list(df.columns[5:])
df['EPIcrcys'] = df['EPIcrcys'].astype(float)

