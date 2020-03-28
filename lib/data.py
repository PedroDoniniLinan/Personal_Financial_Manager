import numpy as np
import pandas as pd
import datetime as dt

# ================================================= GENERAL =================================================== #

def date_trunc(date, unit=None):
    if unit == 'year':
        return date.apply(lambda x : x.year)
    else:
        return date.apply(lambda x : x.year*100 + x.month)

def format_date(date):
    months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'] 
    return date.apply(lambda x : months[(x%100)-1] + '/' + str(int(x/100-2000)))    

def pivot_agg(df, val, idx, agg, col=None, fill_value=0, date='Data'):
    dgroup = df.copy()
    if date is not None:
        dgroup[date] = date_trunc(df[date])
    dgroup = pd.pivot_table(dgroup, values=[val], index=idx, columns=col, aggfunc=agg, fill_value=0)[val].reset_index()
    if date is not None:
        dgroup[date] = format_date(dgroup[date])
    return dgroup 

def window(df, val='Valor', suffix='Avg', window=3, func=np.mean, fill='back'):
    dw = df.copy()
    dw[val + suffix] = dw[val].rolling(window=window).apply(func, raw=False)
    dw = dw.fillna(0) if fill == 0 else dw.fillna(method='backfill')
    return dw

def delta(x):
    x = x.reset_index(drop=True)
    return x[1] - x[0]

def int2pct(i):
    return str(round(i*100,2)) + '%'

def pct2int(p):
    return float(p.split('%')[0]) / 100

def row_percentage(df):
    return df.div(df.sum(axis=1), axis=0).apply(lambda x : x*100)
