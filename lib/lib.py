import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objs as go

# ================================================================ GLOBAL =============================================================== #


months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'] 
month_num = {'Jan': 1, 'Fev': 2, 'Mar': 3, 'Abr': 4, 'Mai': 5, 'Jun': 6, 'Jul': 7, 'Ago': 8, 'Set': 9, 'Out': 10, 'Nov': 11 , 'Dez': 12}


# ========================================================== DATA MANIPULATION ========================================================== #


def group_sum(df, select='Valor', group=[], date='month'):
    if date == 'year':
        calendar = df['Data'].apply(lambda x : x.year)
        group.insert(0, calendar)
    elif date is not None:
        calendar = df['Data'].apply(lambda x : x.year*100 + x.month)
        group.insert(0, calendar)
    dg = df.groupby(group)[select].sum().to_frame()
    if len(group) > 1:
        dg = dg.unstack() 
    if date is not None:
        dg.index = map(lambda x : months[(x%100)-1] + '/' + str(int(x/100-2000)), dg.index)
    return dg[select] if len(group) > 1 else dg

def window_mean(df, select='Valor', window=3):
    dw = df.copy()
    dw['Avg'] = dw[select].rolling(window=window).mean()
    dw = dw.fillna(method='backfill')
    return dw

def row_percentage(df):
    return df.div(df.sum(axis=1), axis=0).apply(lambda x : x*100)


# ========================================================= AUXILIARY FUNCTIONS ========================================================= #


def calc_funds(earnings, spendings, transfers, check=[0, 0, 0, 0]):
    dtin = group_sum(transfers, group=['De'], date=None)
    dtin.columns = ['Transferido']

    dtout = group_sum(transfers, group=['Para'], date=None)
    dtout.columns = ['Recebido']


    dgb = group_sum(earnings, group=['Conta']).sum()
    dgb.index.name = None
    dgb = pd.DataFrame(dgb, columns=['Ganho'])

    dfb = group_sum(spendings, group=['Pagamento']).sum()
    dfb.index.name = None
    dfb = pd.DataFrame(dfb, columns=['Gasto'])


    dmov = dgb.join(dfb).join(dtin).join(dtout)
    dmov = dmov.fillna(0)
    dmov['Saldo'] = dmov['Ganho'] + dmov['Recebido'] - dmov['Transferido'] - dmov['Gasto']
    dmov['Lido'] = check
    dmov['Erro'] = dmov['Saldo'] - dmov['Lido']
    return dmov.loc[:, ['Saldo', 'Lido', 'Erro']]    


def calc_balance(earnings, spendings):
    spendings.columns = ['Gasto', 'Gasto médio']
    earnings.columns = ['Ganho', 'Ganho médio']
       
    fund = earnings.join(spendings, how='outer')
    fund['sort'] = fund.index.map(lambda x : int(x.split('/')[1]) * 100 + month_num[x.split('/')[0]])
    fund = fund.sort_values("sort", axis = 0, ascending = True) 
    fund.pop('sort')
    fund = fund.fillna(0)
    
    fund['Saldo'] = (fund['Ganho'] - fund['Gasto']).cumsum()
    fund['Var'] = fund['Ganho'] - fund['Gasto']
    fund['Crescimento'] = fund['Saldo'].rolling(2).apply(lambda x : 100*(x[1] - x[0])/x[0], raw=False)
    fund = fund.fillna(0)
    return fund


# ========================================================= DATA VIZUALIZATION ========================================================== #


def plot_stacked_area(df, y=None, fig=None, color='#FF5949', name=None):
    fig = go.FigureWidget() if fig is None else fig
    if type(df) == list:
        for i, d in enumerate(df):
            fig.add_trace(go.Scatter(x=d.index, y=d[y], marker=dict(color=color[i]), name=name[i], hoverinfo='y', stackgroup='one'))
    elif y is None:
        for i, c in enumerate(df.columns):
            fig.add_trace(go.Scatter(x=df.index, y=df[c], marker=dict(color=color[i]), name=name[i], hoverinfo='y', stackgroup='one'))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df[y], marker=dict(color=color), name=name, stackgroup='one'))
    return fig

def plot_bar(df, y, fig=None, color='#FF5949', name=None):
    fig = go.FigureWidget() if fig is None else fig
    if type(df) == list:
        for i, d in enumerate(df):
            fig.add_trace(go.Bar(x=d.index, y=d[y], marker=dict(color=color[i]), name=name[i], hoverinfo='y'))
    else:
        fig.add_trace(go.Bar(x=df.index, y=df[y], marker=dict(color=color), name=name))
    return fig

def plot_line(df, y, fig=None, color='#FF5949', name=None):
    fig = go.FigureWidget() if fig is None else fig
    if type(df) == list:
        for i, d in enumerate(df):
            fig.add_trace(go.Scatter(x=d.index, y=d[y], marker=dict(color=color[i]), name=name[i], hoverinfo='y'))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df[y], marker=dict(color=color), name=name))
    return fig

def plot_pie(df, colors=None):
    fig = go.FigureWidget()
    fig.add_trace(go.Pie(labels=df.index, values=df, hole=.4, opacity=.9,
                        marker=dict(colors=colors)))
    return fig

