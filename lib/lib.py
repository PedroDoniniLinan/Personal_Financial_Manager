import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objs as go

months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'] 

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
#     print(dg)
    if date is not None:
        dg.index = map(lambda x : months[(x%100)-1] + '/' + str(int(x/100-2000)), dg.index)
    return dg  

def window_mean(df, select='Valor', window=3):
    dw = df.copy()
    dw['Avg'] = dw[select].rolling(window=window).mean()
    dw = dw.fillna(method='backfill')
    return dw

def row_percentage(df):
    return df.div(df.sum(axis=1), axis=0).apply(lambda x : x*100)

def plot_stacked_area(data):
    fig = go.FigureWidget()
    i = 0
    for c in data.columns.levels[1]:
        fig.add_trace(go.Scatter(x=data.index, y=data['Valor'][c], 
                        hoverinfo='y+name',
                        mode='lines',
                        name = c,
                        line=dict(width=0.5, color='rgb(' + str(180*(i%2)+30*(i%5)) + ', ' + str(90*(i%2)+30*(i%4)) + ', ' + str(30*(i%2)+40*(i%3)) + ')'),
                        stackgroup='one' # define stack group
                    ))
        i += 1
    return fig

def plot_bar(data, fig=None, dtype='Gasto'):
    fig = go.FigureWidget() if fig is None else fig
    colors = ['#FF5949' if dtype == 'Gasto' else '#01FF49', '#800B00' if dtype == 'Gasto' else '#268040']
    fig.add_trace(go.Bar(x=data.index, y=data['Valor'], marker=dict(color=colors[0]), name=(dtype)))
    fig.add_trace(go.Scatter(x=data.index, y=data['Avg'], marker=dict(color=colors[1]), name=(dtype +' médio')))
    return fig

def plot_pie(df):
    fig = go.FigureWidget()
    fig.add_trace(go.Pie(labels=df.index.levels[1], values=df['Valor'], name="Matriz", hole=.4, opacity=.9))
    return fig

