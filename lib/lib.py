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
    dw[select + 'Avg'] = dw[select].rolling(window=window).mean()
    dw = dw.fillna(method='backfill')
    return dw

def window_delta(df, select='Valor'):
    df[select] = df[select].rolling(window=2).apply(lambda x : x[1] - x[0], raw=True)
    return df

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


def calc_flows(df):
    o = df.head(1)['Valor'].item()
    f = df.tail(1)['Valor'].item()
    od = df.head(1)['Data'].item()
    fd = df.tail(1)['Data'].item()
    status = df.tail(1)['Status'].item()
    itype = df.tail(1)['Tipo'].item()
    active = True
    if status == 'Fechado':
        active = False
    gain = f-o
    duration = (fd - od).days
    total_yield = gain / o
    daily_yield = (1+total_yield)**(1/duration)-1 if duration > 0 else 0
    monthly_yield = (1+daily_yield)**30.5-1 
    annual_yield = (1+daily_yield)**365-1 
    year_profit = f * annual_yield
    dr = pd.DataFrame({'Valor inicial': [o], 
                       'Ganho': [gain], 
                       'Meses': [duration / 30.5], 
                       'Rendimento': [int2pct(total_yield)],  
                       'Rendimento mensal': [int2pct(monthly_yield)], 
                       'Rendimento anual': [int2pct(annual_yield)],
                       'Ganho em 1 ano': [year_profit],
                       'Ativo':[active], 
                       'Tipo': [itype]})
    return dr
    
def calc_flow_gains(df):
    dflow = df.groupby('ID').apply(window_delta)
    dflow.pop('ID')
    dflow = dflow.reset_index(drop=True)
    dflow = dflow[dflow['Status'] != 'Aberto']
    dflow.pop('Status')
    return dflow    


def change_df_prop(df, font=22, align='center'):
    heading_properties = [('font-size', str(font-2) + 'px')]
    cell_properties = [('font-size', str(font) + 'px'), ('text-align', align)]

    dfstyle = [dict(selector="th", props=heading_properties),
     dict(selector="td", props=cell_properties)]
    return df.style.set_table_styles(dfstyle).hide_index()


def join_prices(row, dprices):
    di = dprices[dprices['Ticker'] == row['Ticker']]
    for c in di.columns:
        if c == 'Ticker':
            continue
        month = dt.datetime.strptime(c, '%y-%m-%d').date()
        if month.month < row['Date'].month:
            row[c] = 0
        else:
            row[c] = di[c].item()
    return row

def generate_invest_table(dport, dprices):
    dport = dport.apply(lambda x : join_prices(x, dprices), axis=1)
    dinvest = pd.DataFrame(columns=['ID','Valor','Data','Status','Tipo'])
    for i in range(dport.shape[0]):
        di = pd.DataFrame([[float(100+i), dport.iloc[i]['Buy']*dport.iloc[i]['Shares'], dport.iloc[i]['Date'], 'Aberto', dport.iloc[i]['Ticker']]], 
                          columns=['ID','Valor','Data','Status','Tipo'])
        dinvest = pd.concat([dinvest, di])
        for j in range(5, dport.shape[1]):
            date = dt.datetime.strptime(dport.columns[j], '%y-%m-%d').date()
            if date > dt.date.today():
                date = dt.date.today()
            if date > dport.iloc[i]['Date']:
                dj = pd.DataFrame([[float(100.0+i), dport.iloc[i, j]*dport.iloc[i]['Shares'], date, 'Mes', dport.iloc[i]['Ticker']]], 
                                  columns=['ID','Valor','Data','Status','Tipo'])
                dinvest = pd.concat([dinvest, dj])
    return dinvest


def calc_stocks_feat(dreturn, dstocks, dquote, assets, industry):    
    dfull = dreturn.join(dstocks.set_index('Ticker'), on='Ticker', how='inner')
    dfull['Price'] = dquote['2019']

    dfull['Geo'] = dfull[assets].idxmax(axis=1)
    dfull['Domain'] = dfull[industry].idxmax(axis=1)
    dfull['Asset'] = dfull[assets].idxmax(axis=1).apply(lambda x : x if (x == 'Bonds' or x == 'Commod') else 'Equity')

    dfull['Risk'] = dfull['Volatility'].apply(lambda x : int2pct(x/100))
    dfull['Tax'] = dfull['TER'].apply(lambda x : int2pct(x/100))
    dfull['Return'] = dfull['Predicted'].apply(int2pct)
    dfull['YTD'] = dfull[str(dt.date.today().year)].apply(int2pct)
    weights = [0.05, 0.5, 0.35, 0.1]
    dfull['Score'] = (10+(30-dfull['Price'])/27)*weights[0] + dfull['Predicted']/0.3*10*weights[1] + (10+(7-dfull['Volatility'])/2.3)*weights[2] + (10-dfull['TER']/0.2)*weights[3]

    return dfull


def calc_portfolio(dport, dfull, assets, industry):
    dport_full = dport.join(dfull[['Ticker', 'Price'] + assets + industry].set_index('Ticker'), on='Ticker')

    dport_full['Profit'] = (dport_full['Price'] - dport_full['Buy']) * dport_full['Shares']
    dport_full['Value'] = (dport_full['Price'] * dport_full['Shares'])

    dport_full['Yield'] = (dport_full['Profit'] / (dport_full['Buy'] * dport_full['Shares']))
    dport_full['Annual yield'] = (((1 + dport_full['Yield']) ** (1/(dt.date.today() - dport_full['Date']).apply(lambda x : x.days))) ** 365 - 1)
    dport_full['Year profit'] = dport_full['Price'] * dport_full['Shares'] * dport_full['Annual yield']

    dport_full['Keep'] = dport_full['Ticker'].apply(lambda x : dfull[dfull['Ticker'] == x]['TER'].item()) * dport_full['Price'] * dport_full['Shares'] / 100
    dport_full['Sell Tax'] = (dport_full['Profit'] - (dport_full['Buy Tax'] + dport_full['Keep'])) * 0.15
    dport_full['Total Tax'] = dport_full['Buy Tax'] + dport_full['Keep'] + dport_full['Sell Tax']
    dport_full['Tax'] = (dport_full['Total Tax'] / dport_full['Profit']).apply(int2pct)

    dport_full['Yield (Liq.)'] = ((dport_full['Profit'] - dport_full['Total Tax']) / (dport_full['Buy'] * dport_full['Shares']))
    dport_full['Annual yield (Liq.)'] = (((1 + dport_full['Yield (Liq.)']) ** (1/(dt.date.today() - dport_full['Date']).apply(lambda x : x.days))) ** 365 - 1)
    dport_full['Year profit (Liq.)'] = dport_full['Price'] * dport_full['Shares'] * dport_full['Annual yield (Liq.)']

    dport_full['Yield %'] = dport_full['Yield'].apply(int2pct)
    dport_full['Annual yield %'] = dport_full['Annual yield'].apply(int2pct)
    dport_full['Yield (Liq.) %'] = dport_full['Yield (Liq.)'].apply(int2pct)
    dport_full['Annual yield (Liq.) %'] = dport_full['Annual yield (Liq.)'].apply(int2pct)

    for c in industry:
        dport_full[c] = dport_full[c] * dport_full['Value'] / dport_full['Value'].sum()
    
    return dport_full

def add_bonds(dport_full, dflow, assets):
    profit = dflow['Ganho'].sum()
    value = dflow[dflow['Ativo']]['Valor inicial'].sum() + dflow[dflow['Ativo']]['Ganho'].sum()
    yields = profit / dflow['Valor inicial'].sum()
    di = pd.DataFrame([['NU', value, profit, yields, int2pct(yields), 0, 0, 0, 0, 100, 0, 0]], columns=['Ticker', 'Value', 'Profit', 'Yield', 'Yield %'] + assets)
    dport_bonds = pd.concat([dport_full, di], sort=False).reset_index(drop=True)
    return dport_bonds

def calc_distribution(dport, assets, industry):
    for c in assets:
        dport[c] = dport[c] * dport['Value'] / dport['Value'].sum()

    return dport


def int2pct(i):
    return str(round(i*100,2)) + '%'

def pct2int(p):
    return float(p.split('%')[0]) / 100

# ========================================================= DATA VIZUALIZATION ========================================================== #


def plot_stacked_area(df, y=None, fig=None, color='#FF5949', domain=dict(x=[0,1]), name=None):
    fig = go.FigureWidget() if fig is None else fig
    fig.layout.template = 'plotly_dark'
    if type(df) == list:
        for i, d in enumerate(df):
            fig.add_trace(go.Scatter(x=d.index, y=d[y], marker=dict(color=color[i]), name=name[i], hoverinfo='y', stackgroup='one'))
            # fig.add_trace(go.Scatter(x=d.index, y=d[y], marker=dict(color=color[i]), domain=domain, name=name[i], hoverinfo='y', stackgroup='one'))
    elif y is None:
        for i, c in enumerate(df.columns):
            fig.add_trace(go.Scatter(x=df.index, y=df[c], marker=dict(color=color[i]), name=name[i], hoverinfo='y', stackgroup='one'))
            # fig.add_trace(go.Scatter(x=df.index, y=df[c], marker=dict(color=color[i]), domain=domain, name=name[i], hoverinfo='y', stackgroup='one'))
    else:
        # fig.add_trace(go.Scatter(x=df.index, y=df[y], marker=dict(color=color), domain=domain, name=name, stackgroup='one'))
        fig.add_trace(go.Scatter(x=df.index, y=df[y], marker=dict(color=color), name=name, stackgroup='one'))
    return fig

def plot_bar(df, y, fig=None, color='#FF5949', domain=dict(x=[0,1]), name=None):
    fig = go.FigureWidget() if fig is None else fig
    fig.layout.template = 'plotly_dark'
    if type(df) == list:
        for i, d in enumerate(df):
            fig.add_trace(go.Bar(x=d.index, y=d[y], marker=dict(color=color[i]), name=name[i], hoverinfo='y'))
            # fig.add_trace(go.Bar(x=d.index, y=d[y], marker=dict(color=color[i]), domain=domain, name=name[i], hoverinfo='y'))
    else:
        fig.add_trace(go.Bar(x=df.index, y=df[y], marker=dict(color=color), name=name))
        # fig.add_trace(go.Bar(x=df.index, y=df[y], marker=dict(color=color), domain=domain, name=name))
    return fig

def plot_line(df, y, fig=None, color='#FF5949', domain=dict(x=[0,1]), name=None):
    fig = go.FigureWidget() if fig is None else fig
    fig.layout.template = 'plotly_dark'
    if type(df) == list:
        for i, d in enumerate(df):
            fig.add_trace(go.Scatter(x=d.index, y=d[y], marker=dict(color=color[i]), name=name[i], hoverinfo='y'))
            # fig.add_trace(go.Scatter(x=d.index, y=d[y], marker=dict(color=color[i]), domain=domain, name=name[i], hoverinfo='y'))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df[y], marker=dict(color=color), name=name))
        # fig.add_trace(go.Scatter(x=df.index, y=df[y], marker=dict(color=color), domain=domain, name=name))
    return fig

def plot_pie(df, fig=None, colors=None, domain=dict(x=[0,1]), name=None):
    fig = go.FigureWidget() if fig is None else fig
    fig.layout.template = 'plotly_dark'
    fig.add_trace(go.Pie(labels=df.index, values=df, hole=.4, opacity=.9, name=name,
                        marker=dict(colors=colors), domain=domain))
    return fig

