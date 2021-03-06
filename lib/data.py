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

"""
# ================================================= FINANCE =================================================== #

# READ/PREPARE DATA #

def read_invest():
    # bonds values log
    dinv = pd.read_csv('br_invest.csv')
    dinv['Data'] = dinv['Data'].apply(lambda x : dt.datetime.strptime(str(x),'%d/%m/%y').date())

    # portfolio acquisition log
    dport = pd.read_csv('br_portfolio.csv')
    dport['Date'] = dport['Date'].apply(lambda x : dt.datetime.strptime(str(x),'%d/%m/%y').date())
    
    # stocks prices
    dprices = pd.read_csv('br_prices.csv')
    dinv_port = generate_invest_table(dport, dprices)
    return dinv, dport, dinv_port, dprices

def read_finance():
    
    # read investment data
    dinv, dport, di, dprices = read_invest()

    # conatenate bonds and stocks
    dinv = pd.concat([dinv, di]).reset_index(drop=True)

    # calculate investment flows
    dflow = calc_flow_gains(dinv)
    dflow['Conta'] = dflow['Tipo']
    dflow['Conta'] = dflow['Conta'].apply(lambda x : x if x == 'Nu' else 'Easy')
    dflow.pop('Tipo')
    dflow['Area'] = 'Investimento'
    # dflow['Total'] = dflow['Valor']
    dflow['Valor'] = dflow['ValorDelta']

    # income
    dg = pd.read_csv('br_ganhos.csv')
    dg['Data'] = dg['Data'].apply(lambda x : dt.datetime.strptime(str(x),'%d/%m/%y').date())
    dg = pd.concat([dg, dflow], sort=False).reset_index(drop=True)

    # expenses
    df = pd.read_csv('br_gastos.csv')
    df['Data'] = df['Data'].apply(lambda x : dt.datetime.strptime(str(x),'%d/%m/%y').date())

    # tranferences
    dtr = pd.read_csv('br_transf.csv')

    return dg, df, dtr

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

def join_prices(row, dprices):
    di = dprices[dprices['Ticker'] == row['Ticker']]
    for c in di.columns:
        if c == 'Ticker':
            continue
        date = dt.datetime.strptime(c, '%y-%m-%d').date()
        if date < row['Date']:
            row[c] = 0
        else:
            row[c] = di[c].item()
    return row

# FUNDS #

def calc_account_funds(dincome, dexpense, dtransfer, verif_vals):
    # summarize values by account
    dtout = pivot_agg(dtransfer, val='Valor', idx=['De'], agg={'Valor':np.sum}, date=None).set_index('De')    
    dtout.columns = ['Transferido']
    dtin = pivot_agg(dtransfer, val='Valor', idx=['Para'], agg={'Valor':np.sum}, date=None).set_index('Para')
    dtin.columns = ['Recebido']
    dinc = pivot_agg(dincome, val='Valor', idx=['Conta'], agg={'Valor':np.sum}, date=None).set_index('Conta')
    dinc.columns = ['Ganho']
    dexp = pivot_agg(dexpense, val='Valor', idx=['Pagamento'], agg={'Valor':np.sum}, date=None).set_index('Pagamento')
    dexp.columns = ['Gasto']

    
    # join and compute accounts funds
    djoin = dinc.join(dexp).join(dtin).join(dtout).fillna(0)
    djoin['Saldo'] = djoin['Ganho'] + djoin['Recebido'] - djoin['Transferido'] - djoin['Gasto']
    djoin['Lido'] = verif_vals
    djoin['Erro'] = djoin['Saldo'] - djoin['Lido']
    djoin.index.name = ''
    return djoin.loc[:, ['Saldo', 'Lido', 'Erro']]

def calc_fund(balance, income='Renda', expense='Despesa real'):
    fund = pd.DataFrame()
    fund['Data'] = balance['Data']
    fund['Saldo'] = (balance[income] - balance[expense]).cumsum()
    fund['Var'] = balance[income] - balance[expense]
    fund['Crescimento'] = fund['Saldo'].rolling(2).apply(lambda x : 100*(x[1] - x[0])/x[0], raw=True)
    fund['%'] = fund['Saldo'].apply(lambda x : 100*(x - fund.head(1)['Saldo']) / fund.head(1)['Saldo'])
    return fund.fillna(0)

# FLOWS #

def calc_flows(df):
    # read values
    open_val = df.head(1)['Valor'].item()
    final_val = df.tail(1)['Valor'].item()
    open_date = df.head(1)['Data'].item()
    final_date = df.tail(1)['Data'].item()
    status = df.tail(1)['Status'].item()
    itype = df.tail(1)['Tipo'].item()

    # compute metrics
    active = False if status == 'Fechado' else True
    gain = final_val - open_val
    duration = (final_date - open_date).days
    total_yield = gain / open_val
    daily_yield = (1+total_yield)**(1/duration)-1 if duration > 0 else 0
    monthly_yield = (1+daily_yield)**30.5-1 
    annual_yield = (1+daily_yield)**365-1 
    year_profit = final_val * annual_yield
    
    # structure dataframe
    dr = pd.DataFrame({'Valor inicial': [open_val], 
                       'Ganho': [gain], 
                       'Meses': [duration / 30.5], 
                       'Rendimento': [int2pct(total_yield)],  
                       'Rendimento mensal': [int2pct(monthly_yield)], 
                       'Rendimento anual': [int2pct(annual_yield)],
                       'Ganho em 1 ano': [year_profit],
                       'Ativo':[active], 
                       'Tipo': [itype]})
    return dr

def calc_flows_overview(df):
    gain = df['Ganho'].sum()
    value = df[df['Ativo']]['Valor inicial'].sum() + df[df['Ativo']]['Ganho'].sum()
    total_yield = gain / df['Valor inicial'].sum()
    year_profit = df[df['Ativo']]['Ganho em 1 ano'].sum()
    annual_yield = year_profit / value
    doverview = pd.DataFrame([[value, gain, int2pct(total_yield), year_profit, int2pct(annual_yield)]], 
                    columns=['Valor base', 'Ganho atual', 'Rendimento atual', 'Ganho em 1 ano', 'Rendimento anual'])
    return doverview

def calc_flow_gains(df):
    dflow = df.groupby('ID').apply(lambda x : window(x, val='Valor', suffix='Delta', window=2, func=delta, fill=0))
    dflow.pop('ID')
    dflow = dflow.reset_index(drop=True)
    dflow = dflow[dflow['Status'] != 'Aberto']
    dflow.pop('Status')
    return dflow    

# STOCKS #

def calc_stocks_feat(dreturn, dstocks, dquote, assets, industry):
    # join dfs  
    dfull = dreturn.join(dstocks.set_index('Ticker'), on='Ticker', how='inner')
    dfull['Price'] = dquote[str(dt.date.today().year)]

    # distribution
    dfull['Geo'] = dfull[assets].idxmax(axis=1)
    dfull['Domain'] = dfull[industry].idxmax(axis=1)
    dfull['Asset'] = dfull[assets].idxmax(axis=1).apply(lambda x : x if (x == 'Bonds' or x == 'Commod' or x == 'Cash') else 'Equity')

    # risk and return
    dfull['Risk'] = dfull['Volatility'].apply(lambda x : int2pct(x/100))
    dfull['Tax'] = dfull['TER'].apply(lambda x : int2pct(x/100))
    dfull['Return'] = dfull['Predicted'].apply(int2pct)
    dfull['YTD'] = dfull[str(dt.date.today().year)].apply(int2pct)

    # score
    weights = [0.05, 0.5, 0.35, 0.1]
    dfull['Score'] = (10+(30-dfull['Price'])/27)*weights[0] + dfull['Predicted']/0.3*10*weights[1] + (10+(7-dfull['Volatility'])/2.3)*weights[2] + (10-dfull['TER']/0.2)*weights[3]

    return dfull

# PORTFOLIO CONSTRUCTION #

def calc_portfolio(dport_full):
    dport_full['Profit'] = (dport_full['Price'] - dport_full['Buy']) * dport_full['Shares']
    dport_full['Value'] = (dport_full['Price'] * dport_full['Shares'])

    dport_full['Yield'] = (dport_full['Profit'] / (dport_full['Buy'] * dport_full['Shares']))
    dport_full['Annual yield'] = (((1 + dport_full['Yield']) ** (1/(dt.date.today() - dport_full['Date']).apply(lambda x : x.days))) ** 365 - 1)
    dport_full['Year profit'] = dport_full['Price'] * dport_full['Shares'] * dport_full['Annual yield']

    dport_full['Keep'] = dport_full['TER'] * dport_full['Price'] * dport_full['Shares'] / 100
    dport_full['Sell Tax'] = (dport_full['Profit'] - (dport_full['Buy Tax'] + dport_full['Keep'])) * 0.15
    dport_full['Total Tax'] = dport_full['Buy Tax'] + dport_full['Keep'] + dport_full['Sell Tax']
    dport_full['Tax'] = (dport_full['Total Tax'] / dport_full['Profit']).apply(lambda x : 0 if pd.isnull(x) or x == np.inf else x).apply(int2pct)

    dport_full['Yield (Liq.)'] = ((dport_full['Profit'] - dport_full['Total Tax']) / (dport_full['Buy'] * dport_full['Shares']))
    dport_full['Annual yield (Liq.)'] = (((1 + dport_full['Yield (Liq.)']) ** (1/(dt.date.today() - dport_full['Date']).apply(lambda x : x.days))) ** 365 - 1)
    dport_full['Year profit (Liq.)'] = dport_full['Price'] * dport_full['Shares'] * dport_full['Annual yield (Liq.)']

    dport_full['Yield %'] = dport_full['Yield'].apply(int2pct)
    dport_full['Annual yield %'] = dport_full['Annual yield'].apply(int2pct)
    dport_full['Yield (Liq.) %'] = dport_full['Yield (Liq.)'].apply(int2pct)
    dport_full['Annual yield (Liq.) %'] = dport_full['Annual yield (Liq.)'].apply(int2pct)
    
    return dport_full

def add_bonds_to_port(dport_full, dflow, assets):
    profit = dflow['Ganho'].sum()
    value = dflow[dflow['Ativo']]['Valor inicial'].sum() + dflow[dflow['Ativo']]['Ganho'].sum()
    yields = profit / dflow['Valor inicial'].sum()
    di = pd.DataFrame([['NU', value, profit, yields, int2pct(yields), 0, 0, 0, 0, 100, 0, 0]], columns=['Ticker', 'Value', 'Profit', 'Yield', 'Yield %'] + assets)
    dport_bonds = pd.concat([dport_full, di], sort=False).reset_index(drop=True)
    return dport_bonds

def calc_distribution(dport, assets, industry):
    for c in assets:
        dport[c] = dport[c] * dport['Value'] / dport['Value'].sum()
    for c in industry:
        dport[c] = dport[c] * dport['Value'] / dport['Value'].sum()
    return dport

# PORTFOLIO EVOLUTION #

def calc_initial_value(row, date, col):
    if row['Date'].month < date.month:
        row['Initial'] = row[col] * row['Shares']
        return row['Initial']
    else:
        row['Initial'] = row['Buy'] * row['Shares']
        return row['Initial']

def calc_port_months(dport, dfeat, dprices, assets, industry):
    select = ['Ticker', 'Initial', 'Month initial', 'Date', 'Value', 'Profit', 'Total profit'] + assets + industry
    dports = pd.DataFrame(columns=select)
    cols = dprices.columns
    for i in range(cols.shape[0]):
        if i < 2:
            continue
        date = dt.datetime.strptime(cols[i], '%y-%m-%d').date()
        dport_full = dport.join(dfeat[['Ticker', 'Price', 'TER'] + assets + industry].set_index('Ticker'), on='Ticker')
        dport_full = dport_full.join(dprices.set_index('Ticker'), on='Ticker')
        dport_full = dport_full[dport_full['Date'] <= date]
        dport_full['Price'] = dport_full[cols[i]]
        dport_full['Value'] = dport_full['Price'] * dport_full['Shares']
        dport_full['Initial'] = dport_full['Buy'] * dport_full['Shares']
        dport_full['Month initial'] = dport_full.apply(lambda x : calc_initial_value(x, date, cols[i-1]), axis=1)
        dport_full['Profit'] = dport_full['Value'] - dport_full['Month initial']
        dport_full['Total profit'] = (dport_full['Price']-dport_full['Buy']) * dport_full['Shares']
        dport_full['Date'] = date
        dports = pd.concat([dports, dport_full[select]]).reset_index(drop=True)
    return dports

def calc_bonds_months(dprices, dbond, assets, industry):
    dbonds = pd.DataFrame(columns=['Date','Value'])

    for c in dprices.columns:
        if c == 'Ticker':
            continue
        date = dt.datetime.strptime(c, '%y-%m-%d').date()        
        dflow = dbond[dbond['Data'] <= date].groupby('ID').apply(calc_flows)
        dflow = dflow.reset_index(level=[0, 1])
        dflow.pop('level_1')
        value = (dflow[dflow['Ativo']]['Valor inicial'] + dflow[dflow['Ativo']]['Ganho']).sum()
        dbonds = dbonds.append(pd.DataFrame([[date, value]], columns=['Date', 'Value']))
    dbonds = dbonds.reset_index(drop=True)

    dnu_profit = pivot_agg(calc_flow_gains(dbond), val='Valor', idx=['Data'], col=['Tipo'], agg={'Valor':np.sum})
    dnu_profit = dnu_profit.reset_index(drop=True)
    dbonds['Profit'] = dnu_profit['Nu']
    dbonds['Ticker'] = 'NU'
    dbonds['Month initial'] = (dbonds['Value'] - dbonds['Profit'])
    dbonds['Total profit'] = dbonds['Profit'].cumsum()
    
    for a in assets + industry:
        dbonds[a] = 100 if a == 'Bonds' else 0

    return dbonds

# PORTFOLIO ANALYSIS #

def analyze_port(dport, dfeat, dprices, assets, industry):
    select = ['Ticker', 'Initial', 'Month initial', 'Date', 'Value', 'Profit', 'Total profit'] + assets + industry
    dports = pd.DataFrame(columns=select)
    cols = dprices.columns
    for i in range(cols.shape[0]):
        if i < 2:
            continue
        date = dt.datetime.strptime(cols[i], '%y-%m-%d').date()
        dport_full = dport.join(dfeat[['Ticker', 'Price', 'TER'] + assets + industry].set_index('Ticker'), on='Ticker')
        dport_full = dport_full.join(dprices.set_index('Ticker'), on='Ticker')
        dport_full = dport_full[dport_full['Date'] <= date]
        dport_full['Price'] = dport_full[cols[i]]
        dport_full['Value'] = dport_full['Price'] * dport_full['Shares']
        dport_full['Initial'] = dport_full['Buy'] * dport_full['Shares']
        dport_full['Month initial'] = dport_full.apply(lambda x : calc_initial_value(x, date, cols[i-1]), axis=1)
        dport_full['Profit'] = dport_full['Value'] - dport_full['Month initial']
        dport_full['Total profit'] = (dport_full['Price']-dport_full['Buy']) * dport_full['Shares']
        dport_full['Date'] = date
        dports = pd.concat([dports, dport_full[select]]).reset_index(drop=True)
"""