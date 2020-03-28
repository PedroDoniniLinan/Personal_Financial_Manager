import pandas as pd
from plotly import tools
import plotly.graph_objs as go

lpalette = {'g':'#13E881', 'r':'#FF5949', 'o':'#FFB84C', 'b':'#A7BEFA', 'w':'#FFFFFF', 'pk':'#FFA1FD', 'bg':'#FFEDBA', 'c':'#90EBF0',
            'pp':'#BBA0F0', 'g2':'#CCFFED', 'b2':'#E3EEFF', 'y':'#FFFC57'}
dpalette = {'g':'#268040', 'r':'#800B00', 'o':'#A13808', 'b':'#464CC2', 'w':'#B8BFBA', 'pk':'#A04EA6', 'bg':'#807155', 'c':'#1D4544',
            'pp':'#291147', 'g2':'#394742', 'b2':'#414449', 'y':'#666523'}

def make_fig(rows=1, cols=1, specs=[[{}]]):
    fig = go.FigureWidget(tools.make_subplots(rows=rows, cols=cols, specs=specs))
    fig.layout.template = 'plotly_dark'
    return fig

def plot(df, x, fig, palette, y=None, plot_type='line', row=1, col=1, name=None):
    df = df.set_index(x)
    y = df.columns if y is None else y
    name = y if name is None else name
    for i, c in enumerate(y):
        if plot_type == 'area':
            fig.add_trace(go.Scatter(x=df.index, y=df[c], name=name[i], marker=dict(color=palette[c]), 
                hoverinfo='name+y', stackgroup='one'), row=row, col=col)
        elif plot_type == 'bar':
            fig.add_trace(go.Bar(x=df.index, y=df[c], name=name[i], marker=dict(color=palette[c]), 
                hoverinfo='name+y'), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], name=name[i], marker=dict(color=palette[c]), 
                hoverinfo='name+y'), row=row, col=col)
    return fig

def plot_pie(df, fig, palette, domain=dict(x=[0,1]), name=None):
    df = df.sort_values(ascending=False)
    colors = [palette[c] for c in df.index]
    fig.add_trace(go.Pie(labels=df.index, values=df, hole=.4, opacity=.9, name=name,
                        marker=dict(colors=colors), domain=domain))
    return fig

def change_df_prop(df, font=22, align='center'):
    heading_properties = [('font-size', str(font-2) + 'px')]
    cell_properties = [('font-size', str(font) + 'px'), ('text-align', align)]

    dfstyle = [dict(selector="th", props=heading_properties),
     dict(selector="td", props=cell_properties)]
    return df.style.set_table_styles(dfstyle).hide_index()

