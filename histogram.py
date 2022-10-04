import pandas as pd
import datetime as dt
import eod_ohlc_pull as eod
import numpy as np
import scipy.special
from bokeh.layouts import row, column
from bokeh.plotting import figure, show
import math

def data_pull(start, end, symbol):
    df = eod.ohlc(start, end, symbol)

    df = pd.DataFrame.from_dict(df)
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.date
    df = df[['date', 'adjusted_close']]

    df = df.sort_values(by='date', ascending=True)
    df = df.reset_index(drop=True)
    df['pct_change'] = [np.nan]*len(df)
    df['pct_change'] = df['adjusted_close'].pct_change()

    return df

def histogram(start, end, symbol, bins):
    #call get data from eod
    df = data_pull(start, end, symbol)
    #establish min and max pct returns, 
    min = math.floor((df['pct_change'].min())*100)/100
    max = math.ceil((df['pct_change'].max())*100)/100

    #create incrementation of pct_returns from min to max establish the left and right for histogram
    increments = np.arange(min, max+.01 ,abs(min-max)/bins)
    left = increments[0:-1]
    right = increments[1:]
    frequency = [np.nan]*len(left)
    df_hist = {'frequency':frequency, 'left':left, 'right':right}
    df_hist = pd.DataFrame.from_dict(df_hist)

    #count the frequency for histogram bin range
    for i in range(len(df_hist)):
        l = df_hist['left'][i]
        r = df_hist['right'][i]
        count = df.loc[df['pct_change'] > l]
        count = count.loc[count['pct_change'] < r]
        df_hist['frequency'][i] = len(count)

    df_hist['pdf_y'] = [np.nan]*len(df_hist)
    df_hist['pdf_x'] = [np.nan]*len(df_hist)
    total_count = df_hist['frequency'].sum()
    for i in range(len(df_hist)):
        l = df_hist['left'][i]
        r = df_hist['right'][i]
        value = (l + r)/2.0
        pdf_value = df_hist['frequency'][i]/total_count
        df_hist['pdf_y'][i] = pdf_value
        df_hist['pdf_x'][i] = value

    #plot 
    p = figure(plot_height = 800, plot_width = 1000, 
            title = '{} Daily Pct Returns'.format(symbol),
            x_axis_label = 'Daily Return', 
            y_axis_label = 'Frequency')

    # Add a quad glyph
    p.quad(bottom=0, top=df_hist['frequency'], 
        left=df_hist['left'], right=df_hist['right'], 
        fill_color='orange', line_color='black')

    p2 = figure(plot_height = 800, plot_width = 1000, 
            title = '{} PDF'.format(symbol),
            x_axis_label = 'Daily Return Value', 
            y_axis_label = 'Probability')

    p2.line(df_hist['pdf_x'], df_hist['pdf_y'], line_width=2)

    # Show the plot
    show(row(p, p2))
    print(df_hist)
    print(df_hist['pdf_y'].sum())

start = '2017-10-2'
end = '2022-10-2'
symbol = 'MSFT'
bins = 100
histogram(start, end, symbol, bins)