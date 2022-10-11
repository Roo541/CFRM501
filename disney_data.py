from random import sample
import pandas as pd
import datetime as dt
import numpy as np
import scipy.special
from bokeh.layouts import row, column
from bokeh.plotting import figure, show
from bokeh.models import DatetimeTickFormatter
import math
from scipy.stats import norm 

def histogram_arithmetic(start, end, symbol, bins):
    
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

    #define x-return min and max and increment
    p2.line(df_hist['pdf_x'], norm.pdf(df_hist['pdf_x'], df['pct_change'].mean(), df['pct_change'].var()), line_width=2)

    # Show the plot
    show(row(p, p2))
    print(df_hist)
    print(df_hist['pdf_y'].sum())
    return

def histogram_log_return(start, end, symbol, bins, mu, sigma):
    
    #establish min and max pct returns, 
    min = math.floor((df['log_returns'].min())*100)/100
    max = math.ceil((df['log_returns'].max())*100)/100

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
        count = df.loc[df['log_returns'] > l]
        count = count.loc[count['log_returns'] < r]
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
            title = '{} Daily Log Returns'.format(symbol),
            x_axis_label = 'Daily Log Returns', 
            y_axis_label = 'Frequency')

    # Add a quad glyph
    p.quad(bottom=0, top=df_hist['frequency'], 
        left=df_hist['left'], right=df_hist['right'], 
        fill_color='orange', line_color='black')

    pdf_title = symbol + ' PDF' + ' N(' + str(mu) + ',' + str(sigma2) + ')'
    p2 = figure(plot_height = 800, plot_width = 1000, 
            title = pdf_title,
            x_axis_label = 'Daily Log Return Value', 
            y_axis_label = 'Probability Density')

    #define x-return min and max and increment
    p2.line(df_hist['pdf_x'], norm.pdf(df_hist['pdf_x'], mu, sigma), line_width=2)

    # Show the plot
    show(row(p, p2))
    print(df_hist)
    print(df_hist['pdf_y'].sum())
    return

def time_series(symbol):

    p = figure(plot_height = 1000, plot_width = 1500,
                title = '{} Time Series Daily Log return'.format(symbol), 
                x_axis_label = 'Date', y_axis_label = 'Log Return')

    p.vbar(df['date'],                            #categories
      top = df['log_returns'],                      #bar heights
       width = 1,
       fill_alpha = 1,
       fill_color = 'blue',
       line_alpha = .5,
       line_color='blue',
      )
    p.xaxis.formatter=DatetimeTickFormatter(
        years=["%d %m %Y"]
    )

    p.xaxis.major_label_orientation = np.pi/4
    #show(p)
    return

def sample_mean():
    summation = 0.00
    for i in range(1,len(df)):
        summation += df['log_returns'][i]
    mu = summation/len(df)
    return mu

def sample_var(mu):
    summation = 0.00
    for i in range(1,len(df)):
        summation += (df['log_returns'][i] - mu)**2
    sigma2 = summation/(len(df)-1)
    return sigma2

def sample_skewness(mu, sigma2):
    summation = 0.00
    for i in range(1,len(df)):
        summation += (df['log_returns'][i] - mu)**3
    skewness = summation/(len(df)*np.sqrt(sigma2)*sigma2)
    return skewness

def sample_kurtosis(mu, sigma2):
    summation = 0.00
    for i in range(1,len(df)):
        summation += (df['log_returns'][i] - mu)**4
    kurtosis = summation/(len(df)*sigma2**2)
    return kurtosis

#Form dataframe 
df = pd.read_csv('disney_data.csv')
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.date
df['pct_change'] = df['adj close'].pct_change()
df['log_returns'] = [np.nan]*len(df)
    
#create log returns
for i in range(1,len(df)):
    df['log_returns'][i] = np.log(df['adj close'][i]/df['adj close'][i-1])

start = ''
end = ''
symbol = 'DIS'
bins = 100
#time_series(symbol)
print('last 3 day log returns:')
print(df[-3:])
mu = sample_mean()
sigma2 = sample_var(mu)
skewness = sample_skewness(mu, sigma2)
kurtosis = sample_kurtosis(mu, sigma2)
print('sample_mean:',mu, ' sample_variance:', sigma2, ' sample_skewness:', skewness, ' sample_kurtosis:', kurtosis)
histogram_log_return(start, end, symbol, bins, mu, sigma2)