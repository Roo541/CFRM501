from random import sample
from re import X
import pandas as pd
import datetime as dt
import numpy as np
import scipy.special
from bokeh.layouts import row, column, gridplot
from bokeh.plotting import figure, show
from bokeh.models import DatetimeTickFormatter
import math
from scipy.stats import norm 
from bokeh.models import LinearAxis, Range1d

def gaussian(x_min, x_max, mu, sigma2):
    x_array = np.arange(x_min, x_max + 0.001, 0.001)
    y_array = []
    for i in x_array:
        value = 1/(np.sqrt(2*np.pi*sigma2))*np.exp(-1/2*(i-mu)**2/sigma2)
        y_array.append(value)

    return x_array, y_array

def laplace(x_min, x_max, a, mu, sigma):
    x_array_l = np.arange(x_min, x_max + 0.001, 0.001)
    y_array_l = []
    gamma = np.sqrt(2*sigma**2 + mu**2)

    for i in x_array_l:
        if i >= a:
            value = (1/gamma)*(np.exp(((i-a)/sigma**2)*(mu-gamma)))
            y_array_l.append(value)
        if i < a:
            value = (1/gamma)*(np.exp(((i-a)/sigma**2)*(mu+gamma)))
            y_array_l.append(value)
    return x_array_l, y_array_l


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

def histogram_pdf_normal_laplace(start, end, symbol, bins, mu, sigma2):
    
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
            y_axis_label = 'Frequency', y_range = (0,df_hist['frequency'].max()+5))

    # Add a quad glyph
    p.quad(bottom=0, top=df_hist['frequency'], 
        left=df_hist['left'], right=df_hist['right'], 
        fill_color='yellow', line_color='black')

    pdf_title = symbol + ' PDF' + ' N(' + str(mu) + ',' + str(sigma2) + ')'
    p2 = figure(plot_height = 800, plot_width = 1000, 
            title = pdf_title,
            x_axis_label = 'Daily Log Return', 
            y_axis_label = 'Probability Density')

    #calculated by hand
    a_hat = -0.00255304
    mu_hat = 0.002177315
    sigma_hat = 0.0217198

    #figure with new mu_hat
    pdf_title = symbol + ' Laplace PDF' + ' mu_hat = ' + str(mu) + ', a_hat = ' + str(a_hat) + ', (sigma_hat)^2 = ' + str(sigma_hat) + ')'
    p3 = figure(plot_height = 800, plot_width = 1000, 
            title = pdf_title,
            x_axis_label = 'Daily Log Return', 
            y_axis_label = 'Probability Density')
    #add new axis
    x_array, y_array = gaussian(min, max, mu, sigma2)
    x_array_l, y_array_l = laplace(min, max, a_hat, mu_hat, sigma_hat)
    p.extra_y_ranges = {"PDF": Range1d(start=0, end=32)}
    p.line(x_array, y_array, color="blue", y_range_name="PDF", line_width = 3)
    p.line(x_array_l, y_array_l, color="red", y_range_name="PDF", line_width = 3)
    p.add_layout(LinearAxis(y_range_name="PDF"), 'right')

    p2.line(x_array, y_array, color = 'blue', line_width=2)
    p3.line(x_array_l, y_array_l, color = 'red', line_width=2)
    # Show the plot
    show(gridplot([[p, p2], [p3, None]]))
    print(df_hist)
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
    show(p)
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

print('***',len(df))
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
print(' sample_mean:',mu) 
print(' sample_variance:', sigma2) 
print(' sample_skewness:', skewness) 
print(' sample_kurtosis:', kurtosis)
histogram_pdf_normal_laplace(start, end, symbol, bins, mu, sigma2)