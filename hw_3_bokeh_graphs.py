import pandas as pd
import numpy as np
from bokeh.layouts import row, column
from bokeh.plotting import figure, show
from bokeh.models import DatetimeTickFormatter
import time

#define domain of W_0 and U_a, U_b, U_diff
W_0 = np.arange(4,100+.01, 0.1)
U_a = []
U_b = []
U_diff = []

for i in W_0:
    #Game A
    value_a = np.sqrt(i+5) + np.sqrt(i+2) - 2
    U_a.append(value_a)
    #Game B
    value_b = np.sqrt(i+10) + np.sqrt(i-3) - 2
    U_b.append(value_b)
    #Utility Diff between games
    U_diff.append(value_a - value_b)

#plot Utility A & B
p1 = figure(plot_height = 800, plot_width = 1000, 
        title = 'Expected Utility of Game A and Game B',
        x_axis_label = 'W_0 Investor Wealth', 
        y_axis_label = 'Utils')

#plot Utility A - Utility B
p2 = figure(plot_height = 800, plot_width = 1000, 
        title = 'E[Utility A] - E[Utility B]',
        x_axis_label = 'W_0 Investor Wealth', 
        y_axis_label = 'Utils')

#define x-return min and max and increment
p1.line(W_0, U_a, line_width=2, color = 'red', legend_label = 'Utility Game A')
p1.line(W_0, U_b, line_width=2, color = 'blue', legend_label = "Utility Game B")

p2.line(W_0, U_diff, line_width=2, color = 'gold')

p1.legend.location = "top_left"
p1.legend.title = "Utility Curves"

show(row(p1,p2))