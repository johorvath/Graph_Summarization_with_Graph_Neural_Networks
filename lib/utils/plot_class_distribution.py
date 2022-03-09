"""
plot_class_distribution.py: plotting function for element dictionary (key=element, value=count of element)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import operator
import matplotlib.ticker as mtick
import numpy as np

def plot_class_distribution( di, filename="class_distribution.png", max_classes=1000 ):
    """
    Calculates the gini_index over given probabilities

    
    Args:
        di (dict): dictionary of elements with count of elements
        filename (string): plotfile [.png|.pgf]
        max_classes (int): maximum number of plotted classes
    """
    #sort dictionary by inverse count
    sorted_x = sorted(di.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_x)    
    hi = list(map(operator.itemgetter(0), sorted_x))
    c = list(map(operator.itemgetter(1), sorted_x))
    
    #calculate probability based on the occurence
    c_sum = sum(c)
    c = [ x / c_sum for x in c ]
    
    #reduce number of elements based on max_classes
    if(max_classes > 0):
        step = int(len(hi) / max_classes)
        print(len(hi))
        print(max_classes)
        print(step)
        c = c[::step]
        hi = hi[::step]
    
    # 1920, 1080 -> Inches -> 25.6, 14.4
    fig, ax = plt.subplots(1, 1, figsize=(25.6, 14.4))
    
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True)
 
    #creating the bar plot
    print(c)
    hi = list(range(len(c)))
    #create x axis labels and ticks
    perc = np.linspace(0,100,len(c))
    print(perc)
    ax.bar(perc, c, color ='maroon' )
    fig.gca().set_yscale('log')
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
     
    ax.set_xlabel("Classes")
    ax.set_ylabel("Class likelihood")
    
    if( "pgf" in filename ):
        # config plot
        mpl.use("pgf")
        mpl.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
        fig.set_size_inches(4.30, 3.50)
        print(filename)
        fig.savefig(filename)
    else:
        print(filename)
        fig.savefig(filename, dpi=75)
