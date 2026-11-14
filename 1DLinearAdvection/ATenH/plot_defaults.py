import matplotlib.pyplot as plt

def set_plot_defaults(font_size=21, label_font_size=18, line_width=2, tick_font_size=17, legend_font_size=14, grid=True):
    """
    Set default plot settings for matplotlib.

    Parameters:
    font_size (int): Default font size for text in plots.
    line_width (int): Default line width for plot lines.
    grid (bool): Whether to show grid lines by default.
    """
    plt.rc('font', size=font_size)          # controls default text sizes
    plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=label_font_size)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=tick_font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=tick_font_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=legend_font_size)    # legend fontsize
    plt.rc('figure', titlesize=font_size)   # fontsize of the figure title
    plt.rc('lines', linewidth=line_width)   # line width
    plt.rc('grid', linestyle='-')          # grid line style
    plt.rc('grid', linewidth=0.5)           # grid line width
    plt.rc('grid', color='gray')            # grid line color
    plt.rc('axes', grid=grid)               # enable/disable grid