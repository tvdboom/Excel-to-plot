"""
========================================================>
Quick plot from excel to python
tvdboom
--------------------------------------------------------
This script loads the data from an excel or csv file and
makes a plot using Matplotlib and Seaborn.
========================================================>
"""

# ================== Import Packages ================== #

# Standard packages
import numpy as np
import pandas as pd
from optparse import OptionParser
from pathlib import Path
import sys
import warnings

# Plot packages
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.tools as tls
import plotly.offline as py

# Standard internal parameters
warnings.filterwarnings("ignore")  # Don't print out warnings
plt.rcParams["patch.force_edgecolor"] = True  # Activate histogram edgecolor
sns.set(font_scale=1.2)  # Increase font of ticks and labels slightly


# ======================= Core ======================== #

class make_plot(object):

    def __init__(self, load_file, sheet, label, save_file, plot_type, title,
                 bins, fit, rug, color, vertical, cumulative, kind, grid,
                 background, plotly):

        '''
        DESCRIPTION --------------------------------------

        Initialize variables.

        ATTRIBUTES ---------------------------------------

        load_file    --> Name of file to load
        sheet        --> Number of sheet to load
        label        --> Set plot labels
        save_file    --> Name of file to save to
        plot_type    --> Type of pot to make
        title        --> Plot title
        bins         --> Nuber of bins to use
        fit          --> Whether to plot a gaussian kernel density estimate
        rug          --> Whether to draw a rugplot on the support axis
        color        --> Color to plot everything but the fitted curve in
        vertical     --> If True, observed values are on y-axis (only hist)
        cumulative   --> If True, plots a cumualtive distribution
        kind         --> Select jointplot kind
        background   --> White or dark background (Seaborn style)
        grid         --> Wether to draw a grid (Seaborn style)
        plotly       --> Wether to draw using plotly

        '''

        # File variables
        self.load_file = load_file
        self.sheet = sheet
        self.label = label
        self.save_file = save_file
        self.plotly = plotly

        # Plot variables
        self.plot_type = plot_type
        self.title = title
        self.bins = bins
        self.fit = fit
        self.rug = rug
        self.color = color
        self.vertical = vertical
        self.cumulative = cumulative
        self.kind = kind

        # Set Seaborn style
        if grid and background:
            sns.set_style("whitegrid")
        elif grid and not background:
            sns.set_style("darkgrid")
        elif not grid and background:
            sns.set_style("white")
        else:
            sns.set_style("dark")

        # Get the data
        self.data = self.process_data(self.load())

        # Split x and y (first two columns of data)
        self.x = np.array(self.data.iloc[:, 0].values.tolist())
        if self.plot_type not in ('h', 'hist', 'histogram'):
            self.y = np.array(self.data.iloc[:, 1].values.tolist())

        # Run the class
        self.stats()
        self.draw_plot()

    def load(self):

        '''
        DESCRIPTION ------------------------------------

        Load the data into a dataframe

        '''

        def exists(filename):
            return Path(filename).is_file()

        if exists(self.load_file):
            if self.load_file.endswith('.xlsx') or \
              self.load_file.endswith('.xls'):
                return pd.read_excel(self.load_file,
                                     self.sheet,
                                     header=None)

            elif self.load_file.endswith('.csv'):
                return pd.read_csv(self.load_file, header=None)

            else:
                print("Can only read .xls, .xlsx, .csv file types!")
                sys.exit()

        elif exists(self.load_file + ".xlsx"):
            return pd.read_excel(self.load_file + ".xlsx",
                                 self.sheet, header=None)

        elif exists(self.load_file + ".xls"):
            return pd.read_excel(self.load_file + ".xls",
                                 self.sheet, header=None)

        elif exists(self.load_file + ".csv"):
            return pd.read_csv(self.load_file + ".csv",
                               header=None)

        else:
            print("File not found!")
            sys.exit()

    def process_data(self, data):

        '''
        DESCRIPTION ------------------------------------

        Initialize the class

        '''

        # Returns array of boolean values if numeric
        def isnumber(x):
            try:
                float(x)
                return True
            except ValueError:
                return False

        # Replace non-numeric values with NaN
        # Drop columns and rows with all NaN values
        data = data[data.applymap(isnumber)].dropna(axis=(0, 1), how='all')

        # Drop columns and rows with any NaN values and return
        # Cannot be done at once to not remove all rows and columns
        return data.dropna(axis=(0, 1), how='any')

    def stats(self):

        '''
        DESCRIPTION ------------------------------------

        Print out some simple stats

        '''

        print('\nStatistics\n---------------------- \
              \nData shape: {} \n\n<--- First column ---> \
              \nFirst value: {}   Last value: {} \
              \nMin-value: {}   Max-value: {} \
              \nMean: {:.2f}   Std: {:.2f}'
              .format(self.data.shape, self.x[0], self.x[-1],
                      np.min(self.x), np.max(self.x),
                      np.mean(self.x), np.std(self.x)))

        if self.plot_type in ('j', 'joint', 'jointplot'):
            print('\n\n<--- Second column ---> \
                  \nFirst value: {}   Last value: {} \
                  \nMin-value: {}   Max-value: {} \
                  \nMean: {:.2f}   Std: {:.2f}'
                  .format(self.y[0], self.y[-1],
                          np.min(self.y), np.max(self.y),
                          np.mean(self.y), np.std(self.y)))

    def histogram(self):

        '''
        DESCRIPTION ------------------------------------

        Make a histogram. Returns axes object.

        '''

        # Draw plot
        return sns.distplot(self.x,
                            bins=self.bins,
                            hist=True,
                            kde=self.fit,
                            rug=self.rug,
                            color=self.color,
                            vertical=self.vertical,
                            hist_kws=dict(cumulative=self.cumulative),
                            kde_kws=dict(cumulative=self.cumulative))

    def jointplot(self):

        '''
        DESCRIPTION ------------------------------------

        Make a jointplot. Returns jointgrid object.

        '''

        # Draw plot
        if self.kind == 'kde':
            return sns.jointplot(self.x, self.y,
                                 color=self.color,
                                 kind=self.kind)
        else:
            return sns.jointplot(self.x, self.y,
                                 color=self.color,
                                 kind=self.kind,
                                 marginal_kws=dict(bins=self.bins,
                                                   rug=self.rug))

    def draw_plot(self):

        '''
        DESCRIPTION ------------------------------------

        Draw and show the plot

        '''

        # Set labels
        titlesize = 28
        labelsize = 24
        xlabel, ylabel = "", ""
        if ',' in self.label:
            xlabel, ylabel = self.label.split(',')
        else:
            xlabel = self.label

        if self.plot_type in ('h', 'hist', 'histogram'):
            fig, ax = plt.subplots(figsize=(11, 6))
            plt.title(self.title, fontsize=titlesize)
            fig.add_subplot(self.histogram())
            if self.vertical:
                plt.xlabel(ylabel, fontsize=labelsize, labelpad=-20)
                plt.ylabel(xlabel, fontsize=labelsize)
            else:
                plt.xlabel(xlabel, fontsize=labelsize, labelpad=-20)
                plt.ylabel(ylabel, fontsize=labelsize)

            if self.plotly:
                plotly_fig = tls.mpl_to_plotly(fig)

        elif self.plot_type in ('j', 'joint', 'jointplot'):
            fig, ax = plt.subplots(figsize=(6, 6))  # Squared figure
            plt.title(self.title, fontsize=titlesize)
            jointgrid = self. jointplot()
            jointgrid.set_axis_labels(xlabel, ylabel,
                                      fontsize=labelsize,
                                      labelpad=-20)

            # Add jointgrid to mpl figure
            for i in jointgrid.fig.axes:
                fig._axstack.add(fig._make_key(i), i)

            if self.plotly:
                # Plotly operations
                plotly_fig = tls.mpl_to_plotly(fig)
                plotly_fig['layout']['yaxis2'].update({'showline': True})
                plotly_fig['layout']['xaxis3'].update({'showticklabels': False,
                                                       'showline': True})
                plotly_fig['layout']['yaxis3'].update({'showticklabels': False,
                                                       'ticks': ''})
                plotly_fig['layout']['yaxis4'].update({'showticklabels': False,
                                                       'showline': True})
                plotly_fig['layout']['xaxis4'].update({'showticklabels': False,
                                                       'ticks': ''})

        else:
            print("I do not recognize this plot type! Try hist or joint.")
            sys.exit()

        # Save and show the file (with Dash)
        if self.plotly:
            if self.save_file == "":
                py.plot(plotly_fig)
            else:
                py.plot(plotly_fig, image_filename=self.save_file, image='png')
        else:
            if self.save_file != "":
                plt.savefig(self.save_file + '.png')
            plt.show()


def new_option_parser():

    op = OptionParser()

    op.add_option("-l", "--load", default="data.xlsx", dest="load_file",
                  help="File to load (default: data.xlsx)",
                  type='str')

    op.add_option("-S", "--sheet", default=0, dest="sheet",
                  help="Number of sheet to load (default: 0)",
                  type='int')

    op.add_option("-L", "--label", default="", dest="label",
                  help="Choose label (default: None)",
                  type='str')

    op.add_option("-s", "--save", default="", dest="save_file",
                  help="Percentage of the data to use (default: 100)",
                  type='str')

    op.add_option("-t", "--type", default="hist", dest="plot_type",
                  help="Type of the plot (default: hist)",
                  type='str')

    op.add_option("-T", "--title", default="", dest="title",
                  help="Plot title (default: empty)",
                  type='str')

    op.add_option("-b", "--bins", default=None, dest="bins",
                  help="Number of bins to use (default: Freedman-Diaconis)",
                  type='int')

    op.add_option("-f", "--fit", default=False, dest="fit",
                  action="store_true",
                  help="Whether to plot a gaussian kernel density estimate \
                      (default: False)")

    op.add_option("-r", "--rug", default=False, dest="rug",
                  action="store_true",
                  help="Whether to draw a rugplot on the support axis \
                      (default: False)")

    op.add_option("-c", "--color", default="steelblue", dest="color",
                  help="Color to plot everything but the fitted curve in \
                      (default: steelblue)",
                  type='str')

    op.add_option("-v", "--vertical", default=False, dest="vertical",
                  action="store_true",
                  help="If True, observed values are on y-axis \
                      (default: False)")

    op.add_option("-C", "--cumulative", default=False, dest="cumulative",
                  action="store_true",
                  help="If True, plots a cumulative distribution \
                      (default: False)")

    op.add_option("-k", "--kind", default='scatter', dest="kind",
                  help="Select jointplot kind (default: scatter)",
                  type='str')

    op.add_option("-g", "--grid", default=True, dest="grid",
                  action="store_false",
                  help="Wether to draw a grid (default: True)")

    op.add_option("-B", "--background", default=False, dest="background",
                  action="store_true",
                  help="Boolean for white or dark background (default: dark)")

    op.add_option("-p", "--plotly", default=False, dest="plotly",
                  action="store_true",
                  help="Wether to draw using plotly (default: False)")
    return op


if __name__ == "__main__":

    options, arguments = new_option_parser().parse_args()
    make_plot(**options.__dict__)
