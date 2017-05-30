import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


plt.style.use('ggplot')
mpl.rcParams['text.color'] = 'black'
mpl.rc('grid', linewidth=0.5, alpha=0.8, linestyle='-', color='#D3D3D3')
mpl.rc('axes', facecolor='white')
mpl.rc('boxplot', meanline=True, showmeans=True)
mpl.rc('boxplot.meanprops', linestyle='-', linewidth=1.0)
mpl.rc('boxplot.medianprops', linestyle='-', linewidth=1.0)


class Viz:
    @staticmethod
    def _save(fstr, **kwargs):
        plt.savefig(fstr, **kwargs)
        plt.close()

    @staticmethod
    def _format_legend(ax=None):
        if not ax:
            ax = plt.gca()

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    @staticmethod
    def _format_axes_label(ax=None, xlabel=None, ylabel=None):
        if not ax:
            ax = plt.gca()
        ax.set_xlabel(xlabel if xlabel else '')
        ax.set_ylabel(ylabel if ylabel else '')

    @staticmethod
    def _format_axes_ticks(xticks=None, yticks=None, xlabels=None, ylabels=None, x_rotation=0, y_rotation=0):
        plt.xticks(xticks, xlabels, rotation=x_rotation)
        plt.yticks(yticks, ylabels, rotation=y_rotation)

    @staticmethod
    def _format(ax=None, title=None, legend=None,
                xlabel=None, xticks=None, xlabels=None, xpad=None, xlim=None, x_rotation=0,
                ylabel=None, yticks=None, ylabels=None, ypad=None, ylim=None, y_rotation=0):

        if not ax:
            ax = plt.gca()

        if xticks is None:
            xticks = ax.get_xticks()
        if yticks is None:
            yticks = ax.get_yticks()

        if xlabels is None:
            xlabels = xticks
        if ylabels is None:
            ylabels = yticks

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        if xpad:
            ax.tick_params(axis='x', which='major', pad=xpad)
        if ypad:
            ax.tick_params(axis='y', which='major', pad=ypad)

        if legend:
            Viz._format_legend(ax)

        Viz._format_axes_label(ax, xlabel, ylabel)
        Viz._format_axes_ticks(xticks, yticks, xlabels, ylabels, x_rotation, y_rotation)
        plt.title(title if title else '')

    @staticmethod
    def line(results, fstr, columns=None, index=None, **kwargs):
        df = pd.DataFrame(results, columns=columns, index=index)
        df.plot()
        Viz._format(**kwargs)
        Viz._save(fstr, bbox_inches='tight')

    @staticmethod
    def box(results, fstr, columns=None, index=None, **kwargs):
        df = pd.DataFrame(results, columns=columns, index=index)
        df.plot.box()
        Viz._format(**kwargs)
        Viz._save(fstr, bbox_inches='tight')

    @staticmethod
    def bar(results, fstr, columns=None, index=None, stacked=False, legend=True, xerr=None, yerr=None,
            colormap=None, color=None, horizontal=False, **kwargs):
        kwds = {}
        if colormap and color:
            kwds['color'] = color
        elif colormap:
            kwds['colormap'] = colormap
        elif color:
            kwds['color'] = color

        df = pd.DataFrame(results, columns=columns, index=index)
        if horizontal:
            df.plot.barh(stacked=stacked, legend=legend, xerr=xerr, yerr=yerr, **kwds)
        else:
            df.plot.bar(stacked=stacked, legend=legend, xerr=xerr, yerr=yerr, **kwds)
        Viz._format(legend=legend, **kwargs)
        Viz._save(fstr, bbox_inches='tight')

    @staticmethod
    def hist(results, fstr, columns=None, index=None, **kwargs):
        df = pd.DataFrame(results, columns=columns, index=index)
        df.plot.hist(colormap='Paired')
        Viz._format(**kwargs)
        Viz._save(fstr, bbox_inches='tight')

    @staticmethod
    def table(header, data, fstr):
        with open('{}.csv'.format(fstr), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

    @staticmethod
    def error_bar(x, y, fstr, xerr=None, yerr=None, ecolor='r', marker='^', **kwargs):
        plt.errorbar(x, y, xerr=xerr, yerr=yerr, marker=marker, ecolor=ecolor)
        Viz._format(**kwargs)
        Viz._save(fstr, bbox_inches='tight')


class Visualize(Viz):
    def plot_intervals(self, actuals, predicted, data, rstr, **kwargs):
        x = np.arange(0, actuals.size)

        for n, ((low, high), c, nc) in enumerate(reversed(data)):
            yerr = actuals - low, high - actuals
            plt.errorbar(x, actuals, yerr=yerr, ecolor=c, color=c, linestyle='None', zorder=n, capthick=2,
                         label='NC {}'.format(nc))

        for i, a in enumerate(actuals):
            plotted = False
            for (low, high), c, _ in data:
                if low[i] <= a <= high[i]:
                    plotted = True
                    plt.plot([i], [a], marker='o', color=c)
                    plt.plot([i], [predicted[i]], marker='x', color='black')
                    break

            if not plotted:
                plt.plot([i], [a], marker='o', color='red')

        ax = plt.gca()
        ax.grid(False)
        fig = plt.gcf()
        fig.set_size_inches(14, 8)

        self._format(**kwargs)
        self._save(rstr, bbox_inches='tight')

