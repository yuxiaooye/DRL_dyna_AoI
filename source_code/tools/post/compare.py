import copy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
from tools.macro.macro import *
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False




def compare_plot(output_dir, xlabel, ylabel, xname, yname, x, yrange, ours, DPPO, CPPO, IC3Net, ConvLSTM, GCRL, Random):
    output_dir += f'/../pdf'
    if not os.path.exists(output_dir): os.makedirs(output_dir)


    pdf = PdfPages(output_dir + '/%s-%s.pdf' % (xname, yname))
    plt.figure(figsize=(13, 13))

    plt.xlabel(xlabel, fontsize=50)  # 42 by default
    plt.ylabel(ylabel, fontsize=50)
    plt.xticks(fontsize=50)  # 42 by default
    plt.yticks(fontsize=50)

    plt.plot(x, ours, color='red', marker='o', label='MetaCS', markersize=30, markeredgewidth=5,
             markerfacecolor='none', linewidth=4)
    plt.plot(x, DPPO, color='blue', marker='^', label=r'I2Q', markersize=30, markeredgewidth=5,
             markerfacecolor='none', linewidth=4)

    plt.plot(x, CPPO, color='turquoise', marker='s', label='HATRPO', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, IC3Net, color='seagreen', marker='v', label='MAIC', markersize=30, markeredgewidth=5,  # Shortest Path
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, ConvLSTM, color='darkorange', marker='d', label='t-LocPred', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, GCRL, color='fuchsia', marker='x', label='GCRL-min(AoI)', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, Random, color='dimgray', marker='D', label='Random', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)

    plt.xticks(x, x)

    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.ylim(yrange[0], yrange[1])


    plt.grid(True)
    plt.grid(linestyle='--')

    if yname == 'Episodic AoI':
        plt.legend(loc='upper center', fontsize=42, ncol=2, markerscale=0.9, columnspacing=0.5,
                   )  # default, columnspacing = 2.0
    else:
        plt.legend(loc='lower center', fontsize=42, ncol=2, markerscale=0.9, columnspacing=0.5,
               )  # default, columnspacing = 2.0
    plt.tight_layout()

    pdf.savefig()
    plt.close()
    pdf.close()


def get_data(x_dir):
    df = None
    for file in os.listdir(x_dir):
        if not (file.endswith('csv') and 'ALL' in file) : continue
        df = pd.read_csv(os.path.join(x_dir, file), header=None)
    return df


def compare(x_dir):
    df = get_data(x_dir)

    if x_dir.endswith('uavnum'):
        x = 'uav_num'
        ticks = FIVE_UN_INDEX
    elif x_dir.endswith('aoith'):
        x = 'aoith'
        ticks = FIVE_AT_INDEX
    elif x_dir.endswith('txth'):
        x = 'txth'
        ticks = FIVE_TT_INDEX
    elif x_dir.endswith('amount'):
        x = 'user_data_amount'
        ticks = FIVE_AM_INDEX
    elif x_dir.endswith('updatenum'):
        x = 'update_num'
        ticks = FIVE_UPN_INDEX


    for i, metric in enumerate(METRICS):
        if metric == 'energy_consuming': continue
        compare_plot(output_dir=x_dir,
                     xlabel=xlabels[x],
                     ylabel=ylabels[metric],
                     xname=xnames[x],
                     yname=ynames[metric],
                     x=ticks,
                     yrange=yranges[metric],
                     ours=df.values[:, ALGOS.index('G2ANet')*5+METRICS.index(metric)],
                     DPPO=df.values[:, ALGOS.index('DPPO')*5+METRICS.index(metric)],
                     CPPO=df.values[:, ALGOS.index('CPPO')*5+METRICS.index(metric)],
                     IC3Net=df.values[:, ALGOS.index('IC3Net')*5+METRICS.index(metric)],
                     ConvLSTM=df.values[:, ALGOS.index('ConvLSTM')*5+METRICS.index(metric)],
                     GCRL=df.values[:, ALGOS.index('GCRL')*5+METRICS.index(metric)],
                     Random=df.values[:, ALGOS.index('Random')*5+METRICS.index(metric)]
                     )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_dir', type=str)
    args = parser.parse_args()

    compare(args.x_dir)



