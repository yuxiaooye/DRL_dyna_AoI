import copy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
from tools.macro.macro import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def error(input_list):
    input = np.array(input_list)
    input = input.transpose((1, 0))
    error_low = input[0] - input[1]
    error_high = input[2] - input[0]
    error = []
    error.append(error_low)
    error.append(error_high)
    return error


def average(input_list):
    input = np.array(input_list)
    input = input.transpose((1, 0))
    return input[0]


def compare_plot_errorbar(xlabel, ylabel, x, eDivert, woApeX, woRNN, MADDPG):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.errorbar(x=x, y=average(eDivert), yerr=error(eDivert), fmt='r-o', label='e-Divert', capsize=4)
    plt.errorbar(x=x, y=average(woApeX), yerr=error(woApeX), fmt='g-^', label='e-Divert w/o Ape-X', capsize=4)
    plt.errorbar(x=x, y=average(woRNN), yerr=error(woRNN), fmt='m-<', label='e-Divert w/o RNN', capsize=4)
    plt.errorbar(x=x, y=average(MADDPG), yerr=error(MADDPG), fmt='k-*', label='MADDPG', capsize=4)

    plt.ylim(ymin=0, ymax=1)
    plt.grid(True)
    plt.grid(linestyle='--')
    plt.legend()
    plt.show()


def compare_plot(output_dir, xlabel, ylabel, x, yrange, ours, DPPO, CPPO, IC3Net, ConvLSTM, GCRL, Random):
    output_dir += f'/pdf'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    x_label_saved = xlabel

    pdf = PdfPages(output_dir + '/%s-%s.pdf' % (x_label_saved, ylabel))
    plt.figure(figsize=(13, 13))

    plt.xlabel(xlabel, fontsize=42)  # 32 by default
    plt.ylabel(ylabel, fontsize=42)
    plt.xticks(fontsize=42)  # 32 by default
    plt.yticks(fontsize=42)

    plt.plot(x, ours, color='red', marker='o', label='DRL-PCN', markersize=30, markeredgewidth=5,
             markerfacecolor='none', linewidth=4)
    plt.plot(x, DPPO, color='k', marker='^', label=r'I2Q', markersize=30, markeredgewidth=5,
             markerfacecolor='none', linewidth=4)

    plt.plot(x, CPPO, color='orange', marker='s', label='HATRPO', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, IC3Net, color='purple', marker='v', label='MAIC', markersize=30, markeredgewidth=5,  # Shortest Path
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, ConvLSTM, color='b', marker='d', label='t-LocPred', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, GCRL, color='green', marker='x', label='GCRL-min(AoI)', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, Random, color='deepskyblue', marker='D', label='Random', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)

    plt.xticks(x, x)

    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.ylim(yrange[0], yrange[1])


    plt.grid(True)
    plt.grid(linestyle='--')

    if metric == 'episodic_aoi':
        plt.legend(loc='upper center', fontsize=36, ncol=2, markerscale=0.9, columnspacing=0.5,
                   )  # default, columnspacing = 2.0
    else:
        plt.legend(loc='lower center', fontsize=36, ncol=2, markerscale=0.9, columnspacing=0.5,
               )  # default, columnspacing = 2.0
    plt.tight_layout()

    pdf.savefig()
    plt.close()
    pdf.close()


def manually_data(x_dir):
    dfs = dict()

    for file in os.listdir(x_dir):
        if not file.endswith('csv'): continue
        algo = file.split('_')[1]
        df = pd.read_csv(os.path.join(x_dir, file), index_col=0)
        dfs[algo] = df

    return dfs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_dir', type=str)

    args = parser.parse_args()

    dfs = manually_data(args.x_dir)



    if args.x_dir.endswith('uavnum'):
        x = 'uav_num'
        ticks = FIVE_UN_INDEX
    elif args.x_dir.endswith('aoith'):
        x = 'aoith'
        ticks = FIVE_AT_INDEX
    elif args.x_dir.endswith('txth'):
        x = 'txth'
        ticks = FIVE_TT_INDEX
    elif args.x_dir.endswith('updatenum'):
        x = 'update_num'
        ticks = FIVE_UPN_INDEX


    for i, metric in enumerate(METRICS):
        if metric == 'energy_consuming': continue
        compare_plot(output_dir=args.x_dir,
                     xlabel=xlabels[x],
                     ylabel=metric,  # TODO 最终论文不能直接用metric
                     x=ticks,
                     yrange=yranges[metric],
                     ours=dfs['G2ANet'][metric],
                     DPPO=dfs['DPPO'][metric],
                     CPPO=dfs['CPPO'][metric],
                     IC3Net=dfs['IC3Net'][metric],
                     ConvLSTM=dfs['ConvLSTM'][metric],
                     GCRL=dfs['GCRL'][metric],
                     Random=dfs['Random'][metric]
                     )



