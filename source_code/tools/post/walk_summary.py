'''
总结一组实验的性能，生成summary.txt
可被post_process.py调用，总结多组
'''
import json
import numpy as np
import os
import argparse
import pandas as pd
import sys

assert os.getcwd().endswith('source_code'), '请将工作路径设为source_code，否则无法正确导入包'
sys.path.append(os.getcwd())

from tools.macro.macro import *

parser = argparse.ArgumentParser()
parser.add_argument('--group_dir', type=str)
parser.add_argument('--tag', type=str, default='train')
parser.add_argument('--gen_hyper_tune_csv', default=False, action='store_true')
parser.add_argument('--gen_five_csv', default=False, action='store_true')
args = parser.parse_args()

postfix = '/summary.txt' if args.tag == 'train' else '/eval_summary.txt'
sum_file = args.group_dir + postfix

def write_summary():
    with open(sum_file, 'w') as f:
        for root, dirs, files in os.walk(args.group_dir):
            for file in files:
                if not file == f'{args.tag}_output.txt': continue
                result_file = os.path.join(root, file)
                f.write(result_file + '\n')
                with open(result_file, 'r') as re_f:
                    text = re_f.read()
                    metrics = text[text.rindex(f'QoI:'):text.rindex('\n')]
                    f.write(metrics + '\n\n')
                print(1)



'''生成hyper tuning表格'''
def gen_hyper_tune_csv():
    def parse_multi_index(line):
        multi_index = [None, None]
        # index1
        if 'EoiCoef=0.001' in line:
            multi_index[0] = HT_INDEX1[0]
        elif 'EoiCoef=0.003' in line:
            multi_index[0] = HT_INDEX1[1]
        elif 'EoiCoef=0.03' in line:
            multi_index[0] = HT_INDEX1[3]
        else:  # 0.01
            multi_index[0] = HT_INDEX1[2]
        # index2
        if 'ShareLayer_CCobs' in line:
            multi_index[1] = HT_INDEX2[3]
        elif 'ShareLayer' in line:
            multi_index[1] = HT_INDEX2[1]
        elif 'CCobs' in line:
            multi_index[1] = HT_INDEX2[2]
        else:
            multi_index[1] = HT_INDEX2[0]
        return tuple(multi_index)

    metrics = METRICS  # 这里可以改成之前只有五个的老metric

    df = pd.DataFrame(np.zeros((len(HT_INDEX1)*len(metrics), len(HT_INDEX2))), columns=HT_INDEX2)
    with open(sum_file, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            if line == '\n': continue
            # 如下两个if-else交替执行
            if line.endswith('output.txt\n'):  # 定位行
                multi_index = parse_multi_index(line)
                start_row = len(metrics) * HT_INDEX1.index(multi_index[0])
                col = HT_INDEX2.index(multi_index[1])
            else:  # 填数
                item = []
                for metric in metrics:
                    print(metric)
                    if metric in line:
                        sub = line.index(metric) + len(metric) + 2
                        item.append(line[sub:sub+5])
                    else:
                        item.append('0.0')
                df.iloc[start_row:start_row+len(metrics), col] = item  # 4

    df.index = pd.MultiIndex.from_product([HT_INDEX1, metrics])
    df.to_csv(args.group_dir + '/hyper_tune.csv')

    # 选择表格中所有属于data ratio的单元格
    a = [i*len(metrics) for i in range(len(HT_INDEX1))]  # 对于loss ratio, 是i*len(METRICS)+1
    b = range(len(HT_INDEX2))
    data_ratio_array = df.iloc[a, b].values.astype(np.float)
    two_dimensional_spline(data_ratio_array)

'''生成five表格'''
def gen_five_csv():
    if args.group_dir.endswith('uavnum'):
        x_index = FIVE_UN_INDEX
        key = 'uav_num'
    elif args.group_dir.endswith('aoith'):
        x_index = FIVE_AT_INDEX
        key = 'aoith'
    elif args.group_dir.endswith('txth'):
        x_index = FIVE_TT_INDEX
        key = 'txth'
    elif args.group_dir.endswith('updatenum'):
        x_index = FIVE_UPN_INDEX
        key = 'update_num'
    else:
        raise NotImplementedError('未实现的五点图自变量')

    metrics = METRICS

    dfs = [pd.DataFrame(np.zeros((len(x_index), len(metrics))), columns=metrics) \
            for _ in range(len(ALGOS))]  # 每个agent一个表

    with open(sum_file, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            if line == '\n': continue
            # 如下两个if-else交替执行
            if line.endswith('output.txt\n'):  # 定位往哪填
                json_file = os.path.dirname(line) + '\\params.json'
                params = json.load(open(json_file, 'r'))
                row = x_index.index(params['input_args'][key])
                df = dfs[ALGOS.index(params['input_args']['algo'])]
            else:  # 填数
                item = dict()
                for col in metrics:
                    if col in line:
                        start = line.index(col) + len(col) + 2  # +2 是适配output.txt的格式
                        end = start + line[start:].index('.') + 4  # 每个scalar数据保留三位小数
                        item[col] = line[start:end]
                    else:
                        item[col] = '0.0'
                df.loc[row] = item

    for i, df in enumerate(dfs):
        df.index = x_index
        df.columns.name = '{}-{}'.format(ALGOS[i], key)
        df.to_csv(args.group_dir + f'/five_{ALGOS[i]}_{key}.csv')


write_summary()
if args.gen_hyper_tune_csv:
    gen_hyper_tune_csv()
if args.gen_five_csv:
    gen_five_csv()

