
import json
import numpy as np
import os
import argparse
import pandas as pd
import sys

assert os.getcwd().endswith('source_code'), '请将工作路径设为source_code，否则无法正确导入包'
sys.path.append(os.getcwd())

from tools.macro.macro import *



def write_summary(x_dir, sum_file):
    with open(sum_file, 'w') as f:
        for root, dirs, files in os.walk(x_dir):
            for file in files:
                if not file == 'train_output.txt': continue
                result_file = os.path.join(root, file)
                f.write(result_file + '\n')
                with open(result_file, 'r') as re_f:
                    text = re_f.readlines()
                    case1 = 'ConvLSTM' in result_file
                    case2 = 'DPPO' in result_file
                    if case1 or case2:
                        trunc = len(text)//2 if len(text)%4 == 0 else len(text)//2+1
                        assert trunc % 2 == 0
                        text = text[:trunc]
                    metrics = text[-1]
                    f.write(metrics + '\n')

                    # text = re_f.read()
                    # metrics = text[text.rindex(f'QoI:'):text.rindex('\n')]  # rindex即从最后索引
                    # f.write(metrics + '\n\n')
                print(1)



'''生成hyper tuning表格'''
def gen_hypertune_csv(x_dir, sum_file):
    ans = np.zeros((12, 5))

    with open(sum_file, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            if line == '\n': continue
            # 如下两个if-else交替执行
            if line.endswith('output.txt\n'):  # 定位往哪填
                json_file = os.path.dirname(line) + '\\params.json'
                params = json.load(open(json_file, 'r'))
                idx1 = int(params['input_args']['g2a_hops'])
                idx2 = int(params['input_args']['map_size']) // 3 - 1
            else:  # 填数
                values = [0 for _ in range(len(METRICS))]
                for col, metric in enumerate(METRICS):
                    if metric in line:
                        start = line.index(metric) + len(metric) + 2  # +2 是适配output.txt的格式
                        end = start + line[start:].index('.') + 4  # 每个scalar数据保留三位小数
                        values[col] = float(line[start:end])
                    else:
                        raise ValueError
                ans[idx1*4+idx2] = values
    pd.DataFrame(ans).to_csv(x_dir + f'/hyper_ALL.csv', index=None, header=0)




'''生成five表格'''
def gen_five_csv(x_dir, sum_file):
    print(x_dir)
    if x_dir.endswith('uavnum'):
        x_index = FIVE_UN_INDEX
        key = 'uav_num'
    elif x_dir.endswith('aoith'):
        x_index = FIVE_AT_INDEX
        key = 'aoith'
    elif x_dir.endswith('txth'):
        x_index = FIVE_TT_INDEX
        key = 'txth'
    elif x_dir.endswith('amount'):
        x_index = FIVE_AM_INDEX
        key = 'user_data_amount'
    elif x_dir.endswith('updatenum'):
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
        df.to_csv(x_dir + f'/five_{ALGOS[i]}_{key}.csv')
    np_all = np.hstack([df.values for df in dfs])
    pd.DataFrame(np_all).to_csv(x_dir + f'/five_ALL_{key}.csv', index=None, header=0)


def main(x_dir, five=False, hypertune=False):
    sum_file = x_dir + '/summary.txt'
    write_summary(x_dir, sum_file)
    if five:
        gen_five_csv(x_dir, sum_file)
    if hypertune:
        gen_hypertune_csv(x_dir, sum_file)
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--x_dir', type=str)
    args = parser.parse_args()
    main(args.x_dir, hypertune=True)

    


