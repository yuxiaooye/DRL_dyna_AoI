'''
注意：这里DPPO是yali du的使用扩展值函数进行通信的DPPO
from algorithms拷贝自yali du的algorithms文件夹

- 在哪里初始化agent类？
- 在哪里初始化env类？
- 在哪里把env的状态和动作空间告诉agent？ 在initAgent函数中携带了空间信息吗？
'''
import sys
import os

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('---------------------')
print(proj_dir)
sys.path.append(proj_dir)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from logging import log
from datetime import datetime
from re import T
import importlib
import ray
import time
import warnings
import json
import copy
import gym
from algorithms.utils import Config, LogClient, LogServer, mem_report
from torch import distributed as dist
from algorithms.algo.main import OnPolicyRunner

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import torch
import argparse


def getRunArgs(input_args):
    run_args = Config()
    run_args.n_thread = 1
    run_args.parallel = False
    run_args.device = input_args.device
    run_args.name = f'standard{input_args.name}'
    run_args.n_cpu = 1 / 4
    run_args.n_gpu = 0

    '''yyx add start'''
    run_args.profiling = False
    run_args.debug = input_args.debug
    run_args.test = input_args.test
    run_args.init_checkpoint = input_args.init_checkpoint
    run_args.group_postfix = input_args.group_postfix
    run_args.mute_wandb = input_args.mute_wandb
    '''yyx add end'''

    run_args.radius_v = 1
    run_args.radius_pi = 1
    run_args.radius_p = 1

    run_args.radius_pi = 1
    run_args.radius_p = 1

    run_args.start_step = 0
    run_args.save_period = 1800  # in seconds
    run_args.log_period = int(20)
    run_args.seed = None
    return run_args


def initArgs(run_args, env_train, env_test, input_arg):
    ref_env = env_train
    if input_arg.env in ['mobile'] or input_arg.algo in ['CPPO', 'DMPO', 'IC3Net', 'IA2C']:
        env_str = input_arg.env[0].upper() + input_arg.env[1:]
        config = importlib.import_module(f"algorithms.config.{env_str}_{input_args.algo}")

    # 在这里，得到了alg_args.agent_args.action_space
    alg_args = config.getArgs(run_args.radius_p, run_args.radius_v, run_args.radius_pi, ref_env)
    return alg_args


def initAgent(logger, device, agent_args, input_args):
    return agent_fn(logger, device, agent_args, input_args)


def initEnv(input_args):
    sys.path = [os.path.dirname(os.path.dirname(__file__))] + sys.path  # 优先从本项目的adept包中调用
    from env_mobile.env_mobile import EnvMobile
    env_fn_train, env_fn_test = EnvMobile, EnvMobile
    return env_fn_train, env_fn_test


def override(alg_args, run_args, env_fn_train, input_args):
    alg_args.env_fn = env_fn_train

    if run_args.debug:
        alg_args.model_batch_size = 4
        alg_args.max_ep_len = 5
        alg_args.rollout_length = 20
        alg_args.test_length = 1
        alg_args.model_buffer_size = 10
        alg_args.n_model_update = 3
        alg_args.n_model_update_warmup = 3
        alg_args.n_warmup = 1
        # 不过n_iter*rollout_length得比一个episode长，不然一次done都不触发，train_trajs不会保存到外存
        alg_args.n_iter = 10
        alg_args.n_test = 1
        alg_args.n_traj = 4
        alg_args.n_inner_iter = 10
    if run_args.test:  # here it's what I want!
        run_args.debug = True
        alg_args.n_warmup = 0
        alg_args.n_test = 10
    if run_args.profiling:
        raise NotImplementedError
    if run_args.seed is None:
        # 这里这里seed是随机的，我还是改成全统一的吧~
        # run_args.seed = int(time.time() * 1000) % 65536
        run_args.seed = 1

    '''yyx add begin'''
    timenow = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    run_args.name = '{}_{}_{}_{}'.format(timenow, run_args.name, env_fn_train.__name__, agent_fn.__name__)
    if input_args.lr is not None:
        run_args.name += f'_LR={input_args.lr}'
        alg_args.agent_args.lr = input_args.lr
    if input_args.lr_v is not None:
        run_args.name += f'_LR-V={input_args.lr_v}'
        alg_args.agent_args.lr_v = input_args.lr_v
    if input_args.snr != 200:
        run_args.name += f'_SNR={input_args.snr}'
    if input_args.init_energy != 719280:
        run_args.name += f'_InitEnergy={input_args.init_energy}'
    if input_args.dyna_level != '':
        run_args.name += f'_DynaLevel={input_args.dyna_level}'
    run_args.output_dir = '../{}/{}'.format(input_args.output_dir, run_args.name)

    alg_args.algo = input_args.algo
    '''yyx add end'''
    return alg_args, run_args


def parse_args():
    parser = argparse.ArgumentParser()
    # 已经验证这里的参数可被存入params.json

    parser.add_argument('--debug', action='store_true', default=False, )
    parser.add_argument('--test', action='store_true', default=False, )

    parser.add_argument('--env', type=str, default='mobile')
    parser.add_argument('--algo', type=str, required=False, default='DPPO', help="algorithm(DMPO/IC3Net/CPPO/DPPO/IA2C) ")
    parser.add_argument('--device', type=str, required=False, default='cuda:0', help="device(cpu/cuda:0/cuda:1/...) ")
    parser.add_argument('--name', type=str, required=False, default='', help="the additional name for logger")
    # dirs
    parser.add_argument("--dataset", type=str, default='NCSU', choices=['NCSU'])
    parser.add_argument("--output_dir", type=str, default='runs/debug', help="which fold to save under 'runs/'")

    parser.add_argument('--mute_wandb', default=False, action='store_true')
    parser.add_argument('--group_postfix', type=str, default='debug', help='填写我对一组实验的备注，作用与wandb的group和tb的实验保存路径')
    parser.add_argument('--init_checkpoint', type=str)  # load pretrained model

    # tune algo
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_v', type=float)
    # tune env
    parser.add_argument('--snr', type=float, default=200)
    parser.add_argument('--init_energy', type=float, default=719280)
    parser.add_argument('--dyna_level', type=str, default='', help='指明读取不同难度的poi_QoS.npy')

    args = parser.parse_args()

    # == yyx add ==
    if args.dataset == 'purdue':
        args.setting_dir = 'purdue59move'
    elif args.dataset == 'NCSU':
        args.setting_dir = 'NCSU33move'

    if args.debug:
        args.group_postfix = 'debug'
        args.output_dir = 'runs/debug'
        args.Max_train_steps = 100
        args.T_horizon = 10
        args.eval_interval = 50
        args.save_interval = 20
        args.n_rollout_threads = 3
    args.output_dir = f'runs/{args.group_postfix}'

    return args

input_args = parse_args()


# TODO 这个还需要么？可以和昊哥的config整合在一起吧？起码把路径放在一起~
def import_env_config(dataset_str, args=None):
    if dataset_str == 'purdue':
        from src.config.roadmap_config.purdue.env_config_purdue import env_config as env_config
        return env_config
    elif dataset_str == 'NCSU':
        from src.config.roadmap_config.NCSU.env_config_NCSU import env_config as env_config
        return env_config
    else:
        raise NotImplementedError

def yyx_get_env_args(args):
    yyx_args = dict()
    my_env_config = import_env_config(args.dataset, args)
    yyx_args['my_env_config'] = my_env_config
    yyx_args['args'] = args
    return yyx_args


def hook_yyx_args(yyx_args, output_dir):
    # hook the params
    tmp_dict = copy.deepcopy(yyx_args)
    tmp_dict['args'] = vars(tmp_dict['args'])
    tmp_dict['setting_dir'] = yyx_args['args'].setting_dir

    print('------------------os.getcwd()----------')
    print(os.getcwd())
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        f.write(json.dumps(tmp_dict))


if input_args.algo == 'IA2C':
    from algorithms.algo.agent.IA2C import IA2C as agent_fn
elif input_args.algo == 'IC3Net':
    from algorithms.algo.agent.IC3Net import IC3Net as agent_fn
elif input_args.algo == 'DPPO':
    from algorithms.algo.agent.DPPO import DPPOAgent as agent_fn
elif input_args.algo == 'CPPO':
    from algorithms.algo.agent.CPPO import CPPOAgent as agent_fn
elif input_args.algo == 'DMPO':
    from algorithms.algo.agent.DMPO import DMPOAgent as agent_fn

env_fn_train, env_fn_test = initEnv(input_args)

data_amount = 1
bar = 100

yyx_args = yyx_get_env_args(parse_args())
hao_args = {  # 这里环境类的参数抄昊宝
    'test_mode': True,
    'save_path': '.',
    "controller_mode": True,
    "seed": 1,
    "action_mode": 3,
    "weighted_mode": True,
    "mip_mode": False,
    "noisy_power": -90,
    "tx_power": 20,
    "render_mode": True,
    "user_data_amount": data_amount,
    "uav_num": yyx_args['my_env_config']['num_uav'],
    "emergency_threshold": bar,
    "collect_range": input_args.snr,
    "initial_energy": input_args.init_energy,
}
hao_args["max_episode_step"] = yyx_args['my_env_config']['num_timestep']  # TODO unity it

env_train = env_fn_train(hao_args, yyx_args=yyx_args)
env_test = env_fn_test(hao_args, yyx_args=yyx_args)

run_args = getRunArgs(input_args)
print('debug =', run_args.debug)
alg_args = initArgs(run_args, env_train, env_test, input_args)
alg_args, run_args = override(alg_args, run_args, env_fn_train, input_args)
hook_yyx_args(yyx_args, run_args.output_dir)

env_train.args.output_dir = run_args.output_dir  # 统一输出路径
env_test.args.output_dir = run_args.output_dir
env_train.phase = 'train'
env_test.phase = 'test'

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

logger = LogServer({'run_args': run_args, 'algo_args': alg_args})
logger = LogClient(logger)
# logger同时被传入agent类和runner类
agent = initAgent(logger, run_args.device, alg_args.agent_args, input_args)
OnPolicyRunner(logger=logger, run_args=run_args, alg_args=alg_args, agent=agent,
               env_learn=env_train, env_test=env_test, env_args=input_args).run()
print('OK!')
