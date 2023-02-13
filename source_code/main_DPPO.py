import sys
import os

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('---------------------')
print(proj_dir)
sys.path.append(proj_dir)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datetime import datetime
import importlib
import json
import copy
from algorithms.utils import Config, LogClient, LogServer, mem_report
from algorithms.algo.main import OnPolicyRunner

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import argparse


def getRunArgs(input_args):
    run_args = Config()
    run_args.device = input_args.device


    '''yyx add start'''
    run_args.profiling = False
    run_args.debug = input_args.debug
    run_args.test = input_args.test
    run_args.init_checkpoint = input_args.init_checkpoint
    run_args.group = input_args.group
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


def getAlgArgs(run_args, input_args, env):
    assert input_args.env in ['mobile'] and input_args.algo in ['DPPO', 'CPPO', 'DMPO', 'IC3Net', 'IA2C']
    env_str = input_args.env[0].upper() + input_args.env[1:]
    config = importlib.import_module(f"algorithms.config.{env_str}_{input_args.algo}")
    # 在这里，得到了alg_args.agent_args.action_space
    alg_args = config.getArgs(run_args.radius_p, run_args.radius_v, run_args.radius_pi, env)
    return alg_args


def initAgent(logger, device, agent_args, input_args):
    return agent_fn(logger, device, agent_args, input_args)



def override(alg_args, run_args, input_args):
    if run_args.debug:
        alg_args.model_batch_size = 5  # 用于训练一次model的traj数量
        alg_args.max_ep_len = 5
        alg_args.rollout_length = 20 * input_args.n_thread
        alg_args.test_length = 600  # 测试episode的最大步长
        alg_args.model_buffer_size = 10
        alg_args.n_model_update = 3
        alg_args.n_model_update_warmup = 3
        alg_args.n_warmup = 1
        alg_args.model_prob = 1  # 规定一定会执行从model中采样用于更新policy的经验
        # 注意: n_iter*rollout_length得比一个episode长，不然一次train episode done都不触发，train_trajs不会保存到外存
        alg_args.n_iter = 7
        alg_args.n_test = 1
        alg_args.n_traj = 4
        alg_args.n_inner_iter = 2
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
    run_args.name = '{}_{}'.format(timenow, agent_fn.__name__)


    # tune env
    if input_args.snr != 200:
        run_args.name += f'_SNR={input_args.snr}'
    if input_args.dyna_level != '2':
        run_args.name += f'_DynaLevel={input_args.dyna_level}'
    if input_args.init_energy != 719280:
        run_args.name += f'_InitEnergy={input_args.init_energy}'
    if input_args.user_data_amount != 1:
        run_args.name += f'_DataAmount={input_args.user_data_amount}'
    if input_args.update_num != 10:
        run_args.name += f'_UpdateNum={input_args.update_num}'
    if input_args.future_obs != 0:
        run_args.name += f'_FutureObs={input_args.future_obs}'
    if input_args.use_fixed_range:
        run_args.name += f'_FixedRange'
    if not input_args.use_extended_value:
        run_args.name += f'_NotUseExtendedValue'
    if input_args.use_mlp_model:
        run_args.name += f'_MLPModel'
    if input_args.multi_mlp:
        run_args.name += f'_MultiMLP'

    # tune algo
    if input_args.lr is not None:
        run_args.name += f'_LR={input_args.lr}'
        alg_args.agent_args.lr = input_args.lr
    if input_args.lr_v is not None:
        run_args.name += f'_LR-V={input_args.lr_v}'
        alg_args.agent_args.lr_v = input_args.lr_v
    if input_args.debug_use_stack_frame:
        run_args.name += f'_UseStackFrame'
    final = '../{}/{}'.format(input_args.output_dir, run_args.name)
    run_args.output_dir = final
    input_args.output_dir = final
    
    alg_args.algo = input_args.algo
    alg_args.debug_use_stack_frame = input_args.debug_use_stack_frame

    return alg_args, run_args, input_args


def parse_args():
    parser = argparse.ArgumentParser()
    # 已经验证这里的参数可被存入params.json

    parser.add_argument('--debug', action='store_true', default=False, )
    parser.add_argument('--test', action='store_true', default=False, )
    parser.add_argument('--env', type=str, default='mobile')
    parser.add_argument('--algo', type=str, required=False, default='DPPO', help="algorithm(DMPO/IC3Net/CPPO/DPPO/IA2C) ")
    parser.add_argument('--device', type=str, required=False, default='cuda:0', help="device(cpu/cuda:0/cuda:1/...) ")
    parser.add_argument("--dataset", type=str, default='NCSU', choices=['NCSU'])
    # dirs
    parser.add_argument("--output_dir", type=str, default='runs/debug', help="which fold to save under 'runs/'")
    parser.add_argument('--group', type=str, default='debug', help='填写我对一组实验的备注，作用与wandb的group和tb的实验保存路径')
    # system stub
    parser.add_argument('--mute_wandb', default=False, action='store_true')
    # tune agent
    parser.add_argument('--init_checkpoint', type=str)  # load pretrained model
    parser.add_argument('--n_thread', type=int, default=2)
    # tune algo
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_v', type=float)
    parser.add_argument('--debug_use_stack_frame', action='store_true')
    parser.add_argument('--use-extended-value', action='store_false', help='反逻辑，仅用于DPPO')
    parser.add_argument('--use-mlp-model', action='store_true', help='将model改为最简单的mlp，仅用于DMPO')
    parser.add_argument('--multi-mlp', action='store_true', help='在model中分开预测obs中不同类别的信息，仅用于DMPO')
    # tune env
    parser.add_argument('--use_fixed_range', action='store_true')
    parser.add_argument('--snr', type=float, default=200)
    parser.add_argument('--init_energy', type=float, default=719280)
    parser.add_argument('--dyna_level', type=str, default='2', help='指明读取不同难度的poi_QoS.npy')
    parser.add_argument('--user_data_amount', type=int, default=1)
    parser.add_argument('--update_num', type=int, default=10)
    parser.add_argument('--future_obs', type=int, default=0)
    args = parser.parse_args()

    if args.multi_mlp:
        assert args.use_mlp_model

    if args.debug:
        assert args.group == 'debug'
    args.output_dir = f'runs/{args.group}'
    return args

input_args = parse_args()

def record_input_args(input_args, env_args, output_dir):
    params = dict()
    from envs.config_3d import Config
    env_config = Config(env_args, input_args)
    params['input_args'] = vars(input_args)
    params['env_config'] = env_config.dict

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        f.write(json.dumps(params))


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

from envs.env_mobile import EnvMobile
env_fn_train, env_fn_test = EnvMobile, EnvMobile


env_args = {  # 这里环境类的参数抄昊宝
    "action_mode": 3,
    "render_mode": True,
    "emergency_threshold": 100,
    "collect_range": input_args.snr,
    "initial_energy": input_args.init_energy,
    "user_data_amount": input_args.user_data_amount,
    "update_num": input_args.update_num,
}

run_args = getRunArgs(input_args)
print('debug =', run_args.debug)

dummy_env = env_fn_train(env_args, input_args, phase='dummy')
alg_args = getAlgArgs(run_args, input_args, dummy_env)
alg_args, run_args, input_args = override(alg_args, run_args, input_args)
record_input_args(input_args, env_args, run_args.output_dir)

from env_configs.wrappers.env_wrappers import SubprocVecEnv
envs_train = SubprocVecEnv([env_fn_train(env_args, input_args, phase='train') for _ in range(input_args.n_thread)])
envs_test = SubprocVecEnv([env_fn_test(env_args, input_args, phase='test') for _ in range(1)])


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

logger = LogServer({'run_args': run_args, 'algo_args': alg_args})
logger = LogClient(logger)
# logger同时被传入agent类和runner类
agent = initAgent(logger, run_args.device, alg_args.agent_args, input_args)
OnPolicyRunner(logger=logger, agent=agent, envs_learn=envs_train, envs_test=envs_test, dummy_env=dummy_env,
               run_args=run_args, alg_args=alg_args, input_args=input_args).run()
print('OK!')
