import pandas as pd
import os.path as osp
import sys
import logging
import argparse
import os
import shutil
import importlib.util
import torch
import gym
import copy
from datetime import datetime

from tensorboardX import SummaryWriter
from algorithms.GCRL.envs.model.agent import Agent
from algorithms.GCRL.method.trainer import MPRLTrainer
from algorithms.GCRL.method.memory import ReplayMemory
from algorithms.GCRL.method.explorer import Explorer
from algorithms.GCRL.policies.policy_factory import policy_factory
import numpy as np


def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    np.random.seed(seed)
    return None


def main():

    from get_args import parse_args
    input_args, env_args = parse_args()
    input_args.algo = 'GCRL'  # 硬编码
    
    from main_DPPO import getRunArgs, getAlgArgs, override

    run_args = getRunArgs(input_args)
    print('debug =', run_args.debug)
    print('test =', run_args.test)
    
    from envs.env_mobile import EnvMobile
    env_fn_train, env_fn_test = EnvMobile, EnvMobile
    from env_configs.wrappers.env_wrappers import SubprocVecEnv


    envs_train = SubprocVecEnv([env_fn_train(env_args, input_args, phase='train') for _ in range(input_args.n_thread)])
    envs_test = SubprocVecEnv([env_fn_test(env_args, input_args, phase='test') for _ in range(1)])
    dummy_env = env_fn_train(env_args, input_args, phase='dummy')

    alg_args = getAlgArgs(run_args, input_args, dummy_env)

    class A:
        __name__ = 'GCRLAgent'
    alg_args, run_args, input_args = override(alg_args, run_args, input_args, dummy_env, A())

    config = 'algorithms/GCRL/configs/infocom_benchmark/mp_separate_dp.py'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    set_random_seeds(1)

    # configure paths
    if not os.path.exists(input_args.output_dir):
        os.makedirs(input_args.output_dir)
        shutil.copy(config, os.path.join(input_args.output_dir, 'config.py'))
        base_config = os.path.join(os.path.join(os.path.split(config)[0], os.pardir), 'config.py')
        shutil.copy(base_config, os.path.join(input_args.output_dir, 'base_config.py'))

    config = os.path.join(input_args.output_dir, 'config.py')
    log_file = os.path.join(input_args.output_dir, 'output.log')
    rl_weight_file = os.path.join(input_args.output_dir, 'rl_model.pth')

    # 仅仅知道模块名字和路径的情况下import模块
    spec = importlib.util.spec_from_file_location('config', config)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)  # 通过传入模块的spec返回新的被导入的模块对象
    spec.loader.exec_module(config)

    # configure logging
    mode = 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)  # 输出日志信息到磁盘文件
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not input_args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.info('Current config content is :{}'.format(config))

    writer = SummaryWriter(log_dir=input_args.output_dir)




    agent = Agent()
    # human_df = dummy_env.human_df
    if input_args.dataset == 'NCSU' and input_args.poi_num == 33 or input_args.dataset == 'KAIST' and input_args.poi_num == 92:
        postfix = ''
    else:
        postfix = f'-user{input_args.poi_num}'
    csv_name = f'envs/{input_args.dataset}/human{input_args.max_episode_step}{postfix}.csv'
    poi_df = pd.read_csv(osp.join(csv_name))

    # configure policy
    policy_config = config.PolicyConfig()
    policy = policy_factory[policy_config.name]()  # model_predictive_rl
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    policy.set_device(input_args.device)

    # add
    policy_config.obs_dim = dummy_env.obs_space['Box'].shape[1]  # 151 in NCSU with 48 users
    policy_config.act_dim = dummy_env.action_space.shape[0]  # 2
    policy_config.update_num = input_args.update_num

    policy.configure(policy_config, poi_df)

    # read training parameters
    train_config = config.TrainConfig(input_args.debug)
    rl_learning_rate = train_config.train.rl_learning_rate
    num_batches = train_config.train.num_batches
    num_episodes = train_config.train.num_episodes
    sample_episodes = train_config.train.sample_episodes
    warmup_episodes = train_config.train.warmup_episodes
    evaluate_episodes = train_config.train.evaluate_episodes
    target_update_interval = train_config.train.target_update_interval
    evaluation_interval = train_config.train.evaluation_interval
    capacity = train_config.train.capacity
    epsilon_start = train_config.train.epsilon_start
    epsilon_end = train_config.train.epsilon_end
    epsilon_decay = train_config.train.epsilon_decay
    checkpoint_interval = train_config.train.checkpoint_interval

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_value_estimator()
    batch_size = train_config.trainer.batch_size
    optimizer = train_config.trainer.optimizer

    # choose Graph or Vanilla  # 关注一下训练部分的代码
    trainer = MPRLTrainer(model, policy.state_predictor, memory, input_args.device, policy, writer, batch_size, optimizer,
                          input_args.poi_num,
                          reduce_sp_update_frequency=train_config.train.reduce_sp_update_frequency,
                          freeze_state_predictor=train_config.train.freeze_state_predictor,
                          detach_state_predictor=train_config.train.detach_state_predictor,
                          share_graph_model=policy_config.model_predictive_rl.share_graph_model)



    from algorithms.utils import LogClient, LogServer
    logger = LogServer({'run_args': run_args, 'algo_args': alg_args, 'input_args': input_args})
    logger = LogClient(logger)
    explorer = Explorer(envs_train, agent, input_args.device, input_args, logger,
                        memory, policy.gamma, target_policy=policy)

    logging.info('We use random-exploration methods to warm-up.')
    trainer.update_target_model(model)  # target网络的初始参数与model网络一致

    # reinforcement learning
    # policy.set_env(env)
    agent.set_policy(policy)
    agent.print_info()
    # env.set_agent(agent)
    trainer.set_learning_rate(rl_learning_rate)



    episode = 0
    best_val_reward = -1e6
    best_val_model = None

    num_episodes = 3e6 // (input_args.n_thread*input_args.max_episode_step)
    while episode < num_episodes:
        # epsilon-greedy
        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epsilon = epsilon_end
        agent.policy.set_epsilon(epsilon)

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(k=sample_episodes, phase='train', update_memory=True, plot_index=-1)  # 与环境交互采样数据

        # explorer.log('train', episode)
        trainer.optimize_batch(num_batches, episode)  # 训练！其实trainer最核心的就是干这个
        logging.info(f"ep {episode} training is finished. epsilon={epsilon}\n")

        episode += 1

        if episode % target_update_interval == 0:
            trainer.update_target_model(model)
        # evaluate the model
        if episode % evaluation_interval == 0:
            pass
            # average_reward, _, _, _ ,_,_ = explorer.run_k_episodes(k=evaluate_episodes, phase='val',
            #                                                   plot_index=-1)
            # explorer.log('val', episode // evaluation_interval)
            #
            # if episode % checkpoint_interval == 0 and average_reward.mean() > best_val_reward:
            #     logging.info("Best reward model has been changed.")
            #     best_val_reward = average_reward
            #     best_val_model = copy.deepcopy(policy.get_state_dict())
            # # test after every evaluation to check how the generalization performance evolves
            # if test_after_every_eval:
            #     # explorer.run_k_episodes(k=1, phase='test', plot_index=episode)
            #     # explorer.log('test', episode // evaluation_interval)
            #     pass

        if episode % checkpoint_interval == 0:
            current_checkpoint = episode // checkpoint_interval - 1
            save_every_checkpoint_rl_weight_file = rl_weight_file.split('.')[0] + '_' + str(current_checkpoint) + '.pth'
            policy.save_model(save_every_checkpoint_rl_weight_file)

    # # test with the best val model
    if best_val_model is not None:
        policy.load_state_dict(best_val_model)
        torch.save(best_val_model, os.path.join(output_dir, 'best_val.pth'))
        logging.info('Save the best val model with the reward: {}'.format(best_val_reward))

    logging.info('Check the best val model\'s performance')
    # explorer.run_k_episodes(k=1, phase='test', plot_index=66666)

    print('OK!')

if __name__ == '__main__':
    main()
