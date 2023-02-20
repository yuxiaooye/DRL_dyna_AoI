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
    debug = True
    gpu = True
    gpu_id = '0'
    randomseed = 0
    output_dir = 'logs/debug'
    overwrite = True
    config = 'algorithms/GCRL/configs/infocom_benchmark/mp_separate_dp.py'
    test_after_every_eval = True

    # parser = argparse.ArgumentParser('Parse configuration file')
    # parser.add_argument('--config', type=str, default='algorithms/GCRL/configs/infocom_benchmark/mp_separate_dp.py')
    # parser.add_argument('--output_dir', type=str, default='logs/debug')  # output_xxxx
    # parser.add_argument('--overwrite', default=False, action='store_true')
    #
    # parser.add_argument('--weights', type=str)
    # parser.add_argument('--gpu_id', type=str, default='-1')
    # parser.add_argument('--gpu', default=False, action='store_true')
    #
    # parser.add_argument('--debug', default=False, action='store_true')  # 开启debug模式
    # parser.add_argument('--test_after_every_eval', default=False, action='store_true')
    # parser.add_argument('--randomseed', type=int, default=0)
    #
    # parser.add_argument('--vis_html', default=False, action='store_true')
    # parser.add_argument('--plot_loop', default=False, action='store_true')
    # parser.add_argument('--moving_line', default=False, action='store_true')
    #
    # parser.add_argument('--clip', type=str, default='-1,-1')  # 绘制的时间跨度，闭区间（默认不裁剪
    # parser.add_argument('--users', type=str, default='-1')  # 绘制的人群，列表（默认不删人
    # parser.add_argument('--dataset', type=str, default='Purdue')  # 数据集
    # parser.add_argument('--draw_only', default=False, action='store_true')  # 当外存有机器人数据时，置为true
    #
    # sys_args = parser.parse_args()




    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    set_random_seeds(randomseed)

    # configure paths
    make_new_dir = True
    if os.path.exists(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            # key = input('Output directory already exists! Overwrite the folder? (y/n)')
            if True:
            # if key == 'y':
                shutil.rmtree(output_dir)
            else:
                make_new_dir = False
                exit(0)
    if make_new_dir:
        os.makedirs(output_dir)
        shutil.copy(config, os.path.join(output_dir, 'config.py'))
        base_config = os.path.join(os.path.join(os.path.split(config)[0], os.pardir), 'config.py')
        shutil.copy(base_config, os.path.join(output_dir, 'base_config.py'))

    config = os.path.join(output_dir, 'config.py')
    log_file = os.path.join(output_dir, 'output.log')
    rl_weight_file = os.path.join(output_dir, 'rl_model.pth')

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
    level = logging.INFO if not debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.info('Current config content is :{}'.format(config))
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    if torch.cuda.is_available() and gpu:
        logging.info('Using gpu: %s' % gpu_id)
    else:
        logging.info('Using device: cpu')

    writer = SummaryWriter(log_dir=output_dir)

    # configure environment
    # env = gym.make('CrowdSim-v0')
    from envs.env_mobile import EnvMobile
    env_fn_train, env_fn_test = EnvMobile, EnvMobile
    from env_configs.wrappers.env_wrappers import SubprocVecEnv
    from get_args import parse_args
    input_args, env_args = parse_args()

    if input_args.debug:
        input_args.n_thread = 2

    envs_train = SubprocVecEnv([env_fn_train(env_args, input_args, phase='train') for _ in range(input_args.n_thread)])
    envs_test = SubprocVecEnv([env_fn_test(env_args, input_args, phase='test') for _ in range(1)])
    dummy_env = env_fn_train(env_args, input_args, phase='dummy')


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
    policy.set_device(device)
    policy.configure(policy_config, poi_df)

    # read training parameters
    train_config = config.TrainConfig(debug)
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
    trainer = MPRLTrainer(model, policy.state_predictor, memory, device, policy, writer, batch_size, optimizer,
                          input_args.poi_num,
                          reduce_sp_update_frequency=train_config.train.reduce_sp_update_frequency,
                          freeze_state_predictor=train_config.train.freeze_state_predictor,
                          detach_state_predictor=train_config.train.detach_state_predictor,
                          share_graph_model=policy_config.model_predictive_rl.share_graph_model)

    explorer = Explorer(envs_train, agent, device, writer, memory, policy.gamma, target_policy=policy)

    logging.info('We use random-exploration methods to warm-up.')
    trainer.update_target_model(model)  # target网络的初始参数与model网络一致

    # reinforcement learning
    # policy.set_env(env)
    agent.set_policy(policy)
    agent.print_info()
    # env.set_agent(agent)
    trainer.set_learning_rate(rl_learning_rate)

    # warmup: fill the memory pool with some experience
    # agent.policy.set_epsilon(1)
    # explorer.run_k_episodes(k=warmup_episodes, phase='train', update_memory=True, plot_index=-1)  # 100
    # logging.info('Warm-up finished!')
    # logging.info('Experience set size: %d/%d\n', len(memory), memory.capacity)

    episode = 0
    best_val_reward = -1e6
    best_val_model = None

    while episode < num_episodes:
        # epsilon-greedy
        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epsilon = epsilon_end
        agent.policy.set_epsilon(epsilon)

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(k=sample_episodes, phase='train', update_memory=True, plot_index=-1)  # 与环境交互采样数据

        explorer.log('train', episode)
        trainer.optimize_batch(num_batches, episode)  # TODO 训练！其实trainer最核心的就是干这个 其他都可看可不看
        logging.info(f"ep {episode} training is finished. epsilon={epsilon}\n")

        episode += 1

        if episode % target_update_interval == 0:
            trainer.update_target_model(model)
        # evaluate the model
        if episode % evaluation_interval == 0:
            average_reward, _, _, _ ,_,_ = explorer.run_k_episodes(k=evaluate_episodes, phase='val',
                                                              plot_index=-1)
            explorer.log('val', episode // evaluation_interval)

            if episode % checkpoint_interval == 0 and average_reward.mean() > best_val_reward:
                logging.info("Best reward model has been changed.")
                best_val_reward = average_reward
                best_val_model = copy.deepcopy(policy.get_state_dict())
            # test after every evaluation to check how the generalization performance evolves
            if test_after_every_eval:
                explorer.run_k_episodes(k=1, phase='test', plot_index=episode)
                explorer.log('test', episode // evaluation_interval)

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
    explorer.run_k_episodes(k=1, phase='test', plot_index=66666)

    print('OK!')

if __name__ == '__main__':
    main()
