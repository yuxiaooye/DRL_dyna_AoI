## Overview

目录树：

```
.
├── source_code
│   ├── algorithms
│   │   ├── algo
│   │   │   ├── agent  # ours和baseline算法类的定义
│   │   │   ├── main.py  # OnPolicyRunner类定义，负责智能体与环境交互
│   ├── envs  
│   │   ├── env_mobile.py  # generate-at-will的aoi定义的场景
│   │   ├── env_mobile_hao.py  # 昊宝ton的aoi定义的场景
│   │   ├── env_mobile_EveryStepUpdate.py  # 介于以上两者之间，每步都生成新包的场景
│   ├── env_configs  # 环境参数
│   └── tools  # 功能脚本，预处理和后处理
│   │   ├── post
│   │   │   ├── vis.py  # 训练后绘制html可视化文件
│   ├── main_DPPO.py  # 训练启动入口脚本
```



## How to train

conda环境为yyx_adept(10.1.114.77)，yyx_ishen(10.1.114.75)。依赖项不多，哪个包缺了手动pip即可。

Docker部署（Optional）
```sh
docker build -t linc/mcs:drl_dyna_aoi-v1 . --network host

xhost +

docker run -it --privileged --net=host --ipc=host --device=/dev/dri:/dev/dri -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --gpus all --name test_mcs linc/mcs:drl_dyna_aoi-v1 /bin/bash
```

启动训练：

```sh
cd source_code
python main_DPPO.py
```

命令行参数：

- `--debug`：开启debug模式，快速验证代码全流程无bug，将实验结果存入`runs/debug`路径
- `--group foo`： 将实验结果存入`runs/foo`路径
- `--algo foo`：选择使用算法foo
- `--n_thread n`：设置多线程环境数为n，加速训练

更多命令行参数的使用方式参见代码。

## Training outputs 

实验结果文件夹包括以下内容：

```sh
.
├── events.out.tfevents.1675849821.omnisky.107733.0  # tensorboard可视化
├── Models  # 保存最优actor模型
├── params.json  # 记录本次实验的参数防止遗忘
├── test_saved_trajs  # 测试episode的最优uav轨迹
└── train_saved_trajs  # 训练episode的最优uav轨迹
└── vis.html  # 根据最优uav轨迹绘制的html可视化文件
```

除tensorboard外，还实现了基于wandb的可视化，本地结果存放在`wandb`路径下。

手动绘制html可视化文件：

```sh
cd source_code
python tools/post/vis.py --output_dir <OUTPUT_DIR>
```

在实验结果文件夹下生成`vis.html`：

<img src="https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202302112014826.png" alt="image-20230211201439409" style="zoom: 25%;" />

批量绘制一个group下所有实验的可视化文件：
```sh
cd source_code
python tools/post/bat_vis.py --group_dir <GROUP_DIR>
```
其中GROUP_DIR是OUTPUT_DIR的父目录。

## How to inference

加载保存的actor模型，进行测试：

```sh
cd source_code
python main_DPPO.py --test --init_checkpoint <ckpt_dir>  # <ckpt_dir> usually ends with "best_actor.pt"
```

测试结果默认保存在`runs/debug`路径下
