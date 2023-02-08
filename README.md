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
│   │   ├── env_mobile.py  # 群体感知环境定义
│   ├── env_configs  # 环境参数
│   └── tools  # 功能脚本，预处理和后处理
│   │   ├── post
│   │   │   ├── vis_gif.py  # 训练后绘制html可视化文件
│   ├── main_DPPO.py  # 训练启动入口脚本
```



## How to train

conda环境请参见服务器77的yyx_adept环境

启动训练：

```sh
cd source_code
python main_DPPO.py
```

命令行参数：

- --debug：开启debug模式，快速验证代码全流程无bug，将实验结果存入`runs/debug`路径
- --group_postfix : 将实验结果存入`runs/foo`路径
- --algo：选择算法

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
```

除tensorboard外，还实现了基于wandb的可视化，本地结果存放在`wandb`路径下。

绘制html格式的可视化轨迹：

```sh
cd source_code
python tools/post/vis_gif.py --output_dir <OUTPUT_DIR>
```

在实验结果文件夹下生成`_drawUavLines.html`：

<img src="https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202302081401687.png" alt="image-20230208140132293" style="zoom:33%;" />

## How to inference

加载保存的actor模型，进行测试：

```sh
cd source_code
python main_DPPO.py --test --init_checkpoint <ckpt_dir>  # <ckpt_dir> usually ends with "best_actor.pt"
```

测试结果默认保存在`runs/debug`路径下