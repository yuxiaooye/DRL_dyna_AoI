from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from algorithms.algo.agent.DPPO import DPPOAgent
from algorithms.models import MLP, CategoricalActor
from torch.optim import Adam

def get_controller_init_hidden(batch_size, hidden_channels, hidden_size, device):
    init_hidden_states = []
    for hidden_channel in hidden_channels:
        init_hidden_states.append((torch.zeros(batch_size, hidden_channel, hidden_size[0], hidden_size[1]).to(device),
                                   torch.zeros(batch_size, hidden_channel, hidden_size[0], hidden_size[1]).to(device)))
    return init_hidden_states


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channel, hidden_channel, hidden_size, kernel_size, bias=True, is_bn=True):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_channel: int
            Number of channels of input tensor.
        hidden_channel: int
            Number of channels of hidden state.
        hidden_size: (int, int)
            Height and width of input tensor as (height, width).
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = hidden_size
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               # nn.init.xavier_uniform_,

                               nn.init.calculate_gain('conv2d'))
        # if is_bn:
        #     self.conv = nn.Sequential(
        #         init_(nn.Conv2d(in_channels=self.input_channel + self.hidden_channel,
        #                         out_channels=4 * self.hidden_channel,
        #                         kernel_size=self.kernel_size,
        #                         padding=self.padding,
        #                         bias=self.bias)),
        #         nn.BatchNorm2d(4 * self.hidden_channel)
        #     )
        # else:
        self.conv = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.input_channel + self.hidden_channel,
                            out_channels=4 * self.hidden_channel,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            bias=self.bias)),
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # peep_hole

        combined_conv = self.conv(combined)

        combined_conv = torch.layer_norm(combined_conv, combined_conv.shape[1:])

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channel, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMLayers(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    """
    Initialize ConvLSTM cell.

    Parameters
    ----------
    input_channel: int
        Number of channels of input tensor.
    hidden_channel: int
        Number of channels of hidden state.
    hidden_size: (int, int)
        Height and width of input tensor as (height, width).
    kernel_size: int
        Size of the convolutional kernel.

    """

    def __init__(self, input_channel, hidden_channels, hidden_size, kernel_size):
        super(ConvLSTMLayers, self).__init__()
        print('ConvLSTM3')
        self.input_channels = [input_channel] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self._all_layers = nn.ModuleList()
        self.hidden_size = hidden_size
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            # if i == self.num_layers - 1:
            #     cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.hidden_size,
            #                         (self.kernel_size, self.kernel_size), is_bn=False)
            # else:
            #     cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.hidden_size,
            #                         (self.kernel_size, self.kernel_size), is_bn=False)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.hidden_size,
                                (self.kernel_size, self.kernel_size))
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input, layers_hidden_states):
        x = input
        layers_output = []
        for i in range(self.num_layers):  # num_layers=1,较简单的LSTM
            # all cells are initialized in the first step
            name = 'cell{}'.format(i)
            # do forward
            (h, c) = layers_hidden_states[i]
            x, new_c = getattr(self, name)(x, (h, c))  # 这里为什么h和c都需要输入？
            layers_hidden_states[i] = (x, new_c)  # 写当前时刻的hidden states
            layers_output.append(x)
        return layers_output, layers_hidden_states

class PredictiveModel(nn.Module):
    def __init__(self, input_channel, hidden_channels, frame_size, cnn_kernel_size, rnn_kernel_size, device, input_len,
                 seq_len,
                 ):
        super(PredictiveModel, self).__init__()
        self.input_channel = input_channel
        self.hidden_channels = hidden_channels
        self.frame_size = frame_size
        self.cnn_kernel_size = cnn_kernel_size
        self.rnn_kernel_size = rnn_kernel_size
        self.device = device
        self.input_len = input_len
        self.seq_len = seq_len

        self.cnn_padding = int((self.cnn_kernel_size - 1) / 2)
        self.rnn_padding = int((self.rnn_kernel_size - 1) / 2)

        # def init_(m):
        #     init(m, nn.init.orthogonal_, nn.init.calculate_gain('conv2d'))
        init_ = lambda m: init(m, nn.init.orthogonal_, nn.init.calculate_gain('conv2d'))
        # init_relu = lambda m: init(m, nn.init.orthogonal_, nn.init.calculate_gain('relu'))
        # init_tanh = lambda m: init(m, nn.init.orthogonal_, nn.init.calculate_gain('tanh'))

        # 论文中的Φ
        self.cnn_layer = nn.Sequential(
            init_(nn.Conv2d(
                in_channels=6,
                out_channels=32,
                kernel_size=self.cnn_kernel_size,
                stride=2,
            )),
            nn.ReLU(),
            # nn.Dropout(0.5),

            init_(nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self.cnn_kernel_size,
                stride=1,
            )),
            nn.ReLU(),
        )

        self.hidden_size = (14, 14)
        self.rnn_layers = ConvLSTMLayers(input_channel=64, hidden_channels=self.hidden_channels,
                                         hidden_size=self.hidden_size,
                                         kernel_size=self.rnn_kernel_size, )  # 这个kernel还是指cnn的kernel吧？

        '''论文中的ΦT'''
        self.reverse_layer = nn.Sequential(

            init_(nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self.cnn_kernel_size,
                stride=2,
            )),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Dropout(0.5),
            init_(nn.ConvTranspose2d(
                in_channels=64,
                out_channels=6,
                kernel_size=4,
                stride=2,
            )),
        )

    def forward(self, input_seq):
        """
        :param input_seq: [Batch,Time,Channel,Size_m,Size_n]->[B,T,C,M,N] 也即(32, 7, 6, 64, 64)
        TODO yyx:这里的Channel不包含任何时序上的堆叠,感觉就可以先理解成图片的通道
        :param is_valid: bool
        :return: [B,T,C,M,N]
        """

        # len(hidden)=1,因为LSTM只有一层
        # len(hidden[0])=2,因为h和c都要记下来
        # hidden[0][0].shape=(32, 64, 14, 14)
        hidden = get_controller_init_hidden(batch_size=input_seq.shape[0], hidden_size=self.hidden_size,
                                            hidden_channels=self.hidden_channels, device=self.device)
        # zero_tensor = torch.zeros_like(input_seq[:, 0, ...], device=self.device).unsqueeze(dim=1)
        # input_seq = torch.cat([input_seq, zero_tensor], dim=1)

        output_batch_list = []

        for t in range(input_seq.shape[1]):
            # 输入shape=(32, 6, 64, 64) 输出shape=(32,64,14,14) 卷积后，图片尺寸变小了，通道数变多了~~
            cnn_out = self.cnn_layer(input_seq[:, t, ...])
            _, hidden = self.rnn_layers(cnn_out, hidden)  # 把上一时刻的hidden也作为rnn模块的输入，也即peephole
            output_batch_list.append(self.reverse_layer(hidden[-1][0]).unsqueeze(dim=1))
        output_batch = torch.cat(output_batch_list, dim=1)

        return output_batch



class ConvLSTMAgent(DPPOAgent):
    def __init__(self, logger, device, agent_args, input_args):
        DPPOAgent.__init__(self, logger, device, agent_args, input_args)

        if input_args.g2a_hidden_dim is None:
            self.hidden_dim = 64
        else:
            self.hidden_dim = input_args.g2a_hidden_dim
        self.attention_dim = 32

        self.g2a_embed_hard_net = PredictiveModel(        input_channel=CONF['input_channel'],
        hidden_channels=CONF[32],
        frame_size=64,
        cnn_kernel_size=3,
        rnn_kernel_size=3,
        device=device,
        input_len=8,
        seq_len=7).to(device)

        pi_dict, v_dict = self.pi_args._toDict(), self.v_args._toDict()
        pi_dict['sizes'][0] = self.observation_dim  # share邻居的obs时做并集而不是concat
        v_dict['sizes'][0] = self.observation_dim
        self.actors = nn.ModuleList()
        self.vs = nn.ModuleList()
        for i in range(self.n_agent):
            self.actors.append(CategoricalActor(**pi_dict).to(self.device))
            self.vs.append(MLP(**v_dict).to(self.device))
        self.optimizer_pi = Adam(self.actors.parameters(), lr=self.lr)
        self.optimizer_v = Adam(self.vs.parameters(), lr=self.lr_v)






