import logging
import itertools
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
from torch.nn import Parameter
from algorithms.GCRL.method.base import mlp


class RGL(nn.Module):
    def __init__(self, config, robot_state_dim, human_state_dim):
        super().__init__()

        # design choice
        # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
        self.similarity_function = config.gcn.similarity_function
        self.robot_state_dim = robot_state_dim
        self.human_state_dim = human_state_dim
        self.num_layer = config.gcn.num_layer  # 进行多少次图卷积操作
        self.X_dim = config.gcn.X_dim
        self.layerwise_graph = config.gcn.layerwise_graph
        self.skip_connection = config.gcn.skip_connection

        logging.info('Similarity_func: {}'.format(self.similarity_function))
        logging.info('Layerwise_graph: {}'.format(self.layerwise_graph))
        logging.info('Skip_connection: {}'.format(self.skip_connection))
        logging.info('Number of layers: {}'.format(self.num_layer))

        X_dim = config.gcn.X_dim  # embedding后的user和无人机的feature维度
        wr_dims = config.gcn.wr_dims
        wh_dims = config.gcn.wh_dims
        final_state_dim = config.gcn.final_state_dim

        # embedding层
        self.w_r = mlp(robot_state_dim, wr_dims, last_relu=True)  # inputs,64,32
        self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)  # inputs,64,32


        if self.similarity_function == 'embedded_gaussian':
            self.w_a = Parameter(torch.randn(self.X_dim, self.X_dim))  # 论文中的Wc，embedded gaussian中的嵌入矩阵。
        elif self.similarity_function == 'concatenation':
            self.w_a = mlp(2 * X_dim, [2 * X_dim, 1], last_relu=True)


        embedding_dim = self.X_dim
        self.Ws = torch.nn.ParameterList()  # 创建了很多个Wh(l),并且每一次图卷积迭代时的Wh参数不同！
        for i in range(self.num_layer):
            if i == 0:
                self.Ws.append(Parameter(torch.randn(self.X_dim, embedding_dim)))
            elif i == self.num_layer - 1:
                self.Ws.append(Parameter(torch.randn(embedding_dim, final_state_dim)))
            else:
                self.Ws.append(Parameter(torch.randn(embedding_dim, embedding_dim)))


    def compute_similarity_matrix(self, X):  # 可作为调参重点,map
        if self.similarity_function == 'embedded_gaussian':  # 使用embedded gaussian方法 根据Zt得到Ct
            A = torch.matmul(torch.matmul(X, self.w_a), X.permute(0, 2, 1))  # X.permute起到转置的作用
            normalized_A = softmax(A, dim=2)  # 在最里面的维度进行softmax，也即关系矩阵Ct的行和为1
        elif self.similarity_function == 'gaussian':
            A = torch.matmul(X, X.permute(0, 2, 1))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'cosine':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = torch.div(A, norm_matrix)
        elif self.similarity_function == 'cosine_softmax':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = softmax(torch.div(A, norm_matrix), dim=2)
        elif self.similarity_function == 'concatenation':
            indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]
            selected_features = torch.index_select(X, dim=1, index=torch.LongTensor(indices).reshape(-1))
            pairwise_features = selected_features.reshape((-1, X.size(1) * X.size(1), X.size(2) * 2))
            A = self.w_a(pairwise_features).reshape(-1, X.size(1), X.size(1))
            normalized_A = A
        elif self.similarity_function == 'squared':
            A = torch.matmul(X, X.permute(0, 2, 1))
            squared_A = A * A
            normalized_A = squared_A / torch.sum(squared_A, dim=2, keepdim=True)
        elif self.similarity_function == 'equal_attention':
            normalized_A = (torch.ones(X.size(1), X.size(1)) / X.size(1)).expand(X.size(0), X.size(1), X.size(1))
        elif self.similarity_function == 'diagonal':
            normalized_A = (torch.eye(X.size(1), X.size(1))).expand(X.size(0), X.size(1), X.size(1))
        else:
            raise NotImplementedError

        return normalized_A

    def forward(self, state):
        robot_states, human_states = state

        # print('GCL.forward被调用')
        # compute feature matrix X。
        robot_state_embedings = self.w_r(robot_states)  # shape = (batch, robot_num, embedding_dim)
        human_state_embedings = self.w_h(human_states)
        X = torch.cat([robot_state_embedings, human_state_embedings], dim=1)  # 论文中的Zt。# shape = (batch, total_num, embedding_dim)

        # compute matrix A. 原文的关系矩阵Ct
        if not self.layerwise_graph:
            normalized_A = self.compute_similarity_matrix(X)
            self.A = normalized_A[0, :, :].data.cpu().numpy()  # total_num x total_num  这里创建了self.A 将关系矩阵拿出计算图
            # print('关系矩阵：', self.A)  # 在这里打印时机是不对的 应该在explorer中

        next_H = H = X
        for i in range(self.num_layer):  # 2 论文中的K-1 Times图卷积操作只进行两次图卷积操作？
            if self.layerwise_graph:  # False.
                A = self.compute_similarity_matrix(H)  # 每次图卷积操作拥有自己的邻接矩阵，同时高层级的邻接矩阵由低层推导而来。 https://zhuanlan.zhihu.com/p/398059770
                next_H = relu(torch.matmul(torch.matmul(A, H), self.Ws[i]))
            else:  # (A x H) x W_i
                next_H = relu(torch.matmul(torch.matmul(normalized_A, H), self.Ws[i]))  # 论文中 Relu(CtHt(l)Wh(l))

            if self.skip_connection:  # 论文中ReLu(.) + Ht(l)
                next_H += H
            H = next_H

        return next_H  # 得到Ht(-1),将在外部被split为无人机和用户的特征
