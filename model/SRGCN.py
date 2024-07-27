#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # self.att = Parameter(torch.FloatTensor(node_n, node_n))
        self.att = Parameter(torch.FloatTensor(0.01 + 0.99 * np.eye(node_n)[np.newaxis, ...]))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # self.att.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_normal(self.weight)
        if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)
            self.bias.data.zero_()

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        # self.act_f = nn.Tanh()
        self.act_f = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):

        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage
        self.input_feature=input_feature
        self.hidden_feature = hidden_feature
        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gc2 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc3 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc4 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc5 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc6 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc7 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc8 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc9 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc10 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc11 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc12 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        # self.gc13 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)

        self.gc13 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.bn13 = nn.BatchNorm1d(node_n * input_feature)

        self.do = nn.Dropout(p_dropout)
        # self.act_f = nn.Tanh()
        self.act_f = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):

        y1 = self.gc1(x)
        b, n, f = y1.shape
        y1 = self.bn1(y1.view(b, -1)).view(b, n, f)
        y1 = self.act_f(y1)
        y1 = self.do(y1)

        y2 = self.gc2(y1)
        y3 = self.gc3(y2)
        y4 = self.gc4(y3)
        y5 = self.gc5(y4)
        y6 = self.gc6(y5)
        y7 = self.gc7(y6)
        y8 = self.gc8(y7 + y6)
        y9 = self.gc9(y8 + y5)
        y10 = self.gc10(y9 + y4)
        y11 = self.gc11(y10 + y3)
        y12 = self.gc12(y11 + y2)

        y13 = self.gc13(y12 + y1)

        # y14 = self.gc14(y13)


        return y13 + x



