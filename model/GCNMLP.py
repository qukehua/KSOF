import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
# from model import GCN as GCN
from model import SRGCN as GCN
from model import KTCN as TCN
from timm.models.layers import DropPath

'''
    MLP与GCN各一路
'''

class SELayer(nn.Module):
    def __init__(self, c, r=4, use_max_pooling=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, s, h = x.shape
        y = self.squeeze(x).view(bs, s)
        y = self.excitation(y).view(bs, s, 1)
        return x * y.expand_as(x)


def mish(x):
    return (x * torch.tanh(F.softplus(x)))


def topk_pool(input_tensor, k):


    pooled_tensor, _ = torch.topk(input_tensor, k, dim=1, largest=True, sorted=False)


    return pooled_tensor



class TGP_BLOCK(nn.Module):
    def __init__(self, opt):

        super(TGP_BLOCK, self).__init__()
        self.input_feature = 2*opt.inpit_n
        self.output_feature = opt.inpit_n
        self.output_n = opt.output_n
        self.act1 = nn.ReLU()
        # self.act1 = mish()
        self.act2 = nn.ReLU()
        # self.act2 = mish()
        self.act3 = nn.ReLU()
        # self.act3 = mish()
        self.conv1 = nn.Conv1d(in_channels= self.input_feature, out_channels=self.output_feature, kernel_size=7, stride=1 , padding=3)
        self.conv2 = nn.Conv1d(in_channels=self.output_feature, out_channels=self.output_feature, kernel_size=5,
                               stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=self.output_feature, out_channels=self.output_feature, kernel_size=3,
                               stride=1, padding=1)
        self.mlp = nn.Linear(66+33+16+8,self.output_feature)
    def forward(self,x):
        x1 = self.act1(x)
        x1 = self.conv1(x1)      #(16,10,66)
        y1 = topk_pool(x1.permute(0,2,1),k= 33)     #(16,33,10)
        x2 = self.act2(x1)
        x2 = self.conv2(x2)
        y2 = topk_pool(x2.permute(0,2,1),k= 16)     #(16,16,10)
        x3 = self.act3(x2)
        x3 = self.conv3(x3)
        y3 = topk_pool(x3.permute(0,2,1),k= 8)      #(16,8,10)
        y= torch.cat(x,y1,y2,y3, dim=1)
        y = self.mlp(y.permute(0,2,1))            #(16,10,66)

        return y





def gen_velocity(m):
    input = [m[:, 0, :] - m[:, 0, :]]  # 差值[16  66]

    for k in range(m.shape[1] - 1):
        input.append(m[:, k + 1, :] - m[:, k, :])
    input = torch.stack((input)).permute(1, 0, 2)  # [16 35 66]

    return input

def gen_acceleration(m):
    velocity = [m[:, k + 1, :] - m[:, k, :] for k in range(m.shape[1] - 1)]
    acceleration = [velocity[0] - velocity[0]]  # 初始化为0

    for k in range(len(velocity) - 1):
        acceleration.append(velocity[k + 1] - velocity[k])
    acceleration = torch.stack((acceleration))#.permute(1, 0, 2)

    return acceleration

def delta_2_gt(prediction, last_timestep):
    prediction = prediction.clone()

    # print (prediction [:,0,:].shape,last_timestep.shape)
    prediction[:, 0, :] = prediction[:, 0, :] + last_timestep
    for i in range(prediction.shape[1] - 1):
        prediction[:, i + 1, :] = prediction[:, i + 1, :] + prediction[:, i, :]

    return prediction






class GCN_TCN(nn.Module):

    def __init__(self, opt):

        super().__init__()
        self.input_feature = opt.input_feature
        self.hidden_Spatial = opt.hidden_Spatial
        self.hidden_Temporal = opt.hidden_Temporal
        self.hidden_GCN = opt.hidden_gcn
        self.drop_out = opt.drop_out
        self.num_channels = opt.num_channels
        self.kernal_size = opt.kernal_size
        self.input_n = opt.input_n
        self.output_n = opt.output_n
        self.num_gcn = opt.num_tcn
        self.num_gcn = opt.num_gcn
        self.node_n = opt.node_n
        self.cat_feature = opt.cat_feature


        self.POSEembedding0 = nn.Conv2d(in_channels= self.node_n,
                           out_channels=self.node_n,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.POSEembedding1 = nn.Conv2d(in_channels=self.input_n,
                           out_channels=30,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        self.se = SELayer(10)
        self.TCN = TCN.TemporalConvNet(self.input_n,  self.num_channels, self.kernal_size, self.drop_out)

        self.GCN = GCN.GCN(input_feature=self.input_feature, hidden_feature=self.hidden_GCN,
                            p_dropout= self.drop_out,num_stage=11, node_n=self.node_n)   #[input_feature = 66]

        self.linear0 = nn.Linear(2 * self.output_n, self.output_n)
        self.linear1 = nn.Linear(self.cat_feature, self.node_n)


        self.final = nn.Linear(30, self.output_n)



    def forward(self, x):
        b, n, f = x.shape                             #[16, 10, 66]

        x_gcn = x.clone()
        x_tcn = gen_velocity(x.clone())


        x_gcn = self.POSEembedding0(x_gcn.permute(0, 2, 1))  # 空间维度嵌入
        x_tcn = self.POSEembedding1(x_tcn)  # 时间维度嵌入
        for i in range(self.num_tcn):
            y_tcn = self.TCN(x_tcn)

        y_tcn = self.final(y_tcn.permute(0, 2, 1)).permute(0, 2, 1)
        y_tcn = delta_2_gt(y_tcn, x[: , -1, :])
        y_tcn = self.se(y_tcn)

        y_gcn = self.GCN(x_gcn).permute(0, 2, 1)
        y_gcn = self.se(y_gcn)

        y = torch.cat([y_tcn, y_gcn], 1)        #(16,20,66)

        y = TGP_BLOCK(y)

        # y = self.linear0(y.permute(0, 2, 1)).permute(0, 2, 1)    #(16,10,66)


        return y, y_gcn, y_tcn