#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pprint import pprint
from utils import log
import sys


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--cuda_idx', type=str, default='cuda:0', help='cuda idx')
        self.parser.add_argument('--data_dir', type=str,
                                 default='/data/user/gbx/data/3DPW/sequenceFiles',
                                 help='path to dataset')
        self.parser.add_argument('--rep_pose_dir', type=str,
                                 default='./rep_pose/rep_pose.txt', help='path to dataset')
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--is_eval', dest='is_eval', action='store_true',
                                 help='whether it is to evaluate the model')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/3DPW', help='path to save checkpoint')
        self.parser.add_argument('--skip_rate', type=int, default=1, help='skip rate of samples')
        self.parser.add_argument('--skip_rate_test', type=int, default=1, help='skip rate of samples for test')
        self.parser.add_argument('--extra_info', type=str, default='', help='extra information')

        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--input_feature', type=int, default=128, help='size of pose Embedding layer')
        self.parser.add_argument('--num_gcn', type=int, default=4, help='number of GCN ')
        self.parser.add_argument('--hidden_gcn', type=int, default=256, help='number of GCN hidden features')
        self.parser.add_argument('--mask_ratio', type=int, default=0.16, help='ratio of joints mask')
        self.parser.add_argument('--node_n', type=int, default=69, help='number of GCN nodes')
        self.parser.add_argument('--drop_out', type=float, default=0.5, help='drop out probability')
        self.parser.add_argument('--num_mlp', type=int, default=3, help='number of GraphMLP')
        self.parser.add_argument('--hidden_Spatial', type=int, default=128, help='number of SpatialMLP')
        self.parser.add_argument('--hidden_Temporal', type=int, default=128, help='number of TemporalMLP')
        self.parser.add_argument('--activation', type=str, default='mish', help='the activate funtion')
        self.parser.add_argument('--initialization', type=str, default='none',
                                 help='none, glorot_normal, glorot_uniform, hee_normal, hee_uniform')
        self.parser.add_argument('--num_se', type=int, default=4, help='number of SE block')
        self.parser.add_argument('--use_max_pooling', type=bool, default=False, help='use max pooling')
        self.parser.add_argument('--dct_n', type=int, default=15, help='use max pooling')
        self.parser.add_argument('--J', type=int, default=1, help='use max pooling')
        self.parser.add_argument('--sample_rate', type=int, default=2, help='frame sampling rate')
        self.parser.add_argument('--actions', type=str, default='all', help='path to save checkpoint')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--pre_train', type=bool, default=False, help='pre-train or not')
        self.parser.add_argument('--rep_pose_size', type=int, default=200, help='rep_pose_size')
        self.parser.add_argument('--updata_rate', type=float, default=0.3, help='rep pose updata_rate')
        self.parser.add_argument('--input_n', type=int, default=10, help='past frame number')
        self.parser.add_argument('--output_n', type=int, default=15, help='future frame number')  # 10为短时，25为长时
        self.parser.add_argument('--lr_now', type=float, default=0.001)  # 学习率
        self.parser.add_argument('--max_norm', type=float, default=10000)
        self.parser.add_argument('--epoch', type=int, default=100)  # epoch数
        self.parser.add_argument('--batch_size', type=int, default=32)  # 16 for multistage model
        self.parser.add_argument('--test_batch_size', type=int, default=32)  # 16 for multistage model
        self.parser.add_argument('--is_load', dest='is_load', action='store_true',
                                 help='whether to load existing model')
        self.parser.add_argument('--test_sample_num', type=int, default=-1,
                                 help='the num of sample, '                                                                          'that sampled from test dataset'
                                      '{8,256,-1(all dataset)}')

        self.parser.add_argument('--lr_decay', type=int, default=2, help='every lr_decay epoch do lr decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.96)

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self, makedir=True):
        self._initial()
        self.opt = self.parser.parse_args()

        # if not self.opt.is_eval:
        script_name = os.path.basename(sys.argv[0])[:-3]
        if self.opt.test_sample_num == -1:
            test_sample_num = 'all'
        else:
            test_sample_num = self.opt.test_sample_num

        if self.opt.test_sample_num == -2:
            test_sample_num = '8_256_all'

        log_name = 'MixerGCN_{}_in{}_out{}_num{}_hf{}_hg{}_hs{}_ht{}_lr{}'.format(test_sample_num,
                                                                                  self.opt.input_n,
                                                                                  self.opt.output_n,
                                                                                  self.opt.num_mlp,
                                                                                  self.opt.input_feature,
                                                                                  self.opt.hidden_gcn,
                                                                                  self.opt.hidden_Spatial,
                                                                                  self.opt.hidden_Temporal,
                                                                                  self.opt.lr_now)
        self.opt.exp = log_name
        # do some pre-check
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if makedir==True:
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
                log.save_options(self.opt)
            self.opt.ckpt = ckpt
            log.save_options(self.opt)

        self._print()
        # log.save_options(self.opt)
        return self.opt
