#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import json
import os
import torch
import pandas as pd
import numpy as np



def save_npy_log(opt, path, value, is_create=False, file_name='walking'):

    path_npy = os.path.join(opt.point_npy, path)  # 对预测或实际输出创建一个路径
    if not os.path.exists(path_npy):
        os.makedirs(path_npy)

    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    file_path = path_npy + '/{}.npy'.format(file_name)
    # print(file_path)
    np.save(file_path, value)
 



def save_csv_log(opt, head, value, is_create=False, file_name='test'):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    file_path = opt.ckpt + '/{}.csv'.format(file_name)
    print(file_path)
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, 'a') as f:
            df.to_csv(f, header=False, index=False)

def save_csv_eval_log(opt, head, value, is_create=False, file_name='test'):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    test_sample_num = opt.test_sample_num
    if test_sample_num == -1:
        test_sample_num = 'all'
    file_path = opt.ckpt + '/{}_{}_eval.csv'.format(file_name, test_sample_num)
    print(file_path)
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, 'a') as f:
            df.to_csv(f, header=False, index=False)

def save_ckpt(state, is_best=True, file_name=['ckpt_best.pth.tar', 'ckpt_last.pth.tar'], opt=None):
    file_path = os.path.join(opt.ckpt, file_name[1])
    torch.save(state, file_path)
    if is_best:
        file_path = os.path.join(opt.ckpt, file_name[0])
        torch.save(state, file_path)


def save_options(opt):
    with open(opt.ckpt + '/option.json', 'w') as f:
        f.write(json.dumps(vars(opt), sort_keys=False, indent=4))
