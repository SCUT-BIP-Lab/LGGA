# -*- coding: utf-8 -*-
import random
import torch
import numpy as np
import os

def set_random_seed(seed=1314):
    # pass
    # 固定随机参数
    random.seed(seed)
    np.random.seed(seed)

    # os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
    #
    #
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    #
    #
    # torch.backends.cudnn.benchmark = False  # 卷积算法机制
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    # torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.use_deterministic_algorithms(True)  # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法

    # 新框架可复现性已验证 cudnn需要deterministic