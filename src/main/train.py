'''
@Project ：DHGA 
@File    ：train.py.py
@IDE     ：PyCharm 
@Author  ：WXL
@Date    ：2026/4/25
'''

import os
import torch

# add main,src,root to code
import os.path as osp
import sys

DIR = osp.realpath(osp.dirname(__file__))
sys.path.insert(0, DIR)
sys.path.insert(0, osp.join(DIR, '../'))
sys.path.insert(0, osp.join(DIR, '../../'))

import argparse
import yaml
from utils.logger import config_logging
import tensorboardX

from trainer import Trainer
from src.dataset.dataset_utils import make_dataset

def _conf_file_type(name):
    path = osp.join(DIR, '../../conf', name)
    if not osp.isfile(path):
        raise argparse.ArgumentTypeError(f'conf file not found: {path}')
    return path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=('train', 'test'), default='train')
    parser.add_argument("--gpuid", type=str, required=True, help="gpu id")
    parser.add_argument("--conf_file", type=_conf_file_type, required=True, help="yaml config filename under conf/")
    parser.add_argument("--resume", action='store_true', help="resume training from model.resume_from")
    parser.add_argument("--checkpoint", default=None, help="checkpoint file used in test mode")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # get settings and conf
    args = get_args()

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    # parse the .yaml conf file (args.conf_file is already the resolved path)
    with open(args.conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    conf['mode'] = args.mode
    conf['gpuid'] = args.gpuid
    conf['resume'] = args.resume
    conf['distributed'] = False
    if args.checkpoint is not None:
        conf['model']['resume_from'] = args.checkpoint

    conf['device']=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('GPU not available!' if not torch.cuda.is_available() else 'GPU is available!')

    if conf['mode'] == 'train':
        msg = (f"Task={conf['task']['name']} "
               f"mode={conf['mode']} "
               f"gpu={conf['gpuid']} "
               f"resume={conf['resume']} "
               f"distributed={conf['distributed']} "
               f"epochs={conf['train']['num_epochs']} "
               f"bs={conf['train']['batch_size']} "
               f"lr={conf['train']['lr']}")
    else:
        msg = (f"Task={conf['task']['name']} mode={conf['mode']} gpu={conf['gpuid']} "
               f"bs={conf.get('evaluate', {}).get('batch_size', 'NA')}")

    if not os.path.exists(conf['task']['work_dir']):
        os.makedirs(conf['task']['work_dir'])

    print("#" * 30)
    print(msg)
    print("#" * 30)


    logg_dir=os.path.join(conf['task']['work_dir'],"log" ,conf['task']['name']+"_"+conf['mode'])
    tb_dir=os.path.join(conf['task']['work_dir'], 'tb', conf['task']['name']+'_'+conf['mode'])
    model_save_dir=os.path.join(conf['task']['work_dir'], 'checkpoints', conf['task']['name']+'_'+conf['mode'])
    os.makedirs(logg_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    conf['model_save_dir']=model_save_dir

    # logger
    logger = config_logging(logg_dir)
    conf['logger']=logger

    logger.info(msg)

    # tensorboard
    tb_writer = tensorboardX.SummaryWriter(tb_dir)
    conf["tb_writer"] = tb_writer

    # launch Trainer
    if conf['mode'] == 'train':
        trainer=Trainer(conf)
        trainer.train()

    elif conf['mode'] == 'test':
        resume_from = conf.get('model', {}).get('resume_from', None)
        if resume_from in (None, 'None', ''):
            print("test mode requires --checkpoint or model.resume_from in conf/config.yaml")
            exit()
        make_dataset(conf, phase='train', data_sources=conf['dataset']['train_sources'])
        evaluator=Trainer(conf)
        evaluator.evaluate_checkpoints()

    else:
        print("unsupported mode: {}".format(conf['mode']))
        print("supported modes: train, test")
        exit()
















