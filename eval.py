#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import logging
import argparse
import paddle
import paddle.nn as nn

import warnings
warnings.filterwarnings('ignore')

from utils.datasets import get_data, get_dataloader
from utils.tools import set_random_seed, compress, calculate_map, calculate_acc
from utils.lr_scheduler import DecreaseLRScheduler
from models import GreedyHash

ckp_list = {
    12:"output/bit_12.pdparams",
    24:"output/bit_24.pdparams",
    32:"output/bit_32.pdparams",
    48:"output/bit_48.pdparams"
}

def get_arguments():
    parser = argparse.ArgumentParser(description='GreedyHash')
    # normal settings
    parser.add_argument('--model', type=str, default="GreedyHash")
    parser.add_argument('--seed', type=int, default=2000, help="NOTE: IMPORTANT TO REPRODUCE THE RESULTS!")
    parser.add_argument('--bit', type=int, default=48,
                help="choose the model of certain bit type", choices=[12, 24, 32, 48])

    # data settings
    parser.add_argument('--dataset', type=str, default="cifar10-1")
    parser.add_argument('--n_class', type=int, default=10)
    parser.add_argument('--topK', type=int, default=-1)
    parser.add_argument('--crop_size', type=int, default=224)

    # test settings
    parser.add_argument('-bs', '--batch-size', type=int, default=32, help='batch_size')
    parser.add_argument('--log_path', type=str, default="logs/")
    parser.add_argument('--pretrained', type=str, default=None,
                help='If pretrained is None, model load from ckp_list')
    arguments = parser.parse_args()
    return arguments

def get_logger(filename, logger_name=None):
    """set logging file and format
    Args:
        filename: str, full path of the logger file to write
        logger_name: str, the logger name, e.g., 'master_logger', 'local_logger'
    Return:
        logger: python logger
    """
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m%d %I:%M:%S %p")
    # different name is needed when creating multiple logger in one process
    logger = logging.getLogger(logger_name)
    fh = logging.FileHandler(os.path.join(filename))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    return logger

def val(model, test_loader, database_loader):
    acc = calculate_acc(test_loader, model)
    retrievalB, retrievalL, queryB, queryL = compress(database_loader, test_loader, model)
    mAP = calculate_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL)
    return mAP, acc

def main():
    config = get_arguments()
    set_random_seed(config.seed)
    mode = "eval"

    log_path = '{}/{}-{}'.format(config.log_path, mode, time.strftime('%Y%m%d-%H-%M-%S'))
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    master_logger = get_logger(
            filename=os.path.join(log_path, 'log.txt'),
            logger_name='master_logger')

    _, test_dataset, database_dataset = get_data(config)
    test_loader = get_dataloader(config=config,
                                 dataset=test_dataset,
                                 mode='test')
    database_loader = get_dataloader(config=config,
                                     dataset=database_dataset,
                                     mode='test')
    model = GreedyHash(config.bit, config.n_class)
    master_logger.info(f'{config}')

    if config.pretrained is not None and os.path.isfile(config.pretrained):
        model_state = paddle.load(config.pretrained)
        master_logger.info(
                "----- Pretrained: Load model state from {}".format(config.pretrained))
    else:
        ckp = ckp_list[config.bit]
        assert os.path.isfile(ckp), "{} doesn't exist!".format(ckp)
        master_logger.info(
                "----- Pretrained: Load model state from {}".format(ckp))
        model_state = paddle.load(ckp)
    model.set_dict(model_state)
    model.eval()
    mAP, acc = val(model, test_loader, database_loader)
    master_logger.info("EVAL-{}, bit:{}, dataset:{}, MAP:{:.3f}".format(
        config.model, config.bit, config.dataset, mAP))

if __name__ == "__main__":
    main()